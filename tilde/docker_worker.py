#!/usr/bin/env python3
"""
Harbor sandboxes in Docker on Tilde, for a SkyRL training job on LUMI (configs with harbor_env.sandboxes: tilde).

The LUMI job runs marin-harbor's bridge server (launch/harbor_bridge.py). Its trials use Harbor's `apptainer`
environment, which is only an HTTP client of the bridge: every start / exec / upload / download / stop becomes a
job there. This worker polls the bridge for those jobs and carries them out with the docker CLI, the way Harbor's
own docker environment does. Every connection goes from here to LUMI, through the ssh tunnel that
tilde/harbor_workers.sh keeps open. Same protocol and result fields as marin-harbor's worker.py (apptainer
instances on LUMI nodes), which this replaces.

    start      docker build of the task's environment/ dir (the trial sends it with the request; images are
               kept, tag hb-bridge/env:<hash of the dir>), then docker run <image> sh -c 'sleep infinity'
    exec       docker exec [-w cwd] [-e K=V ...] <container> bash -c <command>
    upload     a tar stream into the container (docker cp -)
    download   a tar stream out of it (docker cp <container>:<path> -)
    stop       docker rm -f

Containers run as Docker runs them: as root (unless the image sets USER), with internet unless the task sets
allow_internet = false (--network none), limited to the task's cpus and memory_mb (--cpus / --memory; off
with --no-resource-limits). What the protocol does not carry: the exec user (the bridge drops it: commands run
as the image's user), prebuilt docker_image tasks and docker-compose tasks (start fails, the trial is masked).
The bridge does not send file modes: files written for a build or an upload are 0755 if they start with "#!",
else 0644 (Harbor's docker environment would copy the host's modes).

Stdlib only, Python >= 3.8. Usually started by tilde/harbor_workers.sh. By hand:
    HB_BRIDGE_URL=http://127.0.0.1:<port>/<token> python3 tilde/docker_worker.py --max-envs 8
    python3 tilde/docker_worker.py --selftest       # checks docker here, without a bridge (pulls ubuntu:24.04)
The bridge URL is read from the environment, not a flag: it holds the run's secret, and other users can read a
process's command line (ps), not its environment.

Built images stay for later trials and runs:  docker image ls --filter label=hb-bridge
    remove them:                              docker image prune -a --filter label=hb-bridge
"""
import argparse
import base64
import hashlib
import io
import json
import os
import posixpath
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

LABEL = "hb-bridge"
MAX_OUTPUT_CHARS = 500_000   # per stream and exec, tail kept (worker.py's default)
LOG_DIRS = ("/logs/agent", "/logs/verifier", "/logs/artifacts")   # what Harbor's docker environment mounts


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def run(cmd, timeout=None, input=None):
    """subprocess.run with bytes output; raises subprocess.TimeoutExpired."""
    return subprocess.run(cmd, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                          stdin=None if input is not None else subprocess.DEVNULL)


def text(b, limit=None):
    s = b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
    return s if limit is None else s[-limit:]


def file_mode(data):
    return 0o755 if data[:2] == b"#!" else 0o644


def one_file_tar(name, data):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name)   # uid/gid 0: root's, like the files of an uploaded directory
        info.size, info.mode, info.mtime = len(data), file_mode(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def as_root(tar_bytes):
    """`docker cp -` keeps the owners in the tar; the client's tar has the LUMI user's uid. Make them root's."""
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as src, tarfile.open(fileobj=out, mode="w") as dst:
        for m in src.getmembers():
            m.uid = m.gid = 0
            m.uname = m.gname = "root"
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
    return out.getvalue()


def strip_top_dir(tar_bytes):
    """`docker cp ctr:/a/dir -` names entries dir/...; the client wants the contents (like tar -C /a/dir .)."""
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as src, tarfile.open(fileobj=out, mode="w") as dst:
        for m in src.getmembers():
            parts = m.name.lstrip("./").split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue   # the directory itself
            m.name = "./" + parts[1]
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
    return out.getvalue()


def image_name(files_b64):
    """hb-bridge/env:<hash of every file of the environment dir>. Tasks with the same environment dir (all of
    reasoning-gym: one Dockerfile) share one image and one build, across tasks and runs."""
    h = hashlib.sha256()
    for rel in sorted(files_b64):
        h.update(rel.encode() + b"\0" + files_b64[rel].encode() + b"\0")
    return "{}/env:{}".format(LABEL, h.hexdigest()[:20])


class Images:
    """docker build of environment dirs, once per image: concurrent starts of one task wait for one build."""

    def __init__(self, staging, timeout):
        self.staging, self.timeout = staging, timeout
        self.locks, self.lock = {}, threading.Lock()

    def get(self, task_name, files_b64):
        """-> (image, built now?)"""
        image = image_name(files_b64)
        with self.lock:
            lock = self.locks.setdefault(image, threading.Lock())
        with lock:
            if run(["docker", "image", "inspect", image], timeout=60).returncode == 0:
                return image, False
            ctx = tempfile.mkdtemp(prefix="hbb-build-", dir=self.staging)
            try:
                for rel, b64 in files_b64.items():
                    path = os.path.normpath(os.path.join(ctx, rel))
                    if not path.startswith(ctx + os.sep):
                        raise RuntimeError("environment file outside the build context: {!r}".format(rel))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    data = base64.b64decode(b64)
                    with open(path, "wb") as f:
                        f.write(data)
                    os.chmod(path, file_mode(data))
                start = time.monotonic()
                try:
                    r = run(["docker", "build", "--label", LABEL + "=1", "-t", image, ctx], timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("docker build of {} took over {} s (--build-timeout)".format(task_name, self.timeout))
                if r.returncode != 0:
                    raise RuntimeError("docker build of {} failed: {}".format(task_name, text(r.stdout + r.stderr, 3000)))
                log("built {} for {} in {:.0f} s".format(image, task_name, time.monotonic() - start))
                return image, True
            finally:
                shutil.rmtree(ctx, ignore_errors=True)


class Sandbox:
    """One trial's container."""

    def __init__(self, env_id, worker):
        self.env_id, self.w = env_id, worker
        self.name = "hbb_" + re.sub(r"[^A-Za-z0-9_.-]", "_", env_id)

    def sh(self, script, timeout=120):
        return run(["docker", "exec", self.name, "sh", "-c", script], timeout=timeout)

    def start(self, p):
        files = p.get("files_b64") or {}
        task = p.get("task_name") or "task"
        if "Dockerfile" not in files:
            raise RuntimeError("task {}: no environment/Dockerfile in the request (prebuilt docker_image tasks are not "
                               "supported over the bridge)".format(task))
        if {"docker-compose.yaml", "docker-compose.yml"} & set(files):
            raise RuntimeError("task {}: docker-compose (multi-container) tasks are not supported over the bridge".format(task))
        image, built = self.w.images.get(task, files)

        cfg = p.get("task_env_config") or {}
        cmd = ["docker", "run", "-d", "--name", self.name, "--label", LABEL + "=1",
               "--label", LABEL + ".run=" + self.w.run_label, "--label", LABEL + ".task=" + task]
        if cfg.get("allow_internet") is False:
            cmd += ["--network", "none"]
        if self.w.limits:
            if cfg.get("memory_mb"):
                cmd += ["--memory", "{}m".format(int(cfg["memory_mb"]))]
            if cfg.get("cpus"):
                cmd += ["--cpus", str(cfg["cpus"])]
        r = run(cmd + [image, "sh", "-c", "sleep infinity"], timeout=300)
        if r.returncode != 0:
            raise RuntimeError("docker run of {} failed: {}".format(image, text(r.stderr, 2000)))
        # Harbor's docker environment bind-mounts these and makes them writable; the trial reads them back.
        dirs = " ".join(LOG_DIRS)
        r = self.sh("mkdir -p {d} && chmod 777 /logs {d}".format(d=dirs), timeout=60)
        if r.returncode != 0:
            raise RuntimeError("cannot create {} in {}: {}".format(dirs, self.name, text(r.stderr, 1000)))
        return {"image": image, "container": self.name, "built": built}

    def exec(self, p):
        cmd = ["docker", "exec"]
        if p.get("cwd"):
            cmd += ["-w", p["cwd"]]
        for k, v in (p.get("env_vars") or {}).items():
            cmd += ["-e", "{}={}".format(k, v)]
        timeout = p.get("timeout_sec") or 600
        try:
            r = run(cmd + [self.name, "bash", "-c", p.get("command") or ""], timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Command timed out after {}s".format(timeout), "return_code": -1}
        out, err = text(r.stdout), text(r.stderr)
        return {"stdout": out[-MAX_OUTPUT_CHARS:], "stderr": err[-MAX_OUTPUT_CHARS:], "return_code": r.returncode,
                "stdout_truncated": len(out) > MAX_OUTPUT_CHARS, "stderr_truncated": len(err) > MAX_OUTPUT_CHARS}

    def upload(self, p):
        data = base64.b64decode(p.get("file_b64") or "")
        target = p.get("target_path") or ""
        if not target.startswith("/"):
            return {"error": "upload target must be an absolute path: {!r}".format(target)}
        if p.get("is_dir"):
            dest, tar_bytes = target, as_root(data)   # the client's tar of the directory's contents
        else:
            dest, tar_bytes = posixpath.dirname(target) or "/", one_file_tar(posixpath.basename(target), data)
        r = self.sh("mkdir -p " + shlex.quote(dest), timeout=60)
        if r.returncode == 0:
            r = run(["docker", "cp", "-", "{}:{}".format(self.name, dest)], input=tar_bytes, timeout=600)
        return {"return_code": r.returncode, "stdout": text(r.stdout)[:4000], "stderr": text(r.stderr)[:2000]}

    def download(self, p):
        src, is_dir = p.get("source_path") or "", bool(p.get("is_dir"))
        r = run(["docker", "cp"] + ([] if is_dir else ["-L"]) + ["{}:{}".format(self.name, src.rstrip("/") or "/"), "-"],
                timeout=600)
        if r.returncode != 0:
            return {"error": "Download failed: " + text(r.stderr)[:500]}
        if is_dir:
            return {"file_b64": base64.b64encode(strip_top_dir(r.stdout)).decode(), "is_dir": True}
        with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
            if len(files) != 1 or any(m.isdir() for m in tar.getmembers()):
                return {"error": "Download failed: {} is not a file".format(src)}
            data = tar.extractfile(files[0]).read()
        return {"file_b64": base64.b64encode(data).decode(), "is_dir": False}

    def stop(self, p=None):
        r = run(["docker", "rm", "-f", self.name], timeout=120)
        if r.returncode != 0 and b"No such container" not in r.stderr:
            return {"error": "docker rm -f {} failed: {}".format(self.name, text(r.stderr, 500))}
        return {"stopped": True}


class Worker:
    def __init__(self, a, bridge_url):
        self.url = bridge_url.rstrip("/")
        self.name, self.max_envs, self.threads, self.poll = a.name, a.max_envs, a.num_workers, a.poll_interval
        self.limits, self.run_label = not a.no_resource_limits, a.run_label
        self.images = Images(a.staging, a.build_timeout)
        self.sandboxes, self.lock = {}, threading.Lock()
        self.jobs = queue.Queue()
        self.busy = 0
        self.counts = {"start": 0, "exec": 0, "upload": 0, "download": 0, "stop": 0, "errors": 0}
        self.stopping = threading.Event()
        self.last_poll_ok = time.monotonic()

    # ---- HTTP to the bridge (through the tunnel)
    def get(self, path, timeout=15):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as resp:
            return json.loads(resp.read())

    def post(self, path, data, timeout=60):
        req = urllib.request.Request(self.url + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def post_result(self, result):
        """Retried for 5 min: a tunnel that reconnects must not lose the result of a finished command."""
        deadline = time.monotonic() + 300
        while True:
            try:
                self.post("/worker/result", result)
                return
            except urllib.error.HTTPError as e:
                if e.code == 404:   # the bridge gave up on this job already (worker timeout): nobody waits for it
                    log("result of {} dropped: the bridge no longer knows the job".format(result["job_id"]))
                    return
                err = e
            except (OSError, ValueError) as e:
                err = e
            if time.monotonic() > deadline or self.stopping.is_set():
                log("result of {} lost: {}".format(result["job_id"], err))
                return
            time.sleep(3)

    # ---- the two loops
    def poll_loop(self):
        """Ask for as many jobs as there are free threads. Polls even with none free: the bridge counts a node
        without a poll for 15 s as dead, and after 60 s fails every job it is running."""
        failing_since = None
        while not self.stopping.is_set():
            free = max(0, self.threads - self.busy - self.jobs.qsize())
            try:
                resp = self.get("/worker/get_jobs?worker_id={}&count={}&capacity={}".format(self.name, free, self.max_envs))
                if failing_since is not None:
                    log("bridge reachable again after {:.0f} s".format(time.monotonic() - failing_since))
                    failing_since = None
                self.last_poll_ok = time.monotonic()
            except (OSError, ValueError) as e:
                if failing_since is None:
                    failing_since = time.monotonic()
                    log("bridge unreachable ({}); retrying every 2 s".format(e))
                time.sleep(2)
                continue
            jobs = resp.get("jobs") or []
            for job in jobs:
                self.jobs.put(job)
            if not jobs:
                time.sleep(self.poll if free else 1.0)

    def work_loop(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            with self.lock:
                self.busy += 1
            result = {"job_id": job["job_id"]}
            try:
                out = self.handle(job["type"], job["env_id"], job.get("payload") or {})
                if isinstance(out, dict) and "error" in out:
                    result["error"] = out["error"]
                else:
                    result["result"] = out
            except Exception as e:   # noqa: BLE001 - every failure goes back to the trial
                result["error"] = "{}: {}".format(type(e).__name__, e)
            finally:
                with self.lock:
                    self.busy -= 1
                    self.counts[job["type"]] = self.counts.get(job["type"], 0) + 1
                    self.counts["errors"] += "error" in result
            if "error" in result:
                log("{} {} failed: {}".format(job["type"], job["env_id"], result["error"][:300]))
            self.post_result(result)

    def handle(self, kind, env_id, payload):
        if kind == "start":
            box = Sandbox(env_id, self)
            try:
                out = box.start(payload)
            except Exception:
                box.stop()
                raise
            with self.lock:
                self.sandboxes[env_id] = box
            log("started {} ({}, {})".format(box.name, payload.get("task_name"), out["image"]))
            return out
        if kind == "stop":
            with self.lock:
                box = self.sandboxes.pop(env_id, None)
            return box.stop() if box else {"stopped": True, "note": "no instance"}
        if kind in ("exec", "upload", "download"):
            with self.lock:
                box = self.sandboxes.get(env_id)
            if box is None:
                return {"error": "No instance for env {}".format(env_id)}
            return getattr(box, kind)(payload)
        return {"error": "unknown job type {!r}".format(kind)}

    def status_loop(self, every):
        while not self.stopping.wait(every):
            with self.lock:
                n, busy, c = len(self.sandboxes), self.busy, dict(self.counts)
            log("status: {}/{} sandboxes, {}/{} threads busy, done: {}".format(
                n, self.max_envs, busy, self.threads, " ".join("{}={}".format(k, v) for k, v in c.items())))

    def remove_containers(self):
        """Every container of this run (label hb-bridge.run), also ones a killed earlier worker left behind."""
        ids = run(["docker", "ps", "-aq", "--filter", "label={}.run={}".format(LABEL, self.run_label)], timeout=60).stdout.split()
        if ids:
            run(["docker", "rm", "-f"] + [i.decode() for i in ids], timeout=300)
            log("removed {} container(s) of run {}".format(len(ids), self.run_label))

    def serve(self):
        self.remove_containers()
        log("worker {}: {} sandboxes max, {} threads, bridge {}".format(
            self.name, self.max_envs, self.threads, re.sub(r"(//[^/]+/).*", r"\1<token>", self.url)))
        for _ in range(self.threads):
            threading.Thread(target=self.work_loop, daemon=True).start()
        threading.Thread(target=self.status_loop, args=(60,), daemon=True).start()
        threading.Thread(target=self.poll_loop, daemon=True).start()
        self.stopping.wait()
        log("stopping: removing this run's containers")
        self.remove_containers()


def selftest(a):
    """One sandbox through every operation, the way a trial uses it, without a bridge."""
    w = Worker(a, "http://unused")
    image = a.selftest_image
    dockerfile = "FROM {}\nWORKDIR /app\nCOPY seed.txt /app/seed.txt\n".format(image)
    files = {"Dockerfile": base64.b64encode(dockerfile.encode()).decode(),
             "seed.txt": base64.b64encode(b"seed\n").decode()}
    tests = io.BytesIO()
    with tarfile.open(fileobj=tests, mode="w") as tar:
        script = b"#!/bin/bash\necho 1 > /logs/verifier/reward.txt\necho tests ran\n"
        info = tarfile.TarInfo("./test.sh")
        info.size, info.mode = len(script), 0o755
        tar.addfile(info, io.BytesIO(script))
    box, ok = Sandbox("env-selftest-" + str(os.getpid()), w), True

    def check(what, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("{} {}{}".format("PASS" if cond else "FAIL", what, "" if cond else ": " + str(detail)[:500]), flush=True)

    try:
        out = box.start({"task_name": "selftest", "files_b64": files,
                         "task_env_config": {"memory_mb": 512, "cpus": 1, "allow_internet": True}})
        check("start (docker build + run)", out.get("container") == box.name, out)
        r = box.exec({"command": "id -u; pwd; cat seed.txt; echo $FOO", "env_vars": {"FOO": "bar baz"}, "timeout_sec": 60})
        check("exec as root in WORKDIR with env", r["return_code"] == 0 and r["stdout"].split("\n")[:4] == ["0", "/app", "seed", "bar baz"], r)
        r = box.exec({"command": "pwd", "cwd": "/tmp", "timeout_sec": 60})
        check("exec with cwd", r["stdout"].strip() == "/tmp", r)
        r = box.exec({"command": "sleep 30", "timeout_sec": 2})
        check("exec timeout", r["return_code"] == -1 and "timed out" in r["stderr"], r)
        r = box.upload({"file_b64": base64.b64encode(tests.getvalue()).decode(), "target_path": "/tests", "is_dir": True})
        check("upload dir", r["return_code"] == 0, r)
        r = box.upload({"file_b64": base64.b64encode(b"#!/bin/sh\necho hi\n").decode(), "target_path": "/opt/x/run.sh"})
        check("upload file", r["return_code"] == 0, r)
        r = box.exec({"command": "/opt/x/run.sh && bash /tests/test.sh > /logs/verifier/test-stdout.txt", "timeout_sec": 60})
        check("run uploaded scripts", r["return_code"] == 0, r)
        r = box.download({"source_path": "/logs/verifier/reward.txt"})
        check("download file", base64.b64decode(r.get("file_b64", "")) == b"1\n", r)
        r = box.download({"source_path": "/logs/verifier", "is_dir": True})
        names = sorted(m.name for m in tarfile.open(fileobj=io.BytesIO(base64.b64decode(r.get("file_b64", "")))).getmembers()) if "file_b64" in r else r
        check("download dir", names == ["./reward.txt", "./test-stdout.txt"], names)
        r = box.download({"source_path": "/nonexistent"})
        check("download of a missing file is an error", "error" in r, r)
        if a.no_resource_limits is False:
            r = run(["docker", "inspect", "-f", "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}", box.name], timeout=60)
            check("resource limits", text(r.stdout).split() == [str(512 * 2 ** 20), str(10 ** 9)], text(r.stdout + r.stderr))
    except Exception as e:   # noqa: BLE001
        check("selftest", False, "{}: {}".format(type(e).__name__, e))
    finally:
        r = box.stop()
        check("stop", r == {"stopped": True}, r)
    print("selftest {}".format("passed" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Bridge URL: environment variable HB_BRIDGE_URL (http://127.0.0.1:<port>/<token>)")
    p.add_argument("--name", default="tilde-" + socket.gethostname().split(".")[0],
                   help="node name shown in the bridge's /status (default: %(default)s)")
    p.add_argument("--max-envs", type=int, default=8, help="sandboxes at once, reported to the bridge as capacity "
                   "(default %(default)s). Size it by CPU and RAM: each one gets the task's cpus and memory_mb")
    p.add_argument("--num-workers", type=int, default=32, help="threads running jobs (default %(default)s): "
                   "builds and verifiers hold one for a long time, execs for a moment")
    p.add_argument("--poll-interval", type=float, default=0.05, help="seconds between polls when idle (default %(default)s)")
    p.add_argument("--build-timeout", type=int, default=1800, help="seconds per docker build (default %(default)s)")
    p.add_argument("--no-resource-limits", action="store_true", help="no --cpus / --memory from the task")
    p.add_argument("--run-label", default="manual", help="label hb-bridge.run of this run's containers; removed at start and exit")
    p.add_argument("--staging", default=tempfile.gettempdir(), help="where build contexts are written (default %(default)s)")
    p.add_argument("--selftest", action="store_true", help="check docker here: one sandbox through every operation, no bridge")
    p.add_argument("--selftest-image", default="ubuntu:24.04", help="base image of the selftest (needs bash)")
    a = p.parse_args()

    if shutil.which("docker") is None:
        sys.exit("docker_worker.py: docker not found")
    if a.selftest:
        a.run_label = "selftest"
        sys.exit(selftest(a))
    url = os.environ.get("HB_BRIDGE_URL", "")
    if not url:
        sys.exit("docker_worker.py: set HB_BRIDGE_URL (tilde/harbor_workers.sh does this)")
    w = Worker(a, url)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: w.stopping.set())
    w.serve()


if __name__ == "__main__":
    main()
