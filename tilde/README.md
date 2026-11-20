# Harbor sandboxes in Docker on Tilde, training on LUMI

Status (2026-11-19): built, tested only locally against a fake `docker` and `ssh`. Never run on LUMI or real
Docker. This file is the plan to make it work, stage by stage, and what to look at in each stage.
The judge (`configs/harbor/reasoning_gym_tilde_judge.yaml`) comes after stage 4, not before.

## How it fits together

```
 LUMI compute nodes (one Slurm job)                         LUMI login node         Tilde server
 ┌──────────────────────────────────────────────┐
 │ batch node: launch/harbor_bridge.py           │◄── ssh -L tunnel (opened from Tilde) ──┐
 │   = marin-harbor server.py + secret token     │                                        │
 │   listens on <hsn0 IP>:9910                   │                          tilde/harbor_workers.sh
 │        ▲                                      │                            ├─ keeps the tunnel open
 │        │ HTTP: start / exec / upload /        │                            └─ tilde/docker_worker.py
 │        │       download / stop (= "jobs")     │                                 polls the bridge for jobs,
 │ SkyRL trials (Harbor, environment: apptainer, │                                 runs each one with the docker
 │   which is only an HTTP client of the bridge) │                                 CLI, posts the result back
 └──────────────────────────────────────────────┘
```

* Every connection goes from Tilde to LUMI. LUMI never connects to Tilde.
* The trials don't know they run on Docker: Harbor's `apptainer` environment only talks to the bridge.
* The job writes the bridge URL with its token to `<run dir>/bridge.url` (mode 600). `harbor_workers.sh`
  reads it over ssh. The job deletes it when it ends, which is how the Tilde side learns the run is over.
* Before starting Ray, the job waits up to `harbor_env.tilde_wait_min` (30 min) until the workers offer
  `generator.rate_limit.max_concurrency` sandbox slots (the sum of `--max-envs`).

| file | runs on | what it does |
|---|---|---|
| `launch/harbor_bridge.py` | LUMI batch node (started by `launch/job.sbatch`, `start_tilde_bridge`) | marin-harbor's bridge, with every path prefixed by the token |
| `launch/job.sbatch` (`start_tilde_bridge`) | LUMI batch node | starts the bridge, writes `bridge.url`, waits for slots, cancels the job if the bridge dies |
| `launch/render.py`, `submit.sh` | LUMI login node | config checks for `harbor_env.sandboxes: tilde`; no environment-plane job is submitted |
| `tilde/harbor_workers.sh` | Tilde, in tmux | finds the run's bridge, opens and reopens the tunnel, runs the worker, stops when the job ends |
| `tilde/docker_worker.py` | Tilde | the bridge protocol with `docker build/run/exec/cp/rm` |

## Stage 0: get the code to both machines

The code is uncommitted on the dev machine. Commit it on a branch and push. Then:

* **LUMI**: `git pull` in the clone used as `paths.root`. Check that marin-harbor's bridge is there:
  `ls <root>/marin-harbor/src/harbor/environments/apptainer/server.py` (`setup.sh` does not copy it).
* **Tilde**: clone the repo anywhere. Only `tilde/` is used there. You need: docker without sudo
  (`docker info`), python3 >= 3.8, curl, `setsid`, and in `~/.ssh/config` a host `lumi` you can ssh to
  without a prompt once the key is in an agent.

## Stage 1: Docker on Tilde, no LUMI (~1 min)

```
python3 tilde/docker_worker.py --selftest
```

This takes one sandbox through every operation a trial uses: build, run, exec as root, cwd, env, exec
timeout, upload of a file and a dir, download, resource limits, stop. Every line should print `PASS`, and
the last line should be `selftest passed`. It pulls `ubuntu:24.04`, so Tilde needs internet.

If you see `FAIL resource limits`, the Docker daemon can't apply `--cpus/--memory` (cgroups). Use
`--no-resource-limits` on `harbor_workers.sh` later.

## Stage 2: tunnel + token + worker, no GPUs (~10 min)

This checks the network path and the token on a login node before spending GPU hours. In the real run, the
tunnel goes to a compute node's hsn0 IP instead of the login node, but everything else is the same.

On **LUMI** (note which login node you're on: `hostname`):

```
cd <root>
module load cray-python
( umask 077; python3 -c 'import secrets; print(secrets.token_hex(16))' > ~/bridge-test.token )
python3 launch/harbor_bridge.py --host 127.0.0.1 --port 29910 --token-file ~/bridge-test.token \
    --apptainer-dir $PWD/marin-harbor/src/harbor/environments/apptainer
```

On **Tilde** (in tmux). `lumi` may land on any login node, so pin it to the one above. Run the check in the
foreground first: it must print the hostname with no prompt. A new hostname asks to accept its host key, and a
key that is not in the agent asks for its passphrase. A backgrounded ssh that waits for a prompt competes with the
shell for the terminal and makes everything lag.

```
LOGIN=lumi-uan03.csc.fi   # if `hostname` printed uan03; the bare name only resolves inside LUMI, no user@ here
ssh -o HostName=$LOGIN lumi hostname   # prints uan03; answer `yes` to the host key once
# asked for a passphrase? load the key, then repeat the check:  eval $(ssh-agent); ssh-add
ssh -f -N -o ExitOnForwardFailure=yes -o HostName=$LOGIN -L 127.0.0.1:18910:127.0.0.1:29910 lumi
TOKEN=$(ssh -o HostName=$LOGIN lumi cat bridge-test.token)
HB_BRIDGE_URL=http://127.0.0.1:18910/$TOKEN python3 tilde/docker_worker.py --max-envs 2 --run-label bridgetest
```

Back on **LUMI**, in a second shell on the same login node:

```
curl -s http://127.0.0.1:29910/$(cat ~/bridge-test.token)/status | python3 -m json.tool   # nodes: tilde-<host>, active, capacity 2
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:29910/wrong-token/status            # 404
```

Then stop everything: Ctrl-C the worker, `pkill -f 'L 127.0.0.1:18910'` (the tunnel), Ctrl-C the bridge, `rm ~/bridge-test.token`.

**Pass:** the worker log has no `bridge unreachable`, `/status` shows the worker with capacity 2, and a wrong
token gets 404.

## Stage 3: first real run, hello-world tasks (`helloworld_tilde.yaml`, 4 nodes, dev-g)

This is the same training as `configs/harbor/helloworld.yaml` (sandboxes on LUMI). The only difference is
where the sandboxes run. It waits for 8 slots (`max_concurrency: 8`).

1. **LUMI**, check without submitting: `./submit.sh --dry-run configs/harbor/helloworld_tilde.yaml`.
   The summary line should end with `sandboxes: Docker on Tilde`.
2. **LUMI**, submit: `./submit.sh configs/harbor/helloworld_tilde.yaml`. It prints the run dir and the
   exact Tilde command.
3. **Tilde**, right away (the job can start while you sleep; the script waits up to 180 min for it):
   ```
   tmux new -s hb
   eval "$(ssh-agent)"; ssh-add ~/.ssh/<lumi key>
   tilde/harbor_workers.sh --lumi-run <run dir> --max-envs 8
   ```

What you should see, in order:

| where | lines |
|---|---|
| Tilde terminal | `waiting for the bridge of LUMI job <id> (PENDING…)` → `bridge at <ip>:9910` → `tunnel 127.0.0.1:<port> -> …` → `worker running` |
| `<run dir>/slurm-<id>.out` | `harbor bridge: http://<ip>:9910/<token>` → `waiting up to 30 min for Docker workers…` → `Tilde workers: 0/8 …` → `Tilde workers ready: 8 sandbox slots` → Ray / vLLM / training as in a LUMI Harbor run |
| `logs/tilde/<run>/worker.log` (Tilde) | `worker tilde-<host>: 8 sandboxes max…`, one `started …` per trial, every minute `status: n/8 sandboxes, … done: start=… exec=… errors=…` |
| `<run dir>/harbor_bridge-<id>.log` | the bridge's own log |
| `<root>/trials/` (LUMI) | one dir per trial, Harbor's logs: the first place to look when a trial fails |

**Pass:**
* at least one training step completes with non-zero reward (hello-world is easy). Also run `./check.sh run <run dir>`.
* `errors=0` in the worker status, or every error explained (`grep failed worker.log`).
* when the job ends, `harbor_workers.sh` stops by itself and `docker ps -a --filter label=hb-bridge` is empty.
* record for later: step time and generation time (`tools/collect_metrics.py <run dir>`), next to a
  `helloworld.yaml` run on LUMI if there is one. The difference is roughly what the tunnel costs.

## Stage 4: reasoning-gym (`reasoning_gym_tilde.yaml`, verifier reward)

Same steps, with `configs/harbor/reasoning_gym_tilde.yaml` (288 tasks, `max_concurrency: 4`, so the default
`--max-envs 8` is enough). The tasks must be in `<root>/tasks/reasoning-gym-easy` on LUMI. The first trial of
each distinct task environment runs a `docker build`. Images are cached on Tilde (tag
`hb-bridge/env:<hash>`), so later steps and runs skip it.

**Pass:** rewards are in the same range as `reasoning_gym.yaml` on LUMI, and few trials are masked.
Only after this stage: `reasoning_gym_tilde_judge.yaml`.

## When something goes wrong

| symptom | likely cause / where to look |
|---|---|
| `cannot ssh to lumi without a prompt` | key not in this tmux window's agent, or host key not yet accepted (`ssh lumi` once by hand) |
| `tunnel … failed`, `administratively prohibited` in `tunnel.log` | the login node refuses port forwarding |
| `tunnel … failed`, `Connection refused` | the bridge is not up: `<run dir>/harbor_bridge-<id>.log` |
| job dies with `no Tilde workers with N slots after 30 min` | `harbor_workers.sh` was not started in time, or `--max-envs` < `max_concurrency` |
| job dies with `the Harbor bridge exited` | `<run dir>/harbor_bridge-<id>.log` |
| `start … failed` in `worker.log` | `docker build` of the task's `environment/` failed; the error text is in the line |
| many trials masked, worker shows few `exec` | trials can't reach the bridge from inside the job: trial logs under `<root>/trials/` |
| Tilde script killed / Ctrl-C | the job keeps running and its trials fail (masked); start `harbor_workers.sh` again with the same `--lumi-run`. After 30 min without workers the trials raise (`BRIDGE_WORKERS_DEAD_TIMEOUT`, marin-harbor) |

## Known limits (by design, for now)

* The exec user is dropped by the bridge protocol: commands run as the image's user (root unless `USER` is set).
* Tasks with a prebuilt `docker_image` or docker-compose are not supported: start fails and the trial is masked.
* File modes are not sent: uploaded files are 0755 if they start with `#!`, else 0644.
* Security: the token travels as plain HTTP on Slingshot (shared by all LUMI jobs), and anyone holding it gets
  root shell jobs with internet in containers on Tilde. Acceptable while Tilde is effectively single-user.

## Not decided yet

* Which task set actually needs Tilde (root / internet)? reasoning-gym does not. It is only a pipeline check.
* Is a tunnel through a login node acceptable for long runs? Ask LUMI support before multi-day runs.
* Fail faster when Tilde drops? Today trials are masked for up to 30 min while the GPUs keep training.
* After stage 4: cut the code down to what is needed and walk through it (see the plan of 2026-11-19).
