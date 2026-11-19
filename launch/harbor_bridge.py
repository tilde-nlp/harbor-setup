#!/usr/bin/env python3
"""
The Harbor bridge for sandboxes on Tilde (harbor_env.sandboxes: tilde): marin-harbor's server.py, unchanged, with
one addition: every request must start with the run's secret, http://<host>:<port>/<token>/<bridge path>.

Why: the bridge listens on the Slingshot network of a compute node, which every LUMI job can reach, and a
bridge job is a shell command run as root, with internet, in a container on Tilde. The trials need no change
for this: Harbor's apptainer environment appends its paths to APPTAINER_BRIDGE_URL, so the token travels in that
URL (launch/job.sbatch keeps it in <run dir>/bridge.url, readable only by you). Without the token: 404.

Started by launch/job.sbatch (role batch) on the batch host, with cray-python (stdlib only, Python >= 3.6):
    python3 launch/harbor_bridge.py --apptainer-dir <marin-harbor>/src/harbor/environments/apptainer \
        --port 9910 --token-file <run dir>/bridge.token
"""
import argparse
import hmac
import os
import sys
import threading


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--apptainer-dir", required=True, help="marin-harbor's dir with server.py (harbor_env.apptainer_dir)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--token-file", required=True, help="file with the secret, one line")
    a = p.parse_args()

    with open(a.token_file) as f:
        token = f.read().strip()
    if len(token) < 16:
        sys.exit("harbor_bridge.py: the token in {} is too short".format(a.token_file))
    sys.path.insert(0, a.apptainer_dir)
    import server as bridge   # marin-harbor's server.py (stdlib only)

    class TokenHandler(bridge.BridgeHandler):
        def _authorized(self):
            first, _, rest = self.path[1:].partition("/")
            if hmac.compare_digest(first.encode(), token.encode()):
                self.path = "/" + rest
                return True
            self._json({"error": "not found"}, 404)
            return False

        def do_GET(self):
            if self._authorized():
                bridge.BridgeHandler.do_GET(self)

        def do_POST(self):
            if self._authorized():
                bridge.BridgeHandler.do_POST(self)

    threading.Thread(target=bridge.cleanup_loop, daemon=True).start()
    server = bridge.ThreadedHTTPServer((a.host, a.port), TokenHandler)
    print("Harbor bridge ({}) on {}:{}, requests need the token from {}".format(
        os.path.join(a.apptainer_dir, "server.py"), a.host, a.port, a.token_file), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
