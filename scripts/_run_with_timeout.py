"""Run a command with a wall-clock limit (macOS has no `timeout`). Usage: python _run_with_timeout.py SECONDS cmd args...
Exit code: the command's own, or 124 if the limit was hit.

The command runs in its own process group and the whole group is killed on timeout. Killing only the direct child is
not enough: a wrapper such as `uv run` exits while its grandchild (e.g. the kaggle CLI) keeps running and holds the
output pipe open, so the caller never sees the command end.
"""
import os
import signal
import subprocess
import sys

seconds = int(sys.argv[1])
proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    sys.exit(proc.wait(timeout=seconds))
except subprocess.TimeoutExpired:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            continue
    print(f"timed out after {seconds}s (process group killed)", flush=True)
    sys.exit(124)
