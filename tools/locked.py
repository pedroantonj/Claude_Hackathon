#!/usr/bin/env python3
"""Run any command while holding the global heavy-run lock (/tmp/bankrec_dev.lock), report wall time + peak RSS.

    python3 tools/locked.py -- python3 main.py --phase live --out outputs/submission_live.csv.gz
"""
import os, resource, subprocess, sys, time

LOCK = "/tmp/bankrec_dev.lock"
cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
t0 = time.time()
while True:
    try:
        os.mkdir(LOCK); break
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) > 3600:
            try: os.rmdir(LOCK)
            except OSError: pass
            continue
        time.sleep(5)
print(f"[locked] lock acquired after {time.time() - t0:.0f}s", flush=True)
t1 = time.time()
try:
    rc = subprocess.run(cmd).returncode
finally:
    os.rmdir(LOCK)
ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
peak = ru / 1e9 if sys.platform == "darwin" else ru / 1e6
print(f"[locked] rc={rc} wall={time.time() - t1:.0f}s peak_rss={peak:.2f} GB", flush=True)
sys.exit(rc)
