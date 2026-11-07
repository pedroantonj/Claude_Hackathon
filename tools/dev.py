#!/usr/bin/env python3
"""Run pipeline + official scorer on a TRAIN dev window, serialised by a global lock (RAM: ~7 GB per run).

    python3 tools/dev.py --from 2025-01-01 --to 2025-06-30 [--root <copy of bankrec>] [--tag name]
Lock = directory /tmp/bankrec_dev.lock (waits until free). Sample smoke tests don't need the lock:
    python3 main.py --phase sample --out outputs/sample.csv.gz && python3 evaluation/evaluate.py outputs/sample.csv.gz --phase sample
"""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", default="2025-01-01")
ap.add_argument("--to", dest="end", default="2025-06-30")
ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--tag", default="dev")
ap.add_argument("--dump", default=None)
a = ap.parse_args()
LOCK = "/tmp/bankrec_dev.lock"
t0 = time.time()
while True:
    try:
        os.mkdir(LOCK); break
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) > 3600:   # stale lock
            os.rmdir(LOCK); continue
        time.sleep(5)
print(f"[dev] lock acquired after {time.time() - t0:.0f}s", flush=True)
try:
    out = os.path.join(a.root, "outputs", f"{a.tag}_{a.start}_{a.end}.csv.gz")
    t1 = time.time()
    subprocess.run([sys.executable, os.path.join(a.root, "main.py"), "--phase", "train", "--from", a.start, "--to", a.end,
                    "--out", out], check=True)
    print(f"[dev] pipeline {time.time() - t1:.0f}s", flush=True)
finally:
    os.rmdir(LOCK)
cmd = [sys.executable, os.path.join(a.root, "evaluation", "evaluate.py"), out, "--phase", "train",
       "--from", a.start[:7], "--to", a.end[:7],
       "--data", os.environ.get("BANKREC_DATA", os.path.abspath(os.path.join(a.root, "..", "data")))]
if a.dump:
    cmd += ["--dump", a.dump]
subprocess.run(cmd)
