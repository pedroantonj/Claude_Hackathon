#!/usr/bin/env python3
"""Bank reconciliation pipeline (Talky x Anthropic hackathon).

    python3 main.py --phase live --out outputs/submission_live.csv.gz
    python3 main.py --phase train --from 2025-01-01 --to 2025-06-30 --out outputs/dev_2025H1.csv.gz   # dev window
    python3 main.py --phase sample --out outputs/sample.csv.gz                                          # smoke test

Writes the submission CSV, an audit trail (one row per bank line: rule, confidence, note), an LLM review queue for
low-confidence lines and a short audit report (markdown).
"""
import argparse
import csv
import gzip
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reproducibility: set iteration order must not depend on the hash seed -> re-exec with a fixed seed.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from data_loader.history import build_context  # noqa: E402
from matching.engine import run  # noqa: E402
from agents.reporting import write_audit  # noqa: E402

HEADER = ["bank_txn_id", "item_id", "amount", "category", "exception"]


def write_submission(S, bank, path):
    n = 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        for tid in sorted(bank["bank_txn_id"]):
            rows = S.out.get(tid) or [("", "", "UNIDENTIFIED", "UNIDENTIFIED")]
            items = [r for r in rows if r[0]]
            if items:
                rows = items  # never mix item rows with a category row
            else:
                rows = rows[:1]
            for it, amt, cat, exc in rows:
                w.writerow([tid, it, amt if it else "", "" if it else cat, exc])
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="live", choices=["sample", "train", "live", "final"])
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.phase == "train" and not (a.start and a.end):
        ap.error("train requires --from/--to (dev window)")
    out = a.out or os.path.join(HERE, "outputs", f"submission_{a.phase}.csv.gz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    t0 = time.time()
    ctx = build_context(a.phase, a.start, a.end)
    print(f"[main] context: {len(ctx['bank']):,} bank lines, {len(ctx['items']):,} items ({time.time() - t0:.1f}s)")
    S, bank = run(ctx)
    n = write_submission(S, bank, out)
    print(f"[main] wrote {n:,} rows -> {out} ({time.time() - t0:.1f}s)")
    write_audit(S, bank, out)


if __name__ == "__main__":
    main()
