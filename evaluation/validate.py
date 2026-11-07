#!/usr/bin/env python3
"""Offline FORMAT validation of a LIVE/FINAL submission (no golden needed).

    python3 validate.py SUBMISSION.csv.gz --phase live
    python3 validate.py SUBMISSION.csv.gz --phase final --full-items     # also look up items in train/ (slower)

Hard errors (exit 1) = exactly what the server rejects: the OFFICIAL scorer (data/starter/bankrec_score.py) is run against
a dummy golden made of the phase bank lines with no allocations, so its validation pass is byte-for-byte the server's.
Warnings (exit 0, but read them) = things the server accepts but that cost points or smell like bugs:
missing bank lines, unknown item ids, item issued after the bank line, amount > item total, exception/category
combinations that never occur in the golden, >2 decimals, size close to the limits.
"""
import argparse
import csv
import gzip
import io
import os
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import DEFAULT_DATA, EXC, import_scorer, load_categories, ropen, parse_submission  # noqa: E402


def load_items(data, folders):
    items = {}
    for ph in folders:
        for name, tot in (("ar_invoices", "total_amount"), ("ap_invoices", "total_amount"), ("credit_notes", "total_amount"),
                          ("open_items_at_start", "total_amount")):
            p = os.path.join(data, ph, f"{name}.csv.gz")
            if not os.path.exists(p):
                continue
            with ropen(p) as fh:
                for r in csv.DictReader(fh):
                    items[r["item_id"]] = (r.get("issue_date", ""), float(r[tot] or 0), r.get("currency", ""))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--phase", default="live", choices=["live", "final", "train", "sample"])
    ap.add_argument("--full-items", action="store_true", help="also load train/ items (closed items, duplicates, returns)")
    ap.add_argument("--show", type=int, default=5, help="examples per warning")
    a = ap.parse_args()
    S = import_scorer(a.data)
    folder = os.path.join(a.data, a.phase)
    raw = open(a.submission, "rb").read()
    print(f"file {a.submission}: {len(raw) / 1e6:.2f} MB on disk")
    bank = {}
    with ropen(os.path.join(folder, "bank_transactions.csv.gz")) as fh:
        for r in csv.DictReader(fh):
            bank[r["bank_txn_id"]] = (r["booking_date"], float(r["amount"]), r["currency"])
    # ---- hard validation with the official code path
    with tempfile.TemporaryDirectory() as tmp:
        tp, gp = os.path.join(tmp, "t.csv"), os.path.join(tmp, "g.csv")
        with open(tp, "w") as f:
            f.write("bank_txn_id\n" + "\n".join(bank) + "\n")
        with open(gp, "w") as f:
            f.write("bank_txn_id,item_id,amount,category,exception\n")
        g = S.load_golden(gp, tp, EXC, load_categories(a.data))
    res = S.score_submission(g, raw, os.path.basename(a.submission))
    if not res["ok"]:
        print(f"REJECTED ({len(res['errors'])} errors shown, max {S.MAX_ERRORS}):")
        for e in res["errors"]:
            print(f"  row {e['row']}: {e['message']}")
        sys.exit(1)
    m = res["metrics"]
    print(f"OK: official validation passed. rows={m['rows']} lines covered={m['txns_covered']}/{m['txns_total']}")
    # ---- soft checks
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    first = data[:4096].decode("utf-8-sig", "ignore").splitlines()[0]
    delim = ";" if first.count(";") >= 4 and first.count(",") < 4 else ","
    pred = parse_submission(data, delim)
    W = Counter()
    ex = {}

    def warn(kind, msg):
        W[kind] += 1
        if W[kind] <= a.show:
            ex.setdefault(kind, []).append(msg)

    if len(data) > 0.9 * S.MAX_UNCOMPRESSED:
        warn("size", f"{len(data) / 1e6:.1f} MB uncompressed (limit 60)")
    missing = [t for t in bank if t not in pred]
    for t in missing:
        warn("missing_bank_line (counts as wrong)", t)
    folders = (["train"] if a.full_items else []) + {"live": ["live"], "final": ["live", "final"], "train": ["train"],
                                                       "sample": ["sample"]}[a.phase]
    items = load_items(a.data, folders)
    print(f"loaded {len(items)} known items from {folders}")
    use_cnt = Counter()
    n_item_rows = 0
    for t, p in pred.items():
        d, bamt, bcur = bank[t]
        if p["noitem"] is not None:
            cat, e = p["noitem"]
            if not cat:
                warn("noitem_row_without_category (always wrong)", t)
            if e and e != "UNIDENTIFIED":
                warn("noitem_exception_not_UNIDENTIFIED (never in golden)", f"{t} {e}")
            if (cat == "UNIDENTIFIED") != (e == "UNIDENTIFIED"):
                warn("UNIDENTIFIED_category_exception_mismatch", f"{t} cat={cat} exc={e}")
            if cat == "UNIDENTIFIED" and bamt < 0:
                warn("UNIDENTIFIED_on_debit", f"{t} {bamt}")
        for item, (amt, e) in p["items"].items():
            n_item_rows += 1
            use_cnt[item] += 1
            if e == "UNIDENTIFIED":
                warn("item_row_with_UNIDENTIFIED (never in golden)", f"{t} {item}")
            if e == "FX_DIFFERENCE" and item[:3] in ("ARC", "APC"):
                warn("FX_DIFFERENCE_on_credit_note (policy: stays empty)", f"{t} {item}")
            if round(amt, 2) != amt:
                warn("amount_more_than_2_decimals", f"{t} {item} {amt}")
            if amt == 0:
                warn("zero_amount", f"{t} {item}")
            it = items.get(item)
            if it is None:
                warn("unknown_item_id", f"{t} {item}")
                continue
            idate, tot, icur = it
            if idate and idate > d:
                warn("item_issued_after_bank_line", f"{t} {d} {item} issued {idate}")
            if amt > abs(tot) * 1.005 + 0.01 and e != "DUPLICATE_PAYMENT":
                warn("amount_exceeds_item_total", f"{t} {item} {amt} > {tot}")
    for item, n in use_cnt.items():
        if n > 3:
            warn("item_used_on_>3_lines", f"{item} x{n}")
    exc_cnt = Counter(e for p in pred.values() for (_, e) in p["items"].values() if e)
    exc_cnt.update(p["noitem"][1] for p in pred.values() if p["noitem"] and p["noitem"][1])
    cat_cnt = Counter(p["noitem"][0] for p in pred.values() if p["noitem"])
    print(f"item rows={n_item_rows}  lines with items={sum(1 for p in pred.values() if p['items'])}  "
          f"category lines={sum(cat_cnt.values())}")
    print("exceptions: " + ", ".join(f"{k}={v}" for k, v in exc_cnt.most_common()))
    absent = [c for c in EXC if c not in exc_cnt]
    if absent:
        print(f"  note: never predicted {absent} -> each is a 0-F1 class in macro-F1 if present in golden")
    print("categories: " + ", ".join(f"{k}={v}" for k, v in cat_cnt.most_common()))
    if W:
        print("WARNINGS:")
        for k, n in W.most_common():
            print(f"  {k}: {n}")
            for s in ex.get(k, []):
                print(f"      e.g. {s}")
    else:
        print("no warnings")


if __name__ == "__main__":
    main()
