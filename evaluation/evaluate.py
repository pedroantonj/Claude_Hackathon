#!/usr/bin/env python3
"""Offline evaluation harness built on the OFFICIAL scorer (data/starter/bankrec_score.py, imported, not copied).

    python3 evaluate.py SUBMISSION.csv[.gz] --phase sample
    python3 evaluate.py SUBMISSION.csv[.gz] --phase train --from 2025-01 --to 2025-06 --dump /tmp/errors.csv
    python3 evaluate.py SUBMISSION.csv[.gz] --phase train --from 2025-01-01 --to 2025-06-30 --json /tmp/res.json

Window (--from/--to) filters bank lines by booking_date (inclusive; YYYY, YYYY-MM or YYYY-MM-DD prefixes).
The golden and the bank-line universe are restricted to the window; submission rows of bank lines outside the
window are dropped before scoring (reported), so a submission for the whole phase can be scored on any slice.
The headline score/metrics come from bankrec_score.score_submission; per-exception P/R/F1 and the error dump are
diagnostics recomputed here (cross-checked against the official macro-F1).
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict

DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
EXC = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
       "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]


def import_scorer(data):
    sys.path.insert(0, os.path.join(data, "starter"))
    import bankrec_score as S  # noqa: E402
    return S


def ropen(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", newline="")
    return open(path, encoding="utf-8-sig", newline="")


def in_window(d, lo, hi):
    return (not lo or d[:len(lo)] >= lo) and (not hi or d[:len(hi)] <= hi)


def load_categories(data):
    with open(os.path.join(data, "masters", "categories.csv"), encoding="utf-8") as fh:
        return [r["category"] for r in csv.DictReader(fh)]


def build_golden(S, data, phase, lo, hi, tmpdir):
    """Write window-restricted txn list + golden to temp files and load them with the official loader.
    Returns (golden, bank_info{txn: (date, amount, currency, text)})."""
    folder = os.path.join(data, phase)
    info = {}
    txn_path = os.path.join(tmpdir, "txns.csv")
    with ropen(os.path.join(folder, "bank_transactions.csv.gz")) as fh, open(txn_path, "w", newline="") as out:
        out.write("bank_txn_id\n")
        for r in csv.DictReader(fh):
            if in_window(r["booking_date"], lo, hi):
                t = r["bank_txn_id"]
                out.write(t + "\n")
                txt = " | ".join(x for x in (r["text"], r["ref1"], r["ref2"], r["remittance_info"], r["counterparty_name"]) if x)
                info[t] = (r["booking_date"], r["amount"], r["currency"], r["account_id"], txt[:220])
    gold_path = os.path.join(folder, "golden_allocations.csv.gz")
    if lo or hi:
        wpath = os.path.join(tmpdir, "golden.csv")
        with ropen(gold_path) as fh, open(wpath, "w", newline="") as out:
            rd = csv.reader(fh)
            w = csv.writer(out, lineterminator="\n")
            h = next(rd)
            w.writerow(h)
            it = h.index("bank_txn_id")
            for row in rd:
                if row and row[it] in info:
                    w.writerow(row)
        gold_path = wpath
    g = S.load_golden(gold_path, txn_path, EXC, load_categories(data))
    return g, info


def filter_submission(raw, keep):
    """Drop rows whose bank_txn_id is not in `keep`. Keeps the delimiter the scorer would detect."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8-sig")
    lines = text.splitlines(keepends=True)
    first = lines[0] if lines else ""
    delim = ";" if first.count(";") >= 4 and first.count(",") < 4 else ","
    header = [h.strip().lower() for h in next(csv.reader([first], delimiter=delim), [])]
    it = header.index("bank_txn_id") if "bank_txn_id" in header else 0
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
    w.writerow(next(csv.reader([first], delimiter=delim)))
    dropped = 0
    for row in csv.reader(io.StringIO("".join(lines[1:])), delimiter=delim):
        if row and len(row) > it and row[it].strip() and row[it].strip() not in keep:
            dropped += 1
            continue
        w.writerow(row)
    return buf.getvalue().encode("utf-8"), dropped, delim


def parse_submission(data, delim):
    """pred[t] = {"items": {item: (amount, exc)}, "noitem": (cat, exc) or None}. Assumes scorer validated it."""
    text = data.decode("utf-8-sig")
    rd = csv.reader(io.StringIO(text), delimiter=delim)
    h = [x.strip().lower() for x in next(rd)]
    ix = {k: h.index(k) for k in ("bank_txn_id", "item_id", "amount", "category", "exception")}
    pred = defaultdict(lambda: {"items": {}, "noitem": None})
    for row in rd:
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        t = row[ix["bank_txn_id"]].strip()
        item = row[ix["item_id"]].strip()
        exc = row[ix["exception"]].strip().upper()
        if item:
            a = row[ix["amount"]].strip()
            if delim == ";":
                a = a.replace(",", ".")
            pred[t]["items"][item] = (float(a), exc)
        else:
            pred[t]["noitem"] = (row[ix["category"]].strip().upper(), exc)
    return pred


def fmt_items(items):
    return ";".join(f"{i}:{a:.2f}{':' + e if e else ''}" for i, (a, e) in sorted(items.items()))


def diagnose(S, g, pred, info, dump_path):
    tol = S._tol
    tp = Counter(); fp = Counter(); gold = Counter(); npred = Counter()
    err_types = Counter(); err_by_seg = defaultdict(Counter)
    cat_conf = Counter()
    rows_out = []
    for t, (gitems, gcat, gexc, seg) in g.txns.items():
        p = pred.get(t)
        pitems = p["items"] if p else {}
        pno = p["noitem"] if p else None
        # exception counts (mirror of official logic)
        if gitems:
            for _, ge in gitems.values():
                if ge:
                    gold[ge] += 1
        elif gexc:
            gold[gexc] += 1
        for item, (a, e) in pitems.items():
            if e:
                npred[e] += 1
                gi = gitems.get(item)
                if gi is not None and gi[1] == e:
                    tp[e] += 1
                else:
                    fp[e] += 1
        if pno and pno[1]:
            npred[pno[1]] += 1
            if not gitems and pno[1] == gexc:
                tp[pno[1]] += 1
            else:
                fp[pno[1]] += 1
        # txn correctness + error type
        if p is None:
            kind = "missing_line"
        elif gitems:
            if pno is not None:
                kind = "gold_items_pred_category"
            else:
                miss = [i for i in gitems if i not in pitems]
                extra = [i for i in pitems if i not in gitems]
                badamt = [i for i in gitems if i in pitems and abs(pitems[i][0] - gitems[i][0]) > tol(gitems[i][0])]
                badexc = [i for i in gitems if i in pitems and pitems[i][1] != gitems[i][1]]
                if not (miss or extra or badamt or badexc):
                    kind = None
                elif not pitems:
                    kind = "empty_pred"
                elif miss and extra and not set(gitems) & set(pitems):
                    kind = "wrong_items"
                elif miss or extra:
                    kind = ("missing_items" if miss else "") + ("+" if miss and extra else "") + ("extra_items" if extra else "")
                elif badamt:
                    kind = "amount" + ("+exception" if badexc else "")
                else:
                    kind = "exception"
        else:
            if pitems:
                kind = "gold_category_pred_items"
            elif pno is None:
                kind = "empty_pred"
            else:
                cat_conf[(gcat, pno[0])] += 1
                kind = None if (pno[0] == gcat and pno[1] == gexc) else ("category" if pno[0] != gcat else "exception")
        if kind:
            err_types[kind] += 1
            err_by_seg[seg][kind] += 1
            if dump_path:
                d, amt, cur, acct, txt = info.get(t, ("", "", "", "", ""))
                rows_out.append([t, seg, kind, d, amt, cur, acct,
                                 fmt_items(gitems) if gitems else f"CAT={gcat}{':' + gexc if gexc else ''}",
                                 (fmt_items(pitems) if pitems else (f"CAT={pno[0]}{':' + pno[1] if pno[1] else ''}" if pno else "<none>")),
                                 len(gitems), len(pitems), txt])
    per_exc = {}
    for c in EXC:
        if not gold[c] and not npred[c]:
            continue
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        R = tp[c] / gold[c] if gold[c] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per_exc[c] = {"support": gold[c], "pred": npred[c], "tp": tp[c], "precision": round(P, 4), "recall": round(R, 4),
                      "f1": round(F, 4), "counts_in_macro": bool(gold[c])}
    if dump_path:
        op = gzip.open if dump_path.endswith(".gz") else open
        with op(dump_path, "wt", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["bank_txn_id", "segment", "error", "booking_date", "bank_amount", "currency", "account_id",
                        "golden", "predicted", "n_gold_items", "n_pred_items", "bank_text"])
            w.writerows(sorted(rows_out, key=lambda r: (r[1], r[0])))
    return per_exc, err_types, err_by_seg, cat_conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--phase", default="train", choices=["sample", "train"])
    ap.add_argument("--from", dest="lo", default="", help="first booking date (inclusive), e.g. 2025-01")
    ap.add_argument("--to", dest="hi", default="", help="last booking date (inclusive), e.g. 2025-06")
    ap.add_argument("--dump", default="", help="error dump CSV(.gz) of wrong bank lines")
    ap.add_argument("--json", default="", help="write full result JSON here")
    ap.add_argument("--segments", type=int, default=40, help="segments to print (sorted by points lost)")
    a = ap.parse_args()
    S = import_scorer(a.data)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        g, info = build_golden(S, a.data, a.phase, a.lo, a.hi, tmp)
    t1 = time.time()
    raw = open(a.submission, "rb").read()
    data, dropped, delim = filter_submission(raw, g.txns)
    res = S.score_submission(g, data, "submission.csv")
    t2 = time.time()
    win = f"{a.lo or 'start'}..{a.hi or 'end'}"
    print(f"== {a.phase} window {win}: {len(g.txns)} bank lines, {g.n_pairs} golden pairs; "
          f"{dropped} submission rows outside window dropped (load {t1 - t0:.1f}s, score {t2 - t1:.1f}s)")
    if not res["ok"]:
        print("REJECTED by official scorer:")
        for e in res["errors"]:
            print(f"  row {e['row']}: {e['message']}")
        sys.exit(1)
    m = res["metrics"]
    print(f"SCORE {res['score']:.2f}   = 50*F1 {50 * m['allocation_f1']:.2f} + 30*acc {30 * m['txn_accuracy']:.2f}"
          f" + 20*excF1 {20 * m['exception_macro_f1']:.2f}")
    print("  " + "  ".join(f"{k}={v}" for k, v in m.items()))
    pred = parse_submission(data, delim)
    per_exc, err_types, err_by_seg, cat_conf = diagnose(S, g, pred, info, a.dump)
    k = sum(1 for v in per_exc.values() if v["counts_in_macro"])
    macro = sum(v["f1"] for v in per_exc.values() if v["counts_in_macro"]) / k if k else 1.0
    if abs(macro - m["exception_macro_f1"]) > 1e-3:
        print(f"  WARNING: diagnostic macro {macro:.4f} != official {m['exception_macro_f1']:.4f}")
    N = len(g.txns)
    print(f"\n-- exceptions (K={k} classes in golden; each class = {20 / max(k, 1):.2f} pts at F1=1)")
    print(f"  {'class':20s} {'support':>8s} {'pred':>8s} {'tp':>7s} {'P':>6s} {'R':>6s} {'F1':>6s} {'pts_left':>8s}")
    for c, v in sorted(per_exc.items(), key=lambda x: -x[1]["support"]):
        left = (1 - v["f1"]) * 20 / k if v["counts_in_macro"] and k else 0.0
        print(f"  {c:20s} {v['support']:8d} {v['pred']:8d} {v['tp']:7d} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {left:8.2f}")
    print("\n-- error types (wrong bank lines)")
    for kind, n in err_types.most_common():
        print(f"  {kind:28s} {n:8d}  ({100 * n / N:.2f}% of lines, {30 * n / N:.2f} acc-pts)")
    print(f"\n-- segments by accuracy points lost (30*n_wrong/N)")
    segs = res["segments"]
    rows = sorted(segs.items(), key=lambda x: -x[1]["n"] * (1 - x[1]["accuracy"]))
    for s, v in rows[:a.segments]:
        lost = 30 * v["n"] * (1 - v["accuracy"]) / N
        top = ", ".join(f"{kk}:{nn}" for kk, nn in err_by_seg[s].most_common(2))
        print(f"  {s:32s} n={v['n']:7d} acc={v['accuracy']:.3f} lost={lost:5.2f}  {top}")
    bad_cat = [(gc, pc, n) for (gc, pc), n in cat_conf.most_common() if gc != pc][:10]
    if bad_cat:
        print("\n-- top category confusions (gold -> pred)")
        for gc, pc, n in bad_cat:
            print(f"  {gc:20s} -> {pc:20s} {n}")
    if a.dump:
        print(f"\nerror dump -> {a.dump}")
    if a.json:
        out = dict(res, phase=a.phase, window=win, dropped_rows=dropped, per_exception=per_exc,
                   error_types=dict(err_types))
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
