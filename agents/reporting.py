"""Audit trail, review queue and markdown audit report."""
import csv
import gzip
import os
from collections import Counter

REVIEW_THRESHOLD = 0.75


def write_audit(S, bank, sub_path):
    base = sub_path.replace(".csv.gz", "").replace(".csv", "")
    trail = base + "_audit_trail.csv.gz"
    queue = base + "_review_queue.csv"
    report = base + "_audit_report.md"
    rules, excs, cats = Counter(), Counter(), Counter()
    review = []
    with gzip.open(trail, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["bank_txn_id", "booking_date", "account_id", "amount", "currency", "rule", "confidence",
                    "n_items", "category", "exceptions", "note", "text"])
        for t in bank.itertuples(index=False):
            rule, conf, note = S.why.get(t.bank_txn_id, ("none", 0.0, ""))
            rows = S.out.get(t.bank_txn_id, [])
            items = [r for r in rows if r[0]]
            cat = "" if items else (rows[0][2] if rows else "")
            ex = sorted({r[3] for r in rows if r[3]})
            rules[rule.split(":")[0] + (":" + rule.split(":")[1] if ":" in rule else "")] += 1
            for r in rows:
                if r[3]:
                    excs[r[3]] += 1
            if cat:
                cats[cat] += 1
            w.writerow([t.bank_txn_id, t.booking_date, t.account_id, t.amount, t.currency, rule, f"{conf:.2f}",
                        len(items), cat, "|".join(ex), note, t.text[:160]])
            if conf < REVIEW_THRESHOLD:
                review.append([t.bank_txn_id, t.booking_date, t.entity, t.amount, t.currency, rule, f"{conf:.2f}",
                               cat, "|".join(ex), t.T[:240]])
    with open(queue, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["bank_txn_id", "booking_date", "entity", "amount", "currency", "rule", "confidence",
                    "category", "exceptions", "text"])
        w.writerows(review)
    n = len(bank)
    lines = [f"# Audit report — {S.phase}", "",
             f"- Bank lines: **{n:,}**",
             f"- Lines allocated to items: **{sum(1 for v in S.out.values() if any(r[0] for r in v)):,}**",
             f"- Lines sent to review (confidence < {REVIEW_THRESHOLD}): **{len(review):,}** → `{os.path.basename(queue)}`",
             f"- Full trail: `{os.path.basename(trail)}`", "",
             "## Decisions by rule", "", "| rule | lines | share |", "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {v / n:.1%} |" for k, v in rules.most_common()]
    lines += ["", "## Exceptions (rows)", "", "| exception | rows |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in excs.most_common()]
    lines += ["", "## Categories (non-item lines)", "", "| category | lines |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in cats.most_common()]
    lines += ["", "## Anomaly flags", "",
              f"- DUPLICATE_PAYMENT rows: {excs.get('DUPLICATE_PAYMENT', 0):,} (refunds expected later as CUSTOMER/SUPPLIER_REFUND)",
              f"- RETURNED rows: {excs.get('RETURNED', 0):,}",
              f"- UNIDENTIFIED receipts: {cats.get('UNIDENTIFIED', 0):,} (no open item with name/IBAN/reference evidence — never forced)",
              f"- BANK_REVERSAL lines: {cats.get('BANK_REVERSAL', 0):,}"]
    try:
        from anomaly_detection.flags import audit_flags
        fl = audit_flags(S, bank, getattr(S, "DUP", None))
        fl.to_csv(base + "_audit_flags.csv", index=False)
        lines += ["", "## Audit flags (not scored)", "", "| flag | severity | n |", "|---|---|---|"]
        for (f, sev), n_ in fl.groupby(["flag", "severity"]).size().sort_values(ascending=False).items():
            lines.append(f"| {f} | {sev} | {n_:,} |")
        lines.append(f"\nDetail: `{os.path.basename(base)}_audit_flags.csv`")
    except Exception as e:  # flags are informative only; never block the submission
        lines += ["", f"_audit flags unavailable: {e}_"]
    open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"[audit] {trail}\n[audit] {queue} ({len(review):,} lines)\n[audit] {report}")
