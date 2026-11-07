#!/usr/bin/env python3
"""Optional LLM review agent for AMBIGUOUS cases only (never per transaction).

Reads the review queue written by main.py (`*_review_queue.csv`, confidence < 0.75), groups up to N cases per
request with their top candidate open items, and asks Claude for a structured verdict
(accept / reassign / unidentified + short explanation). Verdicts are written to `*_llm_verdicts.csv` for a human or
for `--apply` in a later version; the deterministic submission is never modified silently.

    python3 agents/llm_reviewer.py outputs/submission_live_review_queue.csv --dry-run     # default: no API call
    python3 agents/llm_reviewer.py outputs/submission_live_review_queue.csv --enable --max-cases 200

Safety: OFF by default (dry run prints the prompt size and estimated tokens). `--enable` requires the `anthropic`
package and credentials; it sends bank texts to the Anthropic API, so only use it with authorisation (synthetic data
here). Token budget: ~150 tokens per case, cases batched 25 per request.
"""
import argparse
import csv
import json
import os
import sys

MODEL = os.environ.get("BANKREC_LLM_MODEL", "claude-opus-5-5")
SYSTEM = (
    "You are a treasury cash-application reviewer. For each bank line decide, using ONLY the given text and "
    "candidate items, whether the proposed decision is right. Never force a match on amount alone: without a name, "
    "IBAN or document reference the receipt is UNIDENTIFIED. Answer with the JSON schema."
)
SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "bank_txn_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["accept", "unidentified", "needs_human"]},
            "explanation": {"type": "string"},
        },
        "required": ["bank_txn_id", "verdict", "explanation"],
        "additionalProperties": False}}},
    "required": ["verdicts"],
    "additionalProperties": False,
}


def build_prompts(rows, per_req=25):
    for i in range(0, len(rows), per_req):
        chunk = rows[i:i + per_req]
        lines = [f"{r['bank_txn_id']} | {r['booking_date']} | {r['amount']} {r['currency']} | rule={r['rule']} "
                 f"conf={r['confidence']} | cat={r['category']} exc={r['exceptions']} | text: {r['text'][:200]}"
                 for r in chunk]
        yield chunk, "Review these bank lines:\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--enable", action="store_true", help="actually call the API (costs tokens)")
    ap.add_argument("--max-cases", type=int, default=200)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.queue, encoding="utf-8")))
    rows.sort(key=lambda r: float(r["confidence"]))
    rows = rows[:a.max_cases]
    prompts = list(build_prompts(rows))
    est = sum(len(p) for _, p in prompts) // 4
    print(f"[llm] {len(rows)} cases in {len(prompts)} requests, ~{est:,} input tokens, model={MODEL}")
    if not a.enable:
        print("[llm] dry run (use --enable to call the API)")
        return
    try:
        import anthropic
    except ImportError:
        sys.exit("[llm] `anthropic` package not installed; run without --enable or install it")
    client = anthropic.Anthropic()
    out_path = a.queue.replace("_review_queue.csv", "_llm_verdicts.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["bank_txn_id", "verdict", "explanation"])
        for chunk, prompt in prompts:
            resp = client.beta.messages.create(
                model=MODEL,
                max_tokens=8000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "refusal":
                print("[llm] request declined; cases left for human review")
                continue
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            for v in json.loads(text).get("verdicts", []):
                w.writerow([v["bank_txn_id"], v["verdict"], v["explanation"]])
    print(f"[llm] verdicts -> {out_path}")


if __name__ == "__main__":
    main()
