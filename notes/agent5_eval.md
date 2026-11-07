# Agent 5 — QA & Evaluation (Phase A)

Tools: `bankrec/evaluation/evaluate.py` (scoring + diagnostics), `bankrec/evaluation/validate.py` (offline LIVE/FINAL format check).
Both import `data/starter/bankrec_score.py` unchanged (official server code).

## 1. Official scorer — exact behaviour (`bankrec_score.py`)

**Score = round(100·(0.5·allocF1 + 0.3·txnAcc + 0.2·excMacroF1), 2).** Amount-weighted accuracy and per-segment accuracy are display only.

Validation (ANY error ⇒ whole file rejected, `ok=False`, up to 50 errors listed):
- gzip detected by magic bytes; `.gz` name with non-gzip content rejected; else name must end `.csv`/`.txt`. ≤60 MB uncompressed, ≤3M rows, UTF-8, no NUL.
- Header: exactly the 5 columns, any order, case-insensitive, stripped; `;` delimiter if first line has ≥4 `;` and <4 `,` (then `,` decimals ok). Extra columns ⇒ reject.
- Every row must have exactly 5 fields; blank lines skipped. `bank_txn_id` must be in the phase. Exception/category upper-cased then checked against lists (empty allowed for both!).
- Item rows: `item_id` matches `^(ARI|ARC|API|APC)-\d{4}-\d{7}$` (case-sensitive); category must be empty; amount `^-?\d+(\.\d+)?$`, finite, 0 ≤ amt ≤ 1e12 (0 is allowed); no duplicate (txn,item).
- No-item rows: amount must be empty; ≤1 per txn; a txn cannot mix item rows and a no-item row (either order).
- Omitted bank lines are NOT an error — they just count as wrong (and lose their golden pairs as FN).

Metrics:
- **Allocation F1**: TP = submitted (txn,item) present in golden with |amt−g| ≤ max(0.01, 0.005·|g|)+1e-9. Precision = TP/all submitted pairs; recall = TP/golden pairs. Exception is irrelevant here.
- **Txn accuracy** (unweighted, over ALL phase lines): item lines correct iff no no-item row, #submitted items == #golden items and every golden item submitted with amount in tolerance AND identical exception (exact set equality). No-item lines correct iff exactly one no-item row with category == golden AND exception == golden (an empty category is accepted by the validator but is always wrong).
- **Exception macro-F1**: one "row key" per item row / no-item row. Gold count per class = item rows with that exception + no-item lines with that exception (only `UNIDENTIFIED` in practice). Submitted item row with exc e: TP iff (txn,item) in golden with exception e — **amount NOT checked**; else FP (wrong item or wrong class). Submitted no-item row with exc e: TP iff golden line has no items and its exception is e. FN = gold − TP. Average only over classes present in the golden (classes absent from golden: FPs are free; if none present macro=1). Empty exception is never a class.
- Golden facts (train 2021-07..2025-06, 1.49M lines, 5.26M pairs): no line mixes items and category; category UNIDENTIFIED ⇔ exception UNIDENTIFIED; no-item exceptions are only `""`/UNIDENTIFIED; UNIDENTIFIED never appears on item rows; ~3.5 pairs per line, 58% of lines are 1-item, 5% have ≥10.

## 2. Strategic implications (live ≈ 203k lines, ≈ 0.75M golden pairs, K≈10 exception classes)

- **Emit-a-pair threshold**: F1 = 2TP/(S+G). Adding a pair correct with prob p raises F1 iff **p > F1/2** (at F1=0.9 → p>0.45). Precision is not sacred: low-confidence pairs are still worth emitting above ~0.45 *for the F1 term*.
- **Marginal values (score points)**: 1 extra correct line ≈ 30/N ≈ **1.5e-4**; 1 extra TP pair ≈ 50·(2−F1)/(S+G) ≈ **3.7e-5** (F1≈0.9); 1 FP pair ≈ −50·F1/(S+G) ≈ **−3e-5**. ⇒ one correct line ≈ 4 pairs.
- **Item-line vs category row**: for a golden-item line, a category row scores 0 on everything; any guess has upside. For a *receipt* the choice is "match (p)" vs "UNIDENTIFIED (q)": UNIDENTIFIED also feeds a 2-point exception class, so wrong forced matches on unidentified receipts cost acc + FP pairs + UNIDENTIFIED recall. Policy explicitly warns about look-alike amounts (segment `unidentified_lookalike`).
- **Partial credit asymmetry**: a multi-item line with one wrong item loses the whole line for accuracy but keeps the other TPs for F1 → for big batches, emitting the best-effort set always beats a category row.
- **Exception macro-F1 is where rare classes are gold**: each class present = 20/K ≈ **2 pts** at F1=1. Marginal TP value ≈ (20/K)·(2−f)/(n_c+P_c). With 6-month class supports estimated from 2025H1 dev:

  | class | dev support (6m) | ≈ pts per extra TP | ≈ lines-equivalent |
  |---|---|---|---|
  | SHORT_BANK_CHARGES | 171 | 1e-2 | ~70 |
  | OVERPAYMENT | 400 | 5e-3 | ~30 |
  | DUPLICATE_PAYMENT | 707 | 3e-3 | ~20 |
  | SHORT_DISPUTE / RETURNED / SHORT_WITHHOLDING / UNIDENTIFIED | 3–4.6k | 5e-4 | ~3 |
  | PARTIAL / SHORT_DISCOUNT / FX_DIFFERENCE | 12.5–15.8k | 1.3e-4 | ~1 |

  Same threshold rule per class: label exception e if P(e) > F1_e/2. Never leaving a class at 0: getting each of the 10 classes from 0 → 0.5 F1 is +10 pts.
- Exception credit needs the right **item** only (amount ignored) → when the item is sure but the PARTIAL/SHORT_DISPUTE amount isn't, still emit the exception.
- **Don't spam UNIDENTIFIED**: starter puts it on every unmatched receipt → P=0.044, F1=0.085. A calibrated UNIDENTIFIED classifier is a direct ~1.8-pt lever.
- Tolerance is tight (0.5% / 1 cent): FX_DIFFERENCE/SHORT_* amounts must be the full open item amount in item currency, not the cash.
- FINAL is a single shot: always run `validate.py` and confirm 100% line coverage before upload.

## 3. Baseline (starter.py) — official numbers

| run | score | allocF1 | P | R | txnAcc | excMacroF1 | amtWAcc |
|---|---|---|---|---|---|---|---|
| sample (2025-06 subset, 8,499 lines) | **11.28** | 0.0735 | 0.999 | 0.038 | 0.248 | 0.0085 | 0.058 |
| train 2025-01..2025-06 window (199,987 lines) | **11.22** | 0.0828 | 0.992 | 0.043 | 0.230 | 0.0085 | 0.072 |
| train full (1,489,931 lines, `score_train.py`) | **11.68** | 0.0892 | 0.993 | 0.047 | 0.235 | 0.0086 | 0.080 |

Biggest losses (dev window, 30·n_wrong/N acc-pts): ar_ref_noise 6.4, ap_payment_multi 3.6, ap_payment_single 1.2, ar_short_discount 0.95, ar_same_amount_fifo 0.92, ar_1toN_text 0.92, ap_paid_by_holding 0.69. 74.5% of lines = "golden has items, we emitted a category". Category confusions led by CASH_POOL, PAYROLL, LEASING, CUSTOMER_ADVANCE, INTERCOMPANY → BANK_FEES/UNIDENTIFIED.

## 4. Harness CLI

```
cd "/Users/pedroantonperez/Desktop/Claude Hackathon/bankrec/evaluation"
python3 evaluate.py SUB.csv.gz --phase sample [--dump /tmp/err.csv]
python3 evaluate.py SUB.csv.gz --phase train --from 2025-01 --to 2025-06 --dump /tmp/err.csv.gz --json /tmp/res.json
python3 validate.py SUB_live.csv.gz --phase live [--full-items]
```
- `--from/--to`: inclusive booking_date prefixes (YYYY, YYYY-MM, YYYY-MM-DD); golden + line universe restricted to the window, submission rows outside the window dropped (count printed). So one full-train submission can be scored on any slice.
- Prints official score/metrics + split into 50·F1 / 30·acc / 20·macro; per-exception support/pred/TP/P/R/F1/points-left (diagnostic macro is cross-checked against the official value); error-type histogram; segments sorted by points lost with dominant error types; top category confusions.
- Error dump columns: `bank_txn_id, segment, error, booking_date, bank_amount, currency, account_id, golden, predicted, n_gold_items, n_pred_items, bank_text`. `golden/predicted` = `item:amount[:EXC];…` or `CAT=X[:EXC]`. Error kinds: missing_line, gold_items_pred_category, gold_category_pred_items, empty_pred, wrong_items, missing_items, extra_items, missing_items+extra_items, amount, amount+exception, exception, category.
- Cost: dev window ≈ 16 s, ~1 GB RAM; sample < 1 s. Full train window works but needs several GB.
- `validate.py`: runs the official `score_submission` against a dummy golden built from the phase's bank lines → hard errors identical to the server's (exit 1). Then warnings: missing lines, unknown item ids (default: phase items + `open_items_at_start`; `--full-items` adds train items for DUPLICATE/RETURNED on closed items), item issued after bank date, amount > item total (+tolerance, except DUPLICATE), noitem exception ≠ UNIDENTIFIED, UNIDENTIFIED category/exception mismatch or on a debit, UNIDENTIFIED/FX_DIFFERENCE on wrong row types, >2 decimals, zero amounts, item reused on >3 lines, never-predicted exception classes, size near limit. Starter live file: passes, 112 unknown-item warnings (train items not open at start).

## 5. Validation protocol

- **Dev A (primary)**: train **2025-01..2025-06** (200k lines ≈ size of LIVE's 203k). History = golden with booking_date ≤ 2024-12-31.
- **Dev B (seasonal check)**: train **2024-07..2024-12** (LIVE is Jul–Dec 2025: same months, year-end effects, rappel credit notes, retention releases). History ≤ 2024-06-30.
- **Fast loop**: `--phase sample` (8.5k lines) or train 2025-06. NB: **sample ⊂ train 2025-06** (all 8,499 ids are in train) — any lookup fitted on all of train leaks into sample; treat sample as a smoke test only.
- Report Dev A and Dev B; ship only changes that help both. Final live/FINAL fit uses all train golden.

Leakage rules (the matcher must run on dev exactly as it will on live):
1. Every golden-derived artefact (payer IBAN→customer, name aliases, reference/PO maps, category text rules, segment priors, customer pay-delay or FIFO habits, thresholds) is fitted on golden with booking_date **< window start**. Pass a `--cutoff` to the pipeline; assert `max(golden_date_used) < from`.
2. Open balances at window start: derive from items + golden **before** the window (mirrors `live/open_items_at_start.csv.gz`); never consume window golden to know an item is closed. Within the window, balances evolve only from our own predictions.
3. Inputs dated in the window (invoices, remittances, batches, PDFs) are fine — LIVE has them too — but item files must not be filtered by golden (e.g. "items that get paid").
4. Don't tune hand rules by reading dev-window dumps and then report Dev A as unbiased: rotate — diagnose on Dev B dumps, confirm on Dev A (or vice versa), keep a last untouched month (2025-05) for a final sanity check.
5. Unmeasurable drift (Moncayo camt.053 from 2025-10, remittance layout change 2025-09, ZA01 in FINAL): add synthetic robustness checks (e.g. strip/format-shift references on a dev slice, hold out one bank account's history) rather than trusting dev scores.

## 6. Test ideas (pytest)

- `test_scorer_parity`: evaluate.py score on sample == `score_train.py --sample` score (11.28 for starter); diagnostic macro == official macro.
- `test_window_isolation`: window score with a submission containing only window rows == same with full-train rows (dropping works).
- `test_golden_as_submission_scores_100`: golden of a window rewritten to the 5 columns → score 100.00 (sanity of loader/filters).
- `test_validator_rejects`: crafted bad files → mixed item+category line, dup pair, negative amount, unknown category, lowercase item prefix, extra column, `.gz` name with plain content → each rejected with expected message; `;`+comma-decimal file accepted.
- `test_coverage`: pipeline output for live covers 100% of `live/bank_transactions` ids, ≤1 no-item row per line.
- `test_no_leakage`: run pipeline with `--cutoff 2025-01-01`, monkeypatch golden reader to raise on any row with booking_date ≥ cutoff.
- `test_amount_rules`: FX_DIFFERENCE/SHORT_DISCOUNT/OVERPAYMENT rows use full open amount in item currency; PARTIAL ≤ open amount; credit-note rows positive with no exception (except RETURNED).
- `test_determinism`: two runs → identical bytes (sorted output).
- `test_size_limits`: live/final output < 60 MB uncompressed, < 3M rows.
