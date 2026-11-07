# Agent 5 — Independent review of the main tree (Phase C)

Evidence: `outputs/base_2025-05-01_2025-06-30.csv.gz` (dev 72.08) scored with `evaluation/evaluate.py --from 2025-05 --to 2025-06`,
joined with its audit trail (rule → line accuracy) and with golden (predicted vs golden exception per pair).
Points are estimates for that window (71k lines). Read-only review: no files outside tests/ and evaluation/ were edited.

## A. Score bugs, ranked by estimated impact

1. **AP payment runs never emit PARTIAL** (`matching/files.py::match_payment_run`). Segment `ap_partial`: 907 lines, acc 0.000,
   848 of them have the right items and amounts but no exception. When `document_amount < item open` the row must be `PARTIAL`
   (FX keeps PARTIAL). `S.take` then also sets the item's open balance to 0 instead of leaving the rest open, so the later
   remainder payment is lost too. Fix: `exc = "PARTIAL" if da < I[it]["open"] - 1 else ""`. ≈ +0.35 acc pts, and PARTIAL recall
   0.32 → ~0.5 (≈ +0.3 macro pts).
2. **SHORT_BANK_CHARGES fires on any small shortfall** (`exceptions/short_payments.py`). 440 predicted, 66 TP (P = 0.15);
   374 lines are wrong (`ar_ref_noise`, instalments, n_to_m, partials, discounts…). POLICIES §3: only cross-border transfers from
   **outside SEPA** (foreign orig currency, non-SEPA payer country/BIC, SWIFT-type bank code). Gate it on that and only accept it
   when no subset of open items explains the cash. SBC is a ~2-pt class with ~70 support: going from F1 0.26 to ~0.7 is
   ≈ +0.9 pts, plus ≈ +1.2 acc pts from fixing the 374 lines. Covered by the xfail test `test_no_bank_charges_for_domestic_sepa_eur`.
3. **Remittance SHORT_DISPUTE is a catch-all** (`matching/remittance.py::rows_from_advice`). 2,165 rows predicted: 448 SHORT_DISPUTE,
   **619 actually SHORT_DISCOUNT**, 131 PARTIAL, 27 FX, and 915 pairs that are not in golden at all (parsing picks the wrong documents).
   Before choosing SHORT_DISPUTE: (a) check the discount terms/date (re-use `short_payments._disc`), (b) require a deduction reason
   in the advice row, (c) check that the advice rows sum to the bank amount and reject the advice otherwise (that removes most of the
   915 bad pairs). SHORT_DISPUTE F1 0.27 → ~0.6 ≈ +0.7 pts; SHORT_DISCOUNT recall also goes up.
4. **FX_DIFFERENCE on payment runs keyed on transfer currency, not item currency** (`match_payment_run`). Of 1,154 FX rows,
   only 464 are TP. 337 are correct pairs whose golden exception is empty (`payment_run:e2e` on `ap_payment_multi/single/partial`),
   and 342 are pairs not in golden. Policy: FX applies when the **bank line currency ≠ invoice currency**. Use
   `I[it]["cur"] != t.currency` per row. FX F1 0.125 → ~0.3+ ≈ +0.4 pts. FX recall (0.074) is the bigger lever: `match_ap` has
   no FX path at all (`ap_fx` 418 lines, acc 0.39) and `fx_rate_single` is 21 % accurate (it picks the closest of all foreign items).
5. **OVERPAYMENT from `ref_partial` when one reference is found but the line pays several invoices** (`matching/party.py`).
   394 predicted: 58 TP, 303 are pairs whose golden exception is empty (`missing_items` 326 lines). Only call OVERPAYMENT if the
   excess is smaller than every other open item of the party or no subset/FIFO explains it; otherwise try subset-sum with the
   ref forced in. OVERPAYMENT is a high-value rare class (≈ 1.6 pts left).
6. **UNIDENTIFIED precision 0.27**: 6,015 lines predicted, 4,047 have golden items (`unidentified` rule acc 0.27). This is mostly
   matcher recall. Cheap partial fix: keep the UNIDENTIFIED exception only when no party is identified (name/IBAN), otherwise use a
   category without the exception, or better, a low-confidence match (the threshold for emitting a pair is p > F1/2 ≈ 0.37).
7. **`ar:subset2` (ambiguous subset sums) accuracy 0.45, 1,770 lines; `ar:subset1` 0.62.** Prefer FIFO/oldest-first among
   solutions (POLICIES §5.6) and refs. Possible gain ≈ 0.5 acc pts.
8. **Category rules match party names** (`exceptions/categories.py`; the `search` text includes `counterparty_name` and
   `final=True` skips matching). 129 lines in the window have golden items but were forced to a category, almost all `TAX_PAYMENT`
   (`IVA ` matches "…OLIVA ", plus `IRS `/`INPS`/`AEAT` substrings). Same risk for `FEE` (COFFEE, FEEDBACK), `SEGURO`, `LEASING`,
   `PRESTAMO` and `INTEREST` in customer names. Use `\b` word boundaries and exclude counterparty_name from final rules.
   Separately, CASH_POOL (372 wrong) and INTERCOMPANY (141) go to UNIDENTIFIED/BANK_FEES. **Verified: all 513/513 of these lines
   have `counterparty_account` = an IBAN in `bank_accounts.csv`** (texts like "TRASPASO PROPIO", "FINANCIACION IC …").
   Rule: if the counterparty IBAN is a group account, then same entity → INTERNAL_TRANSFER/CASH_POOL (header ES00 involved → CASH_POOL),
   different entity → CASH_POOL or INTERCOMPANY (from text/history). ≈ +0.2 acc pts, and cuts 248 false UNIDENTIFIED.

## B. FINAL / LIVE file-rejection and crash risks

1. **FINAL crash: `S.entities.at[t.entity, "country"]` raises KeyError for ZA01** (`short_payments.py` L91). `masters/entities.csv`
   has no ZA01, and the code always reads `data/masters`, never `final/masters_final/`. The first ZA01 receipt that reaches
   `explain_short` aborts the run. Load `masters_final` when present (customers, suppliers, accounts, entities, fx ZAR) and use
   `.get` lookups. Add a FINAL dry-run on live data with ZA01 masters injected.
2. **Non-determinism.** `find_refs` iterates `set(re.findall)`, `parties` is a set, and `pool` is built from set iteration, so the
   choices `refs[0]`, `c2[0]`, the `order` in `explain_short` and `cands[0]` depend on `PYTHONHASHSEED`. On sample only the row order
   differed (content identical, test `test_bytes_deterministic…` xfail). On live, the content can differ between runs. Fix: sort
   those collections and sort rows inside each line in `write_submission`, and run FINAL with `PYTHONHASHSEED=0`.
3. **Stale context cache.** `build_context` pickles `outputs/cache/ctx_{phase}_{start}_{end}.pkl` with no code or data version in
   the key. After any data_loader change, dev runs silently reuse the old context, so scores are misleading. Add a hash of
   `data_loader/*.py` + data mtimes to the key, or delete the cache on merge. Also make sure the live/final run is not built
   from a stale pickle.
4. **Validation is safe today.** `write_submission` never mixes item and category rows, at most one no-item row per line, `take()`
   de-duplicates items, amounts go through `fmt` (2 decimals) of non-negative cents (no negative batch amounts in train/live files),
   categories always come from the list, and every bank line is written. `validate.py` passes on the live starter file; still run it
   on the actual LIVE/FINAL output. Residual risk: `cents(nan)` raises if an item/batch amount is ever empty (crash, not rejection).

## C. Leakage

- Dev context is clean. `golden_hist` is filtered to `booking_date < start`. Items are restricted to `issue_date ≤ end` with open
  balances from history only. `learn()` is empty. No other module reads `golden_allocations` (static test). The heavy test
  `test_dev_context_has_no_window_golden` checks this on the cached 2025-05..06 context.
- `sample` uses no history (correct: sample ⊂ train 2025-06).
- Dev vs live mismatch to check: live items = `open_items_at_start` (given) while dev recomputes open balances. Invoices cancelled by
  a credit note ("anula y sustituye", never settled by bank) stay "open" in dev but may be absent from live `open_items_at_start`.
  Compare dev-reconstructed open items at 2025-07-01 with `live/open_items_at_start.csv.gz` once (item ids + amounts).
- Closed items are dropped (`open_amount > 0.005`) in both dev and live, so DUPLICATE_PAYMENT / RETURNED / CUSTOMER_REFUND of items
  settled **before** the window start can never be found. This is consistent between dev and live (no leakage), but it caps recall.
  Loading closed train items as a lookup-only pool would fix it.

## D. Smaller items

- `is_return` true but `match_return` fails → the line falls through to `match_payment_run`/`match_ap` as if it were a normal payment.
- `match_dd` / `match_cheque` take the first batch token found; `resolve_ar` suffix fallback returns `c2[-1]` (newest) and ignores
  the batch amount when no item matches it. `dd_batch` accuracy is 0.82.
- `explain_short` SHORT_WITHHOLDING uses `tot` without checking `open == tot` (after a partial payment it can misfire).
- `payment_run:e2e/e2e16`: 244 + 114 `extra_items` lines. The run lookup returns every document of a multi-currency or re-used e2e.
  Filter rows by debtor account/entity and by `sum == transfer_amount`.
