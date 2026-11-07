# Agent 5 — Final QA (Phase E) on the merged tree

## 1. Test suite: `python3 tests/run_tests.py` (pytest-compatible)

**47 PASS, 0 FAIL, 1 SKIP.** Tests I updated for the merged code:
- `test_short_payments.py`
  - The fake State now has `ctx`, `phase` and `start`, which the new withholding lookup needs.
  - Rule-name checks are loosened to prefix checks: `fx_orig*`, `fx_rate*`, `a4_*`.
  - Withholding follows the new per-country table (MX 1.25 %, BR/MA 1.5 %). New test: 10 % is refused for customers with no withholding history.
  - The bank-charges cap is now 3.3 %, so the rejection test uses a 4 % shortfall.
  - The two Phase C known-bug markers now pass, so I removed them: SHORT_BANK_CHARGES is no longer applied to domestic SEPA EUR payments, and the output is byte-identical between runs.
- `test_writer.py`: byte-identical output across hash seeds now passes, because `main.py` re-executes itself with `PYTHONHASHSEED=0`. That makes the test a reproducibility check, not a hash-independence check.
- `test_leakage.py`: the cache file name now includes `CTX_VERSION` (`ctx_v3_train_…`).
- Skipped: `test_dev_context_has_no_window_golden`, because there is no `ctx_v3_train_2025-05-01_2025-06-30.pkl` yet. Run it with `BANKREC_HEAVY=1` under the lock. The v1 context passed this test in Phase C.

## 2. LIVE submission: `python3 evaluation/validate.py outputs/submission_live.csv.gz --phase live --full-items`

**OK.** The official validation passes:
- 701,361 rows, 203,083/203,083 bank lines covered, 7.4 MB.
- All 10 exception classes are predicted.

Warnings, checked against train golden for 2025 (545k pairs):

| Warning | Count in LIVE | Times it happens in golden |
|---|---|---|
| Item issued after the bank line | 133 pairs | 0 |
| Amount above item total (not DUPLICATE) | 73 pairs | 1 |
| Item used on more than 3 lines | 233 | not checked; can be legitimate (instalments) |

So the first two groups are almost certainly wrong pairs:
- "Issued after" comes from `dd_batch` 18, `remittance_advice` 16, `ar:ref_partial` 9, `payment_run:e2e16` 5 and `cheque_deposit` 4.
- "Above total" comes from `payment_run:e2e` 33, `run_count` 14, `e2e16` 12 and `dd_batch` 7. These are doc numbers resolved to the wrong item: same number, different supplier or year.

## 3. Review findings, ranked by impact

1. **UNIDENTIFIED precision 0.37: about +0.4–0.5 points available.**
   - Dev 2025-05..06: 4,595 predicted, 1,683 TP.
   - All 1,683 true UNIDENTIFIED lines have an **empty `counterparty_account`**.
   - 259 false positives carry a customer-master IBAN. The other 2,653 are company names the matcher could not allocate (`ar_ref_noise` 836, instalments 310, `1toN_credit_note` 275, `n_to_m` 228…).
   - Patch in `matching/engine.py`, credit fallback:
     ```python
     if t.cat == "UNIDENTIFIED":
         known = bool(t.counterparty_account) or bool(ar_parties(t, T, S))   # from .party import ar_parties
         if known:   # payer identified -> an open item exists, policy's 'no open item' does not apply
             S.category(t, "CUSTOMER_ADVANCE", "", "unidentified_known_payer", 0.2)
         else:
             S.category(t, "UNIDENTIFIED", "UNIDENTIFIED", "unidentified", 0.5)
     ```
   - The line stays wrong either way, because golden has items, but this removes the false positives from the UNIDENTIFIED class.
   - The IBAN half alone has zero true-positive loss on dev. Check the `ar_parties` half on dev before shipping, because look-alike payers can carry names.
2. **FINAL masters gaps.**
   - `data_loader/loader.py::_party_names()` reads `BASE_MASTERS`, so the FINAL `masters_final` is ignored when naming `open_items_at_start`. That affects ZA01 customers and the 2026-04-01 rename. Patch: `os.path.join(MASTERS, ...)`. `MASTERS` is re-pointed by `set_phase_masters`.
   - `masters_dir` puts ZA01 into entities and accounts with country "ZA" and currency from the bank lines. Then `nonsepa` is true and `S.entities.at` is safe (now guarded with `in S.entities.index`). The Phase C crash is fixed.
   - ZAR is in `fx_rates.csv` (1,246 rows).
   - "KWI…" invoice numbers go through the generic `nz` and full-number indexes. The 6-digit suffix index only helps if there are 5 or more digits. Nothing in the code is specific to KWI, so recall for ZA01 is unknown.
   - Recommendation: before FINAL, run `python3 tools/locked.py -- python3 main.py --phase final` as a dry run as soon as `final/` lands, then validate it.
3. **Wrong-pair guards (about 200 lines in LIVE, small but free).** Drop item rows whose `iss > t.booking_date` (never happens in golden). For payment runs, use `I[i]["tot"] == abs(da)` as a hard filter instead of a preference. Patch in `State.take`, with the rows-to-clean loop:
   ```python
   if it and it in self.I and self.I[it]["iss"] > t.booking_date: continue
   ```
   If nothing is left after that, fall back to `self.category(t, "UNIDENTIFIED" if t.amount_f > 0 else "CUSTOMER_REFUND", ...)`.
4. **Cache keys need a manual version bump (stale-risk, not leakage).**
   - `ctx_{CTX_VERSION}_...` only changes when someone bumps `CTX_VERSION`.
   - `outputs/cache/masters_<phase>/` is rebuilt every call (fine).
   - Patch: add `hashlib.md5(open(__file__.replace('history', 'loader'),'rb').read()).hexdigest()[:6]` and the `history.py` hash to the key.
   - Before FINAL, delete `outputs/cache/ctx_*_final_*` if `loader.py` changed after it was built.
5. **OVERPAYMENT F1 0.53 and SHORT_BANK_CHARGES F1 0.46 (about 1 point each left).**
   - Both are recall-limited (R 0.42 / 0.33).
   - SBC: `explain_short` only accepts a bank-charges explanation for a referenced invoice or a single candidate in a pool of 3 or fewer. Allowing a single candidate in a larger pool when the payer is non-SEPA (USD/GBP/CNY…) should lift recall. Measure precision on dev before shipping.
   - OVERPAYMENT: `a4_overpayment` requires `len(rsame)==1` and an excess of 20 % or less. `ref_overpayment` requires 15 % or less. These overlap; unify them at 20 % and also allow a name-only unique invoice with excess of 5 % or less.

## 4. Leakage audit: clean

- `learned.py` uses `ctx['golden_hist']` only (cut at window start). Bank texts are those of the history txns; the item→party map comes from item files, which are inputs.
- `short_payments._wht_customers` caches in `outputs/a4_wht.pkl` under the key `f"{phase}_{start}_{len(ids)}"`. The history depends only on `start`, so the key is correct; a pickle from another window is rejected. It is a single-entry file, so concurrent windows just recompute.
- `returns.HistTables` and `duplicates._seed` use golden history before `lo` = the window's first booking date, plus item and batch files (inputs).
- The static test (no module other than `history.py` opens `golden_allocations`) passes.
- Live/final use all of train golden as history (correct; there is no live golden).

## 5. Policy spot-checks

- No amount-only forced matches. `match_ar`/`match_ap` require a party (name/IBAN/learned IBAN) or a reference. Unreferenced on-account PARTIAL is disabled.
- `UNIDENTIFIED` category always carries the `UNIDENTIFIED` exception (postprocess). FX rows exclude credit notes.
- Determinism is fixed by the seed re-exec in `main.py`. Run FINAL through `main.py`, not by importing `engine.run` directly.
