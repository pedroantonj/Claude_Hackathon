# Agent 2 — Matching analysis (Phase A)

Data: `sample/` (2025-06, 3 entities ES01/FR01/DE01, 8,499 lines) and TRAIN 2025-01..06 (199,987 lines, 735k item pairs,
16 entities — same size as a LIVE half-year). Scripts: `bankrec/notes/agent2_*.py` (prototype: `agent2_proto.py`).

## 0. Headline results
| Submission | sample (official scorer) | TRAIN 2025-05, realistic open pool (16 ent.) |
|---|---|---|
| starter baseline | 11.28 | – |
| `agent2_proto.py` (3.5 s sample / 25 s month) | **69.21** (F1 .799, P .993, R .669, txn acc .817, exc-F1 .237) | **64.42** (F1 .758, P .960, txn .767, exc-F1 .175) |

**Warning: `sample/` leaks.** Its item files hold only the items settled in June plus the ones issued in June, so there are almost
no decoy items and every AR matcher looks better on it than it is. For realistic evaluation use `agent2_mkphase.py <YYYY-MM>`. It
builds `/tmp/a2data/sample` from one TRAIN month. The pool is every item open at month start (settle date taken from golden)
plus the month's new items, including about 350k AR items that are never settled. That is about 660k items, close to LIVE's ~1M.

## 1. Where the points are (TRAIN 2025H1; pts = max score contribution of segment, alloc 50 + txn 30)
| segment | lines | item rows | rows/line | pts | key | notes |
|---|---|---|---|---|---|---|
| ar_ref_noise | 42,462 | 92,722 | 2.2 | **12.7** | party name + amount (subset) | ref in text only 19 % |
| ap_payment_multi | 24,058 | 85,430 | 3.6 | **9.4** | payment_runs e2e | 97 % exact |
| ar_short_dispute | 1,067 | 104,691 | **98** | **7.3** | remittance advice | 7.1 pts are alloc-F1 |
| ar_dd_batch | 1,530 | 94,025 | 61 | **6.6** | batch_id → DD file | 96.5 % exact |
| ar_1to1 | 19,143 | 19,143 | 1 | 4.2 | docno in text 98.6 % | |
| ar_1toN_remittance | 1,206 | 54,817 | 45 | 3.9 | remittance advice | |
| ap_payment_single | 17,368 | 17,368 | 1 | 3.8 | payment_runs e2e | |
| ar_1toN_text | 6,124 | 36,220 | 5.9 | 3.4 | prefix+suffix list in text (76 %) | |
| ap_fx | 1,217 | 45,117 | 37 | 3.3 | payment_runs + orig_amount | |
| ar_1toN_credit_note | 3,184 | 32,775 | 10 | 2.7 | subset incl. ARC (negative) | |
| ar_card_settlement | 1,252 | 21,557 | 17 | 1.7 | merchant id + net | |
| ap_paid_by_holding | 4,798 | 12,497 | 2.6 | 1.6 | payment_runs (ES00 pays ES01/ES02/PT01) | |
| ar_short_discount / same_amount_fifo / ap_credit_note_netted / ap_partial / ap_direct_debit | 6.3k/6.2k/2.6k/2.5k/5.4k | | | 1.4/1.3/1.3/1.2/1.2 | | |
| nonitem_* + unidentified* (all category lines) | ~29k | – | 1 | ~4.4 | categories (other agent) | |

- About 14 % of lines have no item (category). The other 86 % have items, and they carry the 50-pt allocation-F1. Half of the item rows sit in
  4 segments: ar_short_dispute, ar_dd_batch, ar_ref_noise and ap_payment_multi. **Large many-item lines (RA, DD, card,
  aggregated runs) are worth more allocation-F1 than all the 1:1 lines.**
- Exception macro-F1 is 20 pts spread over 10 classes, **2 pts per class however rare**. Row counts in 2025H1: FX 15.8k, SHORT_DISCOUNT 15.6k,
  PARTIAL 12.5k, UNIDENTIFIED 4.6k, SHORT_WITHHOLDING 4.3k, RETURNED 4.1k, SHORT_DISPUTE 3.0k, DUPLICATE 707, OVERPAYMENT 400,
  SHORT_BANK_CHARGES 171. The prototype only emits SHORT_DISCOUNT, PARTIAL, OVERPAYMENT, SHORT_DISPUTE and UNIDENTIFIED, which is why its
  exception-F1 is 0.17–0.24.

## 2. Amount relationships (sample + TRAIN agree)
- **Cash = exact net sum of the golden rows** (invoices + and credit notes −) on 100 % of lines in almost every segment. Exceptions:
  short_discount, overpayment, short_withholding, short_bank_charges and fx (all by design); card (net, not gross); and ar_short_dispute,
  where cash = Σ paid amounts. So a subset-sum on exact cents is the main engine.
- Row amount = full open amount (`total_amount`) in 98–100 % of rows. The exceptions are PARTIAL, instalment, retention, dispute
  and advance rows, where amount = cash.
- Date lags. AR booking−due: p10 0, median **+9 d**, p90 +48, p99 +165, max >1000 (old items do get paid). AP booking−due:
  median +2 d, p1 −6, p99 +63. Issue→booking for AR: median 48 d, p99 218 d. Do **not** filter candidates by date beyond
  `issue_date ≤ booking_date`. Use the date only to rank.
- SHORT_DISCOUNT: 99.1 % satisfy booking ≤ issue + discount_days + 3, and booking−due is negative (median −24 d). This rule is clean.
- DD batch: booking−due is 0..3 d. Same-amount FIFO by **issue_date**: 96 % correct when the party is known.

## 3. Deterministic keys and hit rates (TRAIN 2025H1)
**Invoice-number reference** (pair level; full = whole normalised docno in text, suffix = text digit token ≥5 that is a suffix of
the docno digits, with or without leading zeros):
| seg | full | suffix | PO | any |
|---|---|---|---|---|
| ar_1to1 | .986 | .986 | 0 | .986 |
| ar_1toN_text | .383 | **.757** | 0 | .759 |
| ap_payment_multi / single | .746 / .870 | .49 / .58 | 0 | .75 / .87 |
| ar_instalment / partial | .70 / .49 | .72 / .56 | | .75 / .58 |
| ar_ref_noise | .023 | .153 | .042 | **.195** |
| ar_1toN_credit_note | .108 | .297 | .038 | .325 |
| ar_short_dispute, dd_batch, 1toN_remittance, card, cheque, confirming, same_amount_* | 0 | ~0 | 0 | 0 |
All items: 27 % of pairs have any reference in the bank text.

Noise seen (real examples): `RE252008072` for RE250208072 (transposition); `RE205106342` for RE250106342; `RE 108031` (prefix and
year dropped); `FV25-0106436 0107285 0107357 0 | 101471` (list of 7-digit tails split across N43 38-char fields);
`FA250212924 208171` (first in full, then 6-digit tails); `RECIBO 0106536` in DD files (no prefix); ref2 truncated at 16 chars
(`FRAS FV25-030812`). Normalisation: uppercase, keep [A-Z0-9], and index each item by (entity, full docno) and by
(entity, last 6 digits). **rapidfuzz** helps here: Damerau/Levenshtein ≤1 between a text token and the docnos of the
*candidate party only* catches the transpositions (not yet implemented).

**Party identification** (AR lines, excluding batches; 40k-line sample of 2025H1):
- 42 % of customers share their **exact name** with another customer of the same entity (up to 21 copies, e.g. 3× "Olivia Long"
  in US01). A name alone therefore cannot pick one customer. It yields a candidate set, and the amount decides.
- IDF token matcher: top-1 44 %, top-3 61 %. Not usable, because generic words like POOL/SPA/ACH/CREDIT drown the signal.
- **Truncation-aware prefix matcher** (`PrefixIndex`: normalised text from every word start, longest common prefix by bisect
  against sorted normalised names from master name + trade_name + every `customer_name` seen on invoices): **recall 98.7 %**
  (camt 98.6 / mt940 99.0 / n43 98.5), candidate set median 1, p90 6, 0.03 ms per line. The misses are third-party payers
  (parent: expand with `parent_customer_id`; confirming "ORD. <10-char name>": needs a lower min-prefix after `ORD.`) and renamed
  customers (e.g. "Water Care Mar" → Aquatics Taylor Inc.). For renames, learn an alias table from TRAIN golden: payer
  string/IBAN → party.
- Master IBAN: the `counterparty_account` is present on only about 11 % of AR lines (camt), but when present it gives 99 % correct party.
- For AP debits the bank text has the beneficiary name. Matching supplier names by prefix plus exact amount gets ap_direct_debit to 92–100 %.

**Given the true party: amount logic** (30k AR lines, realistic open pool; open items per party median 7, p90 54, p99 1,017):
unique single exact amount 14.9k lines → 99.9 % correct; several same-amount items → oldest by issue date 87 %; Σ all open = amount
91 %; FIFO prefix by due date 75 %; subset-sum (meet-in-middle, n ≤ 26) unique 97 %, ambiguous 82 %; no exact solution 4.2k
lines (discount, partial, withholding, FX); too many items for subset 5.7k lines (key accounts, which need RA or refs). Result per segment:
ar_ref_noise 83 %, ar_1to1 99.6 %, same_amount_fifo 96 %, same_amount_month 77 % (fix: parse "MANT. MAYO 25"/"SERVICE JANUARY 2025"
→ invoice issued the next month), dd_represented_transfer 99.6 %, third_party_confirming 100 %, 1toN_text 37 % (needs refs).

**Realistic exact-amount matching, no party given**: name candidates ∩ exact open amount is **99.9 % precise** (12k lines). Amount-only
unique is only 63 % precise and **hits 17 % of `unidentified` and 15 % of `unidentified_lookalike` lines**. So never accept
amount-only without a name, IBAN or ref.

**Batch and file keys** (lines exact = predicted item set equals golden set):
| key | link | ref→item | exact |
|---|---|---|---|
| payment_runs (AP) | `PR…-NNNN` e2e in text/camt e2e; else run id + run total; else run id (ref1 truncated at 16 chars) + transfer_amount | 97 % (docno; disambiguate by document_amount, then beneficiary name; factored runs have the confirming bank as beneficiary) | multi .967, single .969, holding 1.00, CN netted .973, partial .967, factored .885, confirming 1.00 when hit (only 333/788 hit: one debit per maturity date, so group e2e by amount) |
| direct_debit_batches | `batch_id` (11 digits) in text: 100 % of lines | full docno 78 % + `RECIBO 0106536` suffix 21 % | **.965** |
| card_settlements | 9-digit merchant_id in text + Σnet == amount (settlement_id = last 4 of MID + YYMMDD) | order_number = AR docno (**resolve within entity**: WEB numbers repeat across entities) | 1.00 on sample |
| cheque_deposits | `deposit_id` (RCH…/LBX…) in text: 100 % | stub full 68 % + suffix 17 %; 14 % empty stub → drawer name + amount | .63 → 1.00 on sample with drawer fallback |
| remittance_advices | customer payment ref in bank text (`REM/2025/1161`, `PAY-2025-002301`, `2071000409`, `250527638438`) found in advice | doc list + paid column | sample: RA found for 84 % of 1toN_remittance and 57 % of short_dispute lines. The rest arrive after the month or use the customer's own voucher numbers |
RA formats: csv (`document;doc_date;gross;discount_or_deduction;reason;paid;your_order`), txt (e-mail), xlsx (sharedStrings),
pdf (ReportLab: ASCII85+Flate; a stdlib extractor is in `agent2_pdf.py`, no pypdf needed). Penalty or deduction rows carry a reason
("-388,88 PENAL. RETRASO") → SHORT_DISPUTE, amount = paid.

## 4. Proposed engine
**Order** (deterministic first, mark items consumed, keep `open` cents per item; seed from `open_items_at_start.open_amount` in LIVE/FINAL):
1. Non-item lines: category classifier (other agent). Run it first on clear patterns (cash pool, fees, payroll, FX deal, IC) so they never
   reach the matcher.
2. AP debits: payment_runs. Try e2e, then run total, then run id + amount, then confirming per maturity group, then FX via `orig_amount`
   (`/OCMT/USD7637,88/EXCH/`) = Σ item amounts → FX_DIFFERENCE. Fallback: supplier prefix name + exact amount (supplier DD, refunds
   of APC: "ABONO AB-INV0006647", "CREDIT NOTE AB-2025-0084" → APC docno).
3. AR batches: DD batch_id; card MID + net; cheque deposit_id; returns ("IMPAYE PRLV MS02 RUM VTM-FR-0002383",
   "RUECKLASTSCHRIFT MD06 MREF+…") → mandate_id + amount → the item collected earlier → RETURNED. That class is worth 2 pts of exc-F1 plus 0.77 pts.
4. Remittance advices: index every payment ref in each advice. Match bank ref to advice (allow advice received up to ~10 d *after*
   booking). Rows = docs in advice. paid < gross with a reason → SHORT_DISPUTE, a discount column → SHORT_DISCOUNT.
5. AR transfers. Candidate parties = PrefixIndex(cp_name + text) ∪ IBAN(master + learned) ∪ parties of referenced docnos ∪
   children/parent of those. Candidate items = open AR items (ARI and ARC, ARC negative) of those parties with issue ≤ booking, same currency
   (or FX conversion with the fx_rates spread ≤ 3 %).
   Scoring cascade (stop at the first hit):
   a. referenced docnos (full / 6-digit suffix / fuzzy ≤1 edit on party docnos) with Σ open == cash;
   b. one exact open amount (prefer referenced, else oldest by issue date; a month in text → invoice issued the following month);
   c. refs ∪ subset-sum of the rest: meet-in-the-middle n ≤ 22–26 (2^13 each side, ~1 ms). For larger pools, first restrict to
      due ≤ booking+60 d, or FIFO prefix by due/issue date. Cap at 2 solutions and prefer the one with the most referenced items and the oldest items;
   d. discount: cash == round(total × (1−pct)) and booking ≤ issue+disc_days+3 → SHORT_DISCOUNT, else PARTIAL with amount = cash;
      withholding: cash == total × (1−r), r ∈ {1.25, 1.5, 10 %} for MX/BR/MA customers → SHORT_WITHHOLDING; non-SEPA shortfall
      ≤ 60 EUR and ≤ 2 % → SHORT_BANK_CHARGES;
   e. one referenced item, cash < open → PARTIAL (instalment "INSTALMENT 3/4" is PARTIAL except the last), cash > open → OVERPAYMENT;
      already fully paid → DUPLICATE_PAYMENT;
   f. else UNIDENTIFIED (never amount-only).
**Confidence**: name+amount unique 0.999; refs-sum 0.99+; subset unique 0.97; FIFO among same-amount items 0.87–0.96; amount-only 0.63
(reject). Only TP pairs count, so a wrong guess costs precision but an abstention costs only recall. Still emit any candidate with
confidence ≥ ~0.5, because an UNIDENTIFIED line that really has items is wrong anyway.

## 5. Reachable accuracy (estimate, LIVE-like)
AP runs/holding/CN-netted ~97 %; DD batch ~96 %; card/cheque ~98 %; ar_1to1 99 %; ar_ref_noise 84 % now → ~90 % with fuzzy refs and
subset pruning; same_amount_fifo 97 %, _month 67 → 90 % with month parsing; RA segments 35–55 % → 80 % if advices parse fully
(+5–6 pts alloc-F1, the **biggest single gain left**); ar_1toN_text 76 → 90 % with tail-list parsing; partial, instalment and retention
40–50 % (needs PDF clauses). Exceptions: adding RETURNED, FX, WITHHOLDING, DUPLICATE and BANK_CHARGES should take exc-F1 from 0.17 to about 0.6.
**Realistic target: about 80 on LIVE** (prototype at 64 on a realistic month; the gap is mostly RA, exceptions, ap_fx and returns).
