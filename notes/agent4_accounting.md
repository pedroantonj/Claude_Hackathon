# Agent 4 — Accounting & exceptions (Phase A analysis)

Everything below was checked against `train/golden_allocations.csv.gz` (1,489,931 bank lines, 5.4M rows) and `sample/`.
Prototype code: `bankrec/notes/agent4_proto.py` (`classify_hard`, `classify_soft`, `decide`, `subset_sum`, `hu`).
Scratch caches were in `bankrec/notes/.a4cache/` and can be deleted.

## 1. No-item lines → `category` (vectorised regex on `upper(text)` + sign, first match wins)

**Hard rules.** These fire on bank fields alone, need no matching, and come first.

| # | category | regex on text (upper) | sign | train n | prec | rec |
|---|---|---|---|---|---|---|
| 1 | BANK_REVERSAL | `ANULACION OPERACION INDEBIDA` + **partner**: same account, amount = −reversal, booked 0–10 d before, text = tail after `N <digits> ` (strip spaces/`\|`) | ± | 2,664 | 1.00 | 1.00 |
| 2 | CASH_POOL | `CASH CONCENTRATION\|ZBA SWEEP\|CASH POOLING\|TRASPASO ENTRE CUENTAS\|TRASPASO PROPIO` | ± | 12,576 | 1.00 | 1.00 |
| 3 | INTERNAL_TRANSFER | `TRASPASO (A\|DE) CTA` | ± | 1,936 | 1.00 | 1.00 |
| 4 | INTERCOMPANY | `VALTERM\|INTRAGRUP\|DEVOLUCION PRESTAM\|TRANSF\. IC\|FINANCIACION IC\|PRESTAMO IC` | ± | 3,187 | 1.00 | 1.00 |
| 5 | FX_DEAL | `FX DEAL\|COMPRA DIVISA\|VENTA DIVISA` | ± | 1,551 | 1.00 | 1.00 |
| 6 | PAYROLL | `NOMINA\|PAYROLL` | − | 3,897 | 1.00 | 1.00 |
| 7 | SOCIAL_SECURITY | `SOCIAL CHARGES\|TGSS\|CUOTAS DE LA SEGURIDAD SOCIAL` | − | 816 | 1.00 | 1.00 |
| 8 | TAX_REFUND | `DEVOLUCION AEAT` | + | 4 | 1.00 | 1.00 |
| 9 | TAX_PAYMENT | `IMPUESTOS MOD\|TAX PAYMENT\|VAT RETURN\|TAX AUTHORITY\|DGFIP` | − | 957 | 1.00 | 1.00 |
| 10 | LOAN_DRAWDOWN | `DISPOSICION POLIZA` | + | 32 | 1.00 | 1.00 |
| 11 | LOAN_REPAYMENT | `AMORTIZACION (DE PRESTAMO\|POLIZA)\|^PRES\.[A-Z]{2}\d` | − | 832 | 1.00 | 1.00 |
| 12 | LEASING | `CUOTA (LEASING\|ARRENDAMIENTO\|LSG\|RENTING)` | − | 2,832 | 1.00 | 1.00 |
| 13 | INSURANCE | `INSURANCE PREMIUM\|PRIMA SEGURO\|SEGUROS MULTIRRIESGO` | − | 220 | 1.00 | 1.00 |
| 14 | CARD_EXPENSES | `ADEUDO MENSUAL DE TARJETA` | − | 718 | 1.00 | 1.00 |
| 15 | INTEREST_INCOME / _EXPENSE | `^LIQUIDACION INTERESES \dT\d\d` → split by sign (+ income, − expense) | ± | 320/181 | 1.00 | 1.00 |
| 16 | BANK_FEES | `^COMISION (TRANSF\|DEVOLUCION\|REMESA\|MANTENIMIENTO)\|^COMISIONES SERVICIOS\|SERV\. EM\. TRANSF\|^FRAIS VIREMENT\|^COM\. TRANSF\|MAINTENANCE FEE\|^LIQUIDACION CTA\.CTO\|^LIQUIDACION DE INTERESES, COMISIONES` | − | 70,152 | 1.00 | 1.00 |

Traps:
- The *original* line of a bank error has a bare text (`RECIBO`, `ABONO`, `COMISION`, `SEG. SOCIAL COTIZ.`,
  `TRANSFERENCIA RECIBIDA`). Use the partner rule to label it. Do not run a keyword rule for these texts.
- `DEVOLUCION RECIBO …` (−) is a DD return, so it gets RETURNED on items, not a category. Its separate fee line
  `COMISION DEVOLUCION RECIBO` is BANK_FEES.
- `REEMBOLSO FRA …` (+) is a supplier refunding its credit note, so it settles the **APC item** (segment
  `ap_supplier_refund`, 1.2 % of lines). It is not SUPPLIER_REFUND.

**Soft rules.** Use these only if the matcher found **no** item.

| category | rule | train n | prec | rec |
|---|---|---|---|---|
| CUSTOMER_ADVANCE | (+) `PROFORMA\|\bPF\d\d-` | 2,008 | 1.00 | 0.83 |
| UNIDENTIFIED | (+) cash deposits `INGRESO EFECTIVO\|INGRESO CAJERO\|VERSEMENT ESPECES\|VERSAMENTO CONTANTI\|DEPOSITO EN EFECTIVO`, **and `ANTICIPO` without PROFORMA** (891 UNID vs 0 advances) | 34,653 | 1.00 | 0.10 |
| SUPPLIER_REFUND | (+) `REEMBOLSO(?! FRA)\|ABONO PAGO DOBLE\|DEVOLUCION PAGO\|PAGO DUPLICADO\|DUPLICATE` | 483 | 1.00 | 0.78 |
| CUSTOMER_REFUND | (−) `PAGO DUPLICADO\|DUPLICATE\|DEVOLUCION PAGO` | 2,611 | 1.00 | 0.44 |

What is left over goes to fallbacks, after item matching fails:
- **(+) → UNIDENTIFIED.** Misses: 346 CUSTOMER_ADVANCE (bare `ACH CREDIT … PAYMENT ID`, to be found by
  proforma PDF amount) and 108 SUPPLIER_REFUND (bare ACH).
- **(−) → CUSTOMER_REFUND.** This means an outgoing transfer to a customer name/IBAN with no AP or payment-run
  match. Negative "VENDOR PMT / SEPA-UEBERWEISUNG AN" lines to customers are 1,456 refunds.

**Measured results.**
- Rule precision is **100 %** on train (109,494 lines fired) and on sample (740 lines).
- Recall: the hard rules alone cover **77 %** of non-item lines (stable 2021–2025).
- With the fallbacks, and assuming item lines are matched perfectly, **99.68 %** of train non-item lines are
  correct (sample: 100 %).
- LIVE: the rules fire at the same rate every month, including MONC after the camt switch (2025-10). The fee,
  pool and payroll texts are unchanged.

## 2. Exceptions: verified formulas

The scorer compares the **amount = open amount** for full clears. Every rule below needs the **open balance**
of the item at line time: total − earlier PARTIAL/SHORT_DISPUTE/plain applications + RETURNED re-openings.
Reconstructing open balances from golden history agrees with golden amounts on 99.94 % of full-clear rows.

| exception | verified rule on golden | evidence |
|---|---|---|
| SHORT_DISCOUNT | terms D2N30/D1N60/D2N30US; cash = T − **round-half-up**(p·T) (`np.round` fails at x.xx5, e.g. 70.25·2 %=1.41); booking ≤ issue + dd + 3 | single-invoice: cash==formula & in-window → 39,205/39,205 SD; cash==formula & late → 3,257/3,257 PARTIAL (amount = cash = T−disc). Multi-invoice lines: discount is **per invoice**, only some invoices take it → subset-sum |
| late discount | PARTIAL, amount = T − disc (paid/open = 0.98 or 0.99) | segment ar_discount_late, ar_1toN_remittance |
| SHORT_WITHHOLDING | shortfall = rate·T per invoice; rate fixed **per customer**: MX01 1.25 %/10 %, BR01 1.5 %/10 %, MA01 (and ES01 MA customers) 1.5 %/10 % | 100 % exact with both roundings. Only ~15 % of MX/BR/MA customers withhold, and they do it 97 % of the time → **learn a customer→rate table from train**. 10 % collides with 10 % retention (MX) → use SW only for known withholders |
| SHORT_BANK_CHARGES | single invoice, non-SEPA/`GASTOS SHA`/`WIRE IN`/AUD/CAD/USD; shortfall **≤ 60 EUR-equiv** (fx on booking date). Golden does **not** enforce 2 %: SBC seen up to 3.15 %, PARTIAL starts at 3.5 % → threshold **≤ 3.3 %** | shortfall ≤60 EUR & ≤3.3 % → 1,130 SBC / 0 other; the fee is a round EUR amount (10/15/20/25/35/45) converted |
| OVERPAYMENT | single invoice, cash > open by any amount (min 0.01) | 2,838/2,838; never multi-item, never FX |
| PARTIAL | cash < open and none of the above. Paid/open ratios are "round": 0.25/0.3/0.4/0.5/0.6/0.7/0.75 (ar_partial, n_to_m), AP 0.3/0.5/0.7, instalment 1/n of **total** (n=2,3,4,6), retention 0.90/0.95, advance deduction 0.5/0.6/0.7, late discount 0.98/0.99 | used to choose **which** invoice is partial in multi-item lines (newest/oldest/largest heuristics are only ~30 % right) |
| SHORT_DISPUTE | only when the **remittance advice** lists a deduction **with reason** (PRICE DIFF, DAMAGED GOODS, SHORT DELIVERY, PENAL. RETRASO, COOP MKT, LOGISTIC FEE); amount = `paid` column; paid/total 0.70–0.97 | sample CSV advices: 74/76 reasoned deductions = SHORT_DISPUTE (2 settled elsewhere). Only in key-account multi-invoice lines (≈93 rows/line) |
| FX_DIFFERENCE | any **invoice** row with item currency ≠ account currency, fully cleared. It replaces '', SD, SW, SBC (orig_amount ≈ 0.90/0.98·net shows SW/SD hidden inside). PARTIAL/SHORT_DISPUTE keep their code; credit-note rows stay '' | 113,508 FX rows, 0 with same currency; 2,766 '' FX-mismatch rows = exactly the ARC/APC rows |
| DUPLICATE_PAYMENT | item already fully settled by an earlier line; amount = cash = total | 4,927 rows, all single |

**FX conversion.**
- AR receipts carry `orig_amount` in the invoice currency in 97.8 % of FX lines. orig_amount = Σ invoice open −
  Σ CN in 95.5 % of lines. The rest are ×0.90/0.985/0.98, i.e. withholding or discount inside an FX payment.
- When orig_amount is missing (AP aggregated debits mixing currencies): convert with
  `units_per_eur[bcur]/units_per_eur[icur]` on **booking date** (ffill weekends).
- Observed |deviation| vs the ECB reference: 79 % ≤ 1 %, 94 % ≤ 2 %, 96 % ≤ 3 % (the outliers are lines hiding
  SW/discount). Use a **3 % relative tolerance**. AR spread is negative (we receive less) and AP is positive.

## 3. Decision procedure — `decide()` in agent4_proto.py

Input: the item set of a line, the open balances, the cash, and the remittance-advice reasons. The steps run in
this order:

0. Convert the cash to the invoice currency C:
   - orig_amount if orig_currency = icur (abs tol 0.011);
   - otherwise convert at the reference rate (rel tol 3 %);
   - same currency: abs tol 0.011.
1. Apply the RA reasoned deductions. Each such item → SHORT_DISPUTE, amount = paid.
2. N = Σ open(invoices) − Σ open(CN); d = C − N.
3. |d| ≤ tol → all rows '' with amount = open. CN rows are always '' with their full amount.
4. d > tol: single invoice → OVERPAYMENT (amount = open). Multi → ''.
5. d < 0, first match wins:
   - (a) **Discount.** Try, in order: all eligible invoices (terms with discount, open = total), the in-window
     ones only, the late ones only, then a **subset-sum** over the discount amounts in cents. In-window invoices
     → SHORT_DISCOUNT; late ones → PARTIAL with amount T − disc.
   - (b) **Withholding.** Use the known customer rate: Σ hu(rate·T) = −d → all invoices SHORT_WITHHOLDING.
   - (c) **Bank charges.** Single invoice, −d ≤ 60 EUR-eq and ≤ 3.3 % of open → SHORT_BANK_CHARGES.
   - (d) **PARTIAL** on one invoice k with paid = open_k + d: prefer the k whose paid/open or paid/total is in
     {.2,.25,.3,⅓,.4,.5,.6,.7,.75,.9,.95,.98,.99,⅙}, otherwise the newest. If the shortfall is larger than any
     invoice, fill oldest-first and mark the unfilled ones PARTIAL.
6. FX line → every invoice row with '', SD, SW, SBC or OVER becomes FX_DIFFERENCE.

**Reproduction on golden.** Golden item set given, open balances from history, 60k random train lines,
excluding RETURNED, DUPLICATE and card settlements:
**99.04 % of lines** exact (item, amount ±0.5 %, exception), 98.1 % of rows, without remittance-advice reasons.
With the RA reasons supplied (an oracle, i.e. a perfectly parsed advice): **99.66 %** of lines.
- Weak segments: ar_1toN_remittance 0.93, ap_fx 0.955 and ap_partial 0.96 (aggregated mixed-currency AP runs:
  use payment_run document_amount instead of cash arithmetic), ar_instalment 0.976, ar_short_withholding 0.968,
  ar_short_discount 0.986.
- Known residual: in **multi-invoice** lines golden gives SHORT_DISCOUNT up to ~20 days after the window
  (903 rows). The likely cause is the RA payment_date. Single-invoice lines follow the +3 grace rule exactly.
- Without RA parsing, every SHORT_DISPUTE line is lost (0.5 % of lines; 20k rows, so it matters for exception
  macro-F1).

## 4. Credit notes and AP side

- **Credit notes (ARC/APC).**
  - Always amount = **full total**, exception '' (105,114 rows; the only exception is 64 RETURNED inside
    returned AP transfers).
  - Cash = Σ invoices − Σ CN.
  - AR netting: ar_1toN_credit_note (1.6 % of lines), plus CNs inside remittance/dispute/withholding lines.
  - A rappel CN *paid out* by our transfer settles the ARC (ar_credit_note_paid_out, 0.02 %).
  - A supplier refunding its CN settles the APC (ap_supplier_refund, 1.2 %, texts `REEMBOLSO FRA`, `ABONO`,
    `DEVOLUCION`, or none).
- **Payment runs** (`payment_runs/*.csv`).
  - Golden amount = |document_amount| in 99.9 % of joined rows.
  - document_amount < open → PARTIAL (ap_partial; ratios 0.3/0.5/0.7). CN rows are negative: use abs, no
    exception. Item currency ≠ debtor-account currency → FX_DIFFERENCE.
  - Join on (supplier via beneficiary IBAN/name, document_number). The supplier invoice number is **not**
    unique across suppliers.
  - Run types:
    - `transfers`: one debit per end_to_end_id (only ~17 % of e2e ids are visible in bank camt; use
      amount + date + beneficiary).
    - `aggregated`: one debit = Σ transfer_amount of the run, which can mix currencies (lines with no
      orig_amount and huge debits).
    - `confirming`: one debit per maturity date.
- **Payment factory.** ap_paid_by_holding = 2.2 % of lines, 82,599 rows.
  - Debits on ES00-PENA-EUR / ES00-MONC-EUR pay items of ES01 (41k), ES02 (31k) and PT01 (10k). They never mix
    entities within one line, and all rows are ''.
  - ES00 accounts also carry subsidiary CN-netted, DD, factored and partial payments.
  - So for any ES00 debit, the candidate AP items = ES00 ∪ ES01 ∪ ES02 ∪ PT01.
  - Texts: `PAGO TRANSFERENCIAS | 4799-…` (aggregated) or `TRANSFERENCIA A <supplier> | FRA(S) …`.

## 5. Instalments, retentions, advances, replaced invoices: frequency (share of train lines)

| segment | % lines | golden pattern | needs PDF? |
|---|---|---|---|
| ar_instalment | 2.22 | single line; amount = T/n (n=2,3,4,6); PARTIAL except the last ('' or FX) | no for the exception (open-balance rule); helps *find* the item since cash ≠ T |
| ar_partial ("A CUENTA") | 1.86 | 50 % PARTIAL (0.25–0.75 of open), 50 % the remainder '' | no |
| ar_dispute_settlement | 0.52 | later payment of the disputed remainder (0.03–0.3 of T) → '' | no |
| ar_retention | 0.45 | first payment 0.90/0.95·T → PARTIAL | no (but 10 % ≈ MX withholding: use the customer list) |
| ar_retention_release | 0.28 | 0.05/0.10·T ~12 months later → '' | no |
| ar_replaced_invoice ("anula y sustituye") | 0.14 | text cites the cancelled X → allocate to the replacement; normal exceptions | yes (PDF maps X→replacement) |
| ar_advance_deduction | 0.13 | 0.5/0.6/0.7·T → PARTIAL | no |
| customer_advance (proforma) | 0.14 | category; 83 % have `PROFORMA/PF..-` in text | rest via proforma PDF amount |

Verdict:
- Worth implementing early as a **"fraction-of-total" candidate generator**: cash ∈ {T/n, 0.9T, 0.95T, 0.05T,
  0.1T, 0.5–0.7T, remaining open}. Together with open-balance tracking it covers ≈ 5.5 % of lines with no PDF
  parsing.
- Replaced invoices (0.14 %) are low priority.
