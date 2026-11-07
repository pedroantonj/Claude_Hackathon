# Agent 3: fraud and anomaly rules (duplicates, returns, unidentified, reversals, treasury, IBAN)

Data: train golden (5.40M rows, 1.49M bank lines, 2021-07..2025-06), the sample month (2025-06, 8,499 lines), and the
last 12 months of train ("L12", 2024-07..2025-06, 396,957 lines). Precision (P) and recall (R) are **per bank line**
unless stated. Scratch scripts are in `/tmp/a3/` (s*.py, final.py, pdftxt.py). They are not part of the deliverable.
`T` = `upper(text + ' ' + remittance_info)`. Debit = amount < 0.

## 0. Label volumes (golden)
| label | train rows | L12 lines | sample rows |
|---|---|---|---|
| DUPLICATE_PAYMENT (exc) | 4,927 (4,436 ARI + 491 API), always 1 item/line | 1,306 | 35 |
| CUSTOMER_REFUND / SUPPLIER_REFUND (cat) | 2,611 / 483 | | 20 / 4 |
| RETURNED (exc) | 30,358 rows on 28,416 lines | 7,369 | 210 |
| UNIDENTIFIED (cat+exc) | 34,653 (= 2.3% of lines, 3.8% of credits) | 9,186 | 218 |
| BANK_REVERSAL / CASH_POOL / INTERNAL / FX_DEAL / IC | 2,664 / 12,576 / 1,936 / 1,551 / 3,187 | | 18/63/6/0/7 |

## 1. DUPLICATE_PAYMENT (the later line gets the exception; amount = item amount, which equals the line amount in 99.2% of cases)
What the golden shows:
- The original line E is **always a single-item, full, plain settlement**: from segments ar_ref_noise 2,820, ar_1to1 1,580,
  ap_payment_single 415, ap_paid_by_holding 52, and a few others. The duplicate L has the same signed amount
  (99.2%) and the same or near-same text (the bank wording can change, e.g. "TRANSF. A SU FAVOR" vs "TRANSFER INMEDIATA";
  an RMR suffix may appear). Gap E→L: median 7 days, p95 28, p99 49, max 119.
- L is in the **same entity**. It is in the same account only about 94% of the time (US01-HPNB-USD vs -USD-3, AU01-WATT vs -2),
  so **key on entity, not account**.
- Customer side: 90%. Supplier side (we pay twice): 10%.

**Rule DUP (validated):** flag L as DUPLICATE_PAYMENT of item X when all of these hold:
1. An earlier line E in the **same entity** has the **same signed amount**, and the gap is **1–60 days**.
2. E was matched to exactly **one** ARI/API item X, with exception empty or FX_DIFFERENCE, and `|amount| == X.total`
   (a full clearing, not an instalment).
3. Party of X has **only one invoice with that total** (`count(party,total)==1`). This kills fixed monthly fees.
4. Text similarity: Jaccard of the **name tokens** of E and L ≥ 0.5, or the texts are identical. Name tokens are
   `[A-Z]{3,}` minus boilerplate such as ACH CREDIT DEBIT CCD PAYMENT ID SEPA UEBERWEISUNG EREF NOTPROVIDED SVWZ TRANSFER
   TRANSFERENCIA(S) TRANSF FAVOR INMEDIATA DIRECT TRTP OVERBOEKING NAME REMI ORDP BONIFICO CAUSALE PRZELEW PRZYCH TYTUL VIR
   RECU EMIS MOTIF REF INV FRA FRAS PAGO FACTURA INVOICE ORDER RMR VENDOR PMT TT IN OUT SPEI CONCEPTO PIX…
5. X was **not RETURNED** between E and L. A returned DD followed by a transfer, or a rejected AP transfer that we pay again,
   is a normal payment (segments ar_dd_represented_transfer and ap_returned_transfer).
6. The primary matcher found **no other open item** for L. Run DUP only on lines that are still unmatched.
7. (Advised) Skip supplier direct debits (ap_direct_debit gives 29 of the false positives when filter 6 is off).

| variant (train all) | flagged | P | R |
|---|---|---|---|
| filters 1–6, entity key | 4,562 | **0.9998** | **0.926** |
| same, L12 slice (account key) | 1,178 | 0.997 | 0.900 |
| same, 2025-06 (account key) | 126 | 1.000 | 0.900 |
| without filter 6 | 4,623 | 0.987 | 0.926 |
| without filters 5 and 6 | 4,749 | 0.960 | 0.926 |
| account key + boilerplate Jaccard ≥ 0.5 (naive) | 4,300 | 0.981 | 0.856 |
| same account+amount, ≤ 60 days, nothing else | 163,745 | **0.028** | 0.94 |

The last row is the trap: identical amount and text recurring within 60 days is **mostly** fixed monthly fees
(ar_same_amount_fifo and ar_same_amount_month, about 90k lines) and repeat orders. Filters 2 (full clearing),
3 (unique amount per party) and 6 are what make the rule precise. Instalments fail filter 2 because the line amount is
smaller than the invoice total.

**Refunds that follow:** the refund is a line of the opposite sign in the same account, gap 10–60 days
(median 27 days customer side, 32 supplier side). 99.8% are in the same account as the duplicate.
| rule | P | R |
|---|---|---|
| CUSTOMER_REFUND = debit & `DUPLICAD\|DUPLICATE\|PAGO DOBLE` | 1.000 | 0.962 (sample 0.85) |
| SUPPLIER_REFUND = credit & `REEMBOLSO\|DEVOLUCION PAGO\|DUPLICAD\|PAGO DOBLE` & NOT `REEMBOLSO FRA` | **1.000** | **1.000** |
| structural: −amount of a DUP line, same account, 5–90 days | 0.997 | 0.983 (both refund categories) |
| structural OR dup-keyword | 0.997 | 0.999 |
- `REEMBOLSO FRA <n>` on a credit (3,583 lines) is **not** SUPPLIER_REFUND. It is a supplier paying back its credit note:
  allocate it to the APC item with no exception.
- The CUSTOMER_REFUND misses (3.8%) are bare N43 "TRANSFERENCIA A <customer name>" lines. The structural rule catches them.

## 2. RETURNED (credit-side reversal of a collection or payment; amount = the original allocation re-opened)
| segment | lines | sign | items/line |
|---|---|---|---|
| ar_dd_returned | 26,397 | debit | 1 |
| ap_returned_transfer | 1,192 | **credit** | mean 2.6, max 29 |
| ar_cheque_returned | 538 | debit | 1 |
| ar_card_chargeback | 289 | debit | 1 |

**Detection (P = R = 1.000 on train, L12 and sample):**
- Codes: `bank_code ∈ {PMNT/RDDT/UPDD, PMNT/ICDT/RRTN, PMNT/CCRD/CHRG, PMNT/RCHQ/UPCQ, NRTI}` or
  (`14` & subcode ∈ {021, 110, 370, 908}). Subcode 908 covers 75 AP returns that carry no reference.
- Regex: `^RETURN |SDD RETURN|RUECKLASTSCHRIFT|^DEVOLUCION RECIBO|^DEVOL\.RECIBOS|^DEVOLUCION TRANSFERENCIA|CHARGEBACK|CONTRACARGO|RETURNED ITEM|IMPAYE`
  (P = R = 1.000). Do **not** use a bare `DEVOLUCION`: it is only 26% returns.
  `^COMISION DEVOLUCION RECIBO` is BANK_FEES (3,728 lines, P = 1.0 for BANK_FEES).
  `DEVOLUCION PAGO DUPLICADO` is a refund (section 1). `DEVOLUCION PRESTAMO` is INTERCOMPANY.
- Reason codes seen: AM04 10.7k, MS02 4.3k, MD06 2.4k, AC04 1.9k, AC06 1.2k, AC01 1.0k, MD01, MS03, MD07, BE04, SL01, AM05;
  cheques: REFER TO MAKER, SIN FONDOS, NSF, STOP PAYMENT. **Any reason code still means RETURNED.**

**Linking to items:**
- **DD return**: the mandate `VTM-[A-Z]{2}-\d{7}` is in 100% of texts.
  Candidates = DD-batch rows with the same `mandate_id` and the same amount (= |line|), with collection_date between
  **1 and 140 days** before the return. Take the **most recent**; its `remittance_info` gives the document number.
  This picks the right item in 85.5% of cases. Requiring a gap ≥ 0 instead of ≥ 1 adds no correct picks, because the
  same-day collection is a new one. Golden RETURNED amount = line amount in 100% of cases.
  51% of returns have more than one candidate (fixed monthly DD). After a return the item is re-collected
  (`re_presentation=Y`, 13,370 rows) or paid by transfer. **Both are plain settlements, never DUPLICATE.**
- **AP returned transfer**: `PR\d{12}-\d{4}` (payment-run end_to_end_id) is in 93.7% of texts. Return every document of
  that e2e from `payment_runs` (doc count = golden in 89%). With no ref (N43 14/908), fall back to beneficiary name + amount
  against our earlier outgoing debit.
- **Chargeback**: `\*{4}(\d{4})` + |amount| → `card_settlements` (card last-4, gross) settled before the line.
  282/289 give a unique match.
- **Returned cheque**: `CHK (\d+)` + amount → `cheque_deposits.cheque_number`. 460/538 give a unique match.
  The rest are MT940 NRTI variants, which need another pattern.

## 3. UNIDENTIFIED (one row: category UNIDENTIFIED + exception UNIDENTIFIED; always a credit)
Segments: `unidentified` 24,299 and `unidentified_lookalike` 10,354 (the amount equals some unrelated open invoice).
About 3.8% of credits, flat across entities (3.9–4.2%; US01 has the most by volume). Median amount 632 (L12).
UNIDENTIFIED is mainly a **residual**: about 50% come from payers whose name matches a known customer, so "unknown payer"
alone gives only P ≈ 0.09.
High-precision positive triggers:
| rule (credits) | P (train/L12/sample) | R (train) |
|---|---|---|
| cash deposit `ESPECES\|INGRESO CAJERO\|INGRESO EFECTIVO` | 1.000 / 1.000 / 1.000 | 0.06 (sample 0.20) |
| `ANTICIPO` and not `PROFORMA` | 1.000 | 0.026 |
| `PROFORMA` → **CUSTOMER_ADVANCE**, not UNID | 1.000 | 1.000 of CUSTOMER_ADVANCE |
| round amount (`amt % 100 == 0`) & not a treasury category | 0.980 / 0.988 / 1.000 | 0.21 |
**Most important guard (lookalikes):** in genuine AR 1:1, ref-noise, FIFO and 1:N matches, a name token of the invoice
customer (invoice name or current master name or trade name) appears in the bank text or counterparty in **99.85%** of
cases. For lookalike UNID lines, the lookalike invoice's customer name appears in only **1.2%**.
=> **Never allocate on amount alone.** Require name, IBAN or reference evidence, or a known third-party pattern
(parent, CONFIRMING … ORD., EXPTE agency). Otherwise classify as UNIDENTIFIED.
Generic remittance values are spread evenly across UNID lines (ORDER 3.2k, PAYMENT 3.2k, REF n 3.2k, TRANSFER 3.2k,
INVOICE 3.1k). They are not discriminative on their own, because ar_ref_noise uses the same words. Use them only in
combination with "no amount or subset fit among the payer's open items".
Known-payer UNID vs PARTIAL: real partials usually carry wording or a reference ("BALANCE…", "RESTANTE PAG FT n",
"PAYMENT INV n", "PRESTACAO 2/4"). A known payer with no reference and no fit among its open items → UNIDENTIFIED.

## 4. Treasury non-item pairs (all keyword rules P = R ≈ 1.000 on train, L12 and sample)
- **BANK_REVERSAL** (both legs): the reversal matches `^ANULACION OPERACION INDEBIDA N [\d |]+<SUFFIX>` (1,332 lines).
  The wrong original is the line in the **same account, with the opposite amount, gap 0–8 days, whose text is exactly
  SUFFIX**. Pairing with `suffix == original.text` gives exactly 1 candidate and P = R = 1.000. Pairing on amount alone
  is ambiguous in 2% of cases, because CASH_POOL sweeps can have the same amount.
  Original texts are bare one-liners: `RECIBO|COMISION|SEG. SOCIAL COTIZ.|TRANSFERENCIA RECIBIDA|ABONO` (1,332/1,332,
  and these bare texts never occur elsewhere). **Do not classify a bare "COMISION" as BANK_FEES or "SEG. SOCIAL COTIZ."
  as SOCIAL_SECURITY.** Codes: NMSC, PMNT/MCRD/OTHR, N43 02/98.
- **CASH_POOL**: `CASH CONCENTRATION|ZBA SWEEP|CASH POOL|TRASPASO PROPIO|^TRASPASO ENTRE CUENTAS` (P = R = 1.0).
  Note that "TRASPASO ENTRE CUENTAS ES00" is CASH_POOL, not INTERNAL. Codes CAMT/ACCB/ZABA and N43 99 (622/867).
- **INTERNAL_TRANSFER**: `^TRASPASO (A|DE) CTA \d{4}` (P = R = 1.0). The legs pair within the same entity and currency:
  same |amount|, opposite sign, same day (all 968 debits pair on gap 0).
- **FX_DEAL**: `FX DEAL|COMPRA DIVISA|VENTA DIVISA` (P = R = 1.0). Both legs are FX_DEAL. The deal number links the legs.
- **INTERCOMPANY**: `\bIC VALTER|(TRANSF\.|FINANCIACION|PRESTAMO) IC\b|PRESTAMO INTRAGRUPO|PRESTAMO VALTERMIA`
  (P = 1.0, R = 0.996). This includes truncated CHAPS text "TRANSF. IC VALTERM". Every line that mentions VALTERMIA is
  INTERCOMPANY (2,982/2,982).

## 5. IBAN changes (handle safely, as a soft key, never a hard filter)
- `pdfs/index.csv` has `document_type=bank_account_notice`: 633 in train (only 172 files present locally), 79 in live.
  **All are customers** (party_id C…). There are no supplier-IBAN-change notices, so no vendor-fraud scenario is planted.
- Notice text (EN or ES): "from DD/MM/YYYY our payments will be made from account NEW (previously OLD)" or
  "a partir del DD/MM/YYYY nuestros pagos … desde la cuenta NEW, en sustitución de la anterior OLD".
  Kind "dd" ("los recibos domiciliados deberán cargarse … Referencia de mandato VTM-…"): 62 of 267.
  The effective date is always the notice date + 10 days.
  Parse with `pdftxt.py` (zlib + ASCII85, pure stdlib, works on the ReportLab PDFs): first IBAN = new, second = old.
- The master IBAN equals the notice's NEW IBAN in only 14% of cases and the OLD IBAN in 8%. **The master is stale.**
  Keep an IBAN set per customer = master ∪ every notice IBAN (old and new), valid from the effective date onwards.
  **Both old and new stay valid** (policy §5.11; IT01 changes its own collection account on 2026-02-01 and customers keep
  paying into the old account, so never restrict matching by our receiving account).
- Segment ar_new_iban (1,951 rows, 1,122 lines) has **no exception**. These are name-only payments (empty /MOTIF, no ref)
  from a new payer IBAN. Match them by name + amount as usual. Only 26% show the IBAN at all (camt cp or "ORDENANTE ES…").
- Our outgoing side: 124 suppliers are paid **consistently** to an IBAN not in the master (master is stale, 0.6% of
  transfers). AP returns with AC01/AC04 ("wrong account") are not enriched on those IBANs (13+8 of 37 vs base 3.4%).
  This deserves a weak audit mention, but there is **no evidence of planted payment fraud**.

## 6. Audit-report anomaly flags (not scored; keep them light)
- Duplicates detected (rule DUP) plus whether a refund was observed. In train, about 59% of customer duplicates are refunded
  within about 60 days (2,611 CUSTOMER_REFUND / 4,436 AR duplicates); on the supplier side it is 98% (483 / 491).
- Unidentified receipts aged above 30 days, with amount and payer. Cash deposits and round amounts (≥ 98% UNID).
- Returned DDs by reason code (AM04 insufficient funds dominates; MD06/MD01 mandate disputes are the risky ones).
- Bank errors (ANULACION pairs): count per bank and the days the error stayed open (0–8).
- Amount outliers: a credit more than 10× the payer's median (payers with ≥ 6 lines) flags 2.2% of lines in L12, mostly
  legitimate 1:N payments. **Use only as a soft report item.**
- Payments to or from an IBAN not seen before for that party (unless announced by a notice).

## 7. Recommended order in the pipeline
1. Treasury keyword rules (§4), BANK_REVERSAL pairing, RETURNED detection and linking (§2), refunds (§1), PROFORMA → CUSTOMER_ADVANCE.
2. Primary matching that requires name, reference or IBAN evidence.
3. On the residual credits and debits: rule DUP against already-settled items (§1).
4. Remaining unexplained credits → UNIDENTIFIED (do not force amount-only matches).
