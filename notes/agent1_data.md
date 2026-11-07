# Agent 1 — Data inspection (Phase A)

Loader prototype: `bankrec/notes/agent1_loader_proto.py` (`load_bank`, `load_items`, `load_batches`, `extract_pdf_text`,
`pdf_spans`, `pdf_rows`). All CSVs: UTF-8, comma, header, read with `dtype=str, keep_default_na=False`.

## 1. Inventory

| file | sample (2025-06, ES01/DE01/FR01 only) | train (2021-07..2025-06) | live (2025-07..12) |
|---|---|---|---|
| bank_transactions | 8,499 rows / 10 accts | 1,489,931 / 48 accts / 17 ent | 203,083 / 49 accts / 18 ent |
| ar_invoices | 50,775 | 3,963,735 (issue 2021-03-01..) | 487,243 (issue 2025-07..12) |
| ap_invoices | 13,709 | 1,530,203 | 199,414 |
| credit_notes | 1,362 | 150,215 | 19,500 |
| open_items_at_start | – | – | 325,419 (ARI 229,534 / API 68,356 / ARC 25,434 / APC 2,095) |
| golden_allocations | 36,574 rows | 5,403,411 rows (5,260,781 item rows) | – |
| direct_debit_batches | 9 files / 5,184 rows | 1,440 / 713,554 | 180 / 93,786 (1,590 batches) |
| card_settlements | 2 / 1,150 | 240 / 147,622 | 30 / 19,110 (1,265 settlements) |
| cheque_deposits | 2 / 164 | 816 / 42,284 | 102 / 6,083 (1,967 deposits) |
| payment_runs | 4 / 9,132 | 1,056 / 1,420,037 | 132 / 205,713 (386 runs) |
| pdfs | 2,298 | 33,204 | 16,688 |
| remittance_advices | 331 (123 csv,123 pdf,73 txt,12 xlsx) | 13,440 | 6,890 (2,381 csv, 2,196 pdf, 1,815 txt, 497 xlsx) |
| bank_statements raw | 10 | 574 (n43/sta/xml) | 287 (84 n43, 109 sta, 94 xml) |

Sizes (gz): train bank 64 MB, ar 141 MB, ap 58 MB, golden 63 MB; live bank 8.9 MB, ar 17.5 MB, ap 7.5 MB, oi 7.1 MB.
Train and live have the same folder structure (pdfs + remittance advices in both). Train-only: golden. Live-only:
`open_items_at_start`, account `ZA01-UMVC-ZAR` (ZA01/ZAR; only 2 lines in live).

Exact columns:
- bank_transactions: `bank_txn_id, account_id, entity, bank, statement_format, statement_file, booking_date, value_date,
  amount, currency, bank_code, bank_subcode, ref1, ref2, text, counterparty_name, counterparty_account, end_to_end_id,
  remittance_info, orig_amount, orig_currency`
- ar_invoices: `item_id, entity, document_number, issue_date, due_date, customer_id, customer_name, currency, net_amount,
  tax_amount, total_amount, payment_terms, customer_po, payment_method`
- ap_invoices: `item_id, entity, supplier_id, supplier_name, supplier_invoice_number, issue_date, due_date, currency,
  net_amount, tax_amount, total_amount, payment_terms, our_po, payment_method`
- credit_notes: `item_id, side, entity, document_number, issue_date, party_id, party_name, currency, net_amount,
  tax_amount, total_amount, related_document, reason` (no due_date)
- open_items_at_start: `item_id, item_type, entity, party_id, document_number, issue_date, due_date, currency,
  total_amount, open_amount` (**no party name, no terms/PO** → names via masters or train files)
- golden: `bank_txn_id, item_id, amount, category, exception, segment, difficulty`
- batch files: see DATA_DICTIONARY (verified identical).

Field population by statement_format (live): **n43** (37,980): ref2 + bank_subcode + text, no ref1/counterparty/e2e;
**mt940** (59,565): ref1 (16 ch, often `NONREF`) + ref2 (bank ref) + text (`:86:`); **camt053** (105,538): ref1 (bank ref),
counterparty_name (98.7k), counterparty_account (25.6k), end_to_end_id (92.9k, 49k = `NOTPROVIDED`), remittance_info (86k).
orig_amount/orig_currency on 2,278 live lines (USD 1,232 / EUR 523 / CAD 523).
Live format by month: n43 ~8.3k/mo Jul–Sep → ~4.9k Oct–Dec (Moncayo → camt053 from 2025-10).

## 2. Relationships bank line → batch/run (verified)

| batch | key in bank | bank amount | coverage |
|---|---|---|---|
| direct_debit_batches | `batch_id` in **text** (all fmts) and N43 **ref2** `REM<batch_id>` | = `batch_total` (100%) | 100% of batch_ids found as token (SEPA/ACH/BACS/BECS) |
| card_settlements | `settlement_id` mostly **NOT** in bank (0% live). Key = `merchant_id` + settlement_date + n TRX | = **sum(net)** (98.8% sample); allocations at **gross** | match on account+date+MID+sum(net) |
| cheque_deposits | `deposit_id` in text (100%) | = `deposit_total` (100%) | |
| payment_runs aggregated / confirming | `run_id` in text/ref2 (100%) | aggregated = one debit per run; confirming = one debit per maturity date ("n PROVEEDORES") | |
| payment_runs transfers | `end_to_end_id` (`<run_id>-NNNN`): camt `end_to_end_id` 100%; mt940 `/REF ...` in text 35% + ref1 = first 16 chars `PR010025061208-0`; n43 ref2 = 16-char truncation `PR000025112545-0` (17% full) | = -transfer_amount (one row per document; dedupe by e2e) | camt 100%, others: run prefix + amount |

Examples (sample):
- DD camt: `BT-2025-0179976 DE01-RHEI-EUR +24516.95 PMNT/RDDT/ESDD "SEPA-LASTSCHRIFT EINZUG 02250613728 39 POSTEN"`
- DD n43: `BT-2025-0185390 ES01-PENA-EUR +30676.98 03/045 ref2=REM01250620841 "ABONO REMESA SEPA CORE 01250620841 | PRESENTADA 250620 42 RECIBOS"`
- DD live ACH `"ACH DEBIT ORIGINATION BATCH 06251202264 63 ENTRIES"`, BACS `"SDD COLLECTION BATCH 01250902402 89 ITEMS"`
- DD return (n43): `-268.62 14/021 ref2=DEV01250617840 "DEVOLUCION RECIBO AM04 REM 01250617840 | MANDATO VTM-ES-0005671 | VICTOR PARDO BELTRAN"`;
  batch fee: `-5.55 17/310 "COMISION REMESA 05250624997 37 RECIBOS"`
- Card: `FR01-LORV-EUR mt940 +5028.63 "REMISE CB 993110427 DU 300525 17 OP"` (17 orders, sum net 5028.63, gross 5074.28);
  n43 `"LIQUIDACION REMESA DE COMERCIOS | COMERC 021899465 REM 9465250611190369"`, truncated variant `"ON 097759027 2306"`;
  live camt `"CARD SETTLEMENT MID 762279862 2025-07-31 10 TRX"` (date in text = operation date, booking = settlement_date).
- Cheque: `FR01-RHEI-EUR +6641.68 PMNT/RCHQ/CCHQ "CHEQUE DEPOSIT RCH25061901431 10"`
- Confirming: `ES01-RIBA-EUR -34895.45 99/824 ref2=CF010025061071 "PAGO CONFIRMING | REMESA CF010025061071 3 PROVEEDORES"`
- Aggregated (live): `ES01-RIBA-EUR -2474428.71 ref2=PR010125111010 "TRANSF. EMITIDAS FICHERO C34 PR0101251 | 11010 | (268 ORD.)"`
  — run_id split by ` | `! Run file = 268 e2e: EUR 2,385,047.71 + USD 103,121.59 (USD part converted by bank).
- Transfer camt: `DE01-RHEI-EUR -58875.39 e2e=PR010025061275-0083 rmt="FRAS INV0005403 INV0005461 ..."`
- Transfer mt940: `ref1=PR010025061208-0 "VIR SEPA EMIS /A TELECOM INTEGRAL SARL /MOTIF FRAS 2025-0038 2025-0039 /REF PR010025061208-0084"`
- Transfer n43: `ES00-PENA-EUR -4677.53 04/102 ref2=PR000025112545-0 "PAGO TRANSFERENCIAS | 4799-56-9098464-94 802652 000001"`

Batch row → item keys: card `order_number` = AR `document_number` (100%); DD `remittance_info` = AR document_number (~90%);
cheque `stub_reference` = doc number (71%, 15% empty, rest noisy); payment_runs `document_number` = AP
`supplier_invoice_number` (INVOICE, 79% exact in-entity) / credit-note `document_number` (CREDIT_NOTE, negative
`document_amount`). Payment runs of ES00 (payment factory) pay ES01/ES02/PT01 items. 29% of transfer rows have
currency ≠ debtor account currency (FX). 11 run_ids span >1 file; 1 deposit_id spans 2 files.
In the sample golden one DD batch maps to several bank lines (batch credit + later returns/re-presentations).
RA `payment_ref` appears in bank text/remittance_info for 78% of sample advices (csv 92%, txt 93%, xlsx 92%, pdf 54% —
Spanish PDF layout uses `Ref.` label, not parsed in that test); the rest arrived for cash outside the sample month.

## 3. Open items for LIVE

`items_live = open_items_at_start (325,419, open_amount given) ∪ live ar/ap/credit_notes (706,157, open = total)` =
**1,031,576** items, no id overlap. Verified: all 325,419 oi ids exist in train item files; all golden item ids
(5.26M rows, 5.12M unique) exist in train item files (0 missing). oi has 32,091 partially-open items; open ≤ total always.
Reconstruction `train total − Σgolden (RETURNED counted negative)` matches oi `open_amount` for 99.7% (mismatch = items
with pre-2021-07 history). 562,877 train items have unallocated balance but only 325,419 are in oi → ~237k items are
closed off-bank (cancelled/replaced invoices + cancelling credit notes, dispute leftovers, etc.) — don't treat
"never allocated" as open in TRAIN simulations; use oi-style logic.
TRAIN item files = all items issued in period + invoices issued before 2021-07 still open (no open_amount column).

## 4. PDFs (stdlib works, 100%)

ReportLab PDFs, Helvetica WinAnsi, content stream `/Filter [/ASCII85Decode /FlateDecode]`: `base64.a85decode` → `zlib.decompress`
→ `BT 1 0 0 1 x y Tm (text) Tj ET` per cell. Decode string bytes latin-1 then cp1252 (em-dash). 2,298 sample PDFs
in 0.9 s, 0 empty; live 16,688 in 13.6 s. `pdf_rows()` groups cells by y and keeps x → table columns.
pdfs/index.csv: `file, document_type, document_number, entity, party_id, document_date`; types (sample): invoice 1,136,
supplier_invoice 999, supplier_credit_note 105, credit_note 32, bank_account_notice 16, proforma 10.
File names: `ARI_<docnum>.pdf`, `ARC_...`, `API_<supplier>_<num>.pdf`, `APC_...`, `PF_...`, `NOTICE_<cust>_<date>.pdf`.

Invoice (DE01/ARI_RE240106864.pdf) lines: `RECHNUNG / Nr. / RE240106864 / Datum 22/04/2024 / Fällig 21/07/2024 / Kunde /
Velasco Merino Gartenbau GmbH & Co. KG / ...items... / GESAMT / 3.871,12 EUR / Zahlungsbedingungen: 90 días fecha factura /
IBAN ... / Retention 5% (193,55 EUR) withheld until final acceptance (22/04/2025). Amount payable at due date: 3.677,57 EUR.`
Credit note (ARC_GS250100201): `GUTSCHRIFT Nr. GS250100201 Ref. RE250107421 ... GESAMT 647,04 EUR`.
Proforma: `PROFORMA-RECHNUNG PF25-0100002 ... PROFORMA INVOICE not a tax invoice. Advance payment for order; the final invoice will deduct it.`
Notice: `Cambio de cuenta bancaria ... a partir del 03/07/2025 nuestros pagos ... desde la cuenta ES22..., en sustitución de la anterior ES94...`
Supplier CN: `FACTURA RECTIFICATIVA / ABONO Nº RF25/04250 Ref. F25/04250 ... Este abono se compensará en sus próximos pagos.`
Clause lines (after terms/IBAN, wrap over 2-3 lines, sample counts): factoring "ha sido cedido a <Confirming X>" / "assigned to"
(~390); retention 5/10 % (~160); "anula y sustituye a la factura X" / "cancels and replaces invoice X" (~200);
"credit note to be deducted from next payment"/"se compensará" (~60); "será reembolsado"/"refunded by bank transfer" (~37);
"IMPORTANT: from dd/mm our bank details change. New IBAN" (~60); "Cuota de mantenimiento ... febrero 2025"/"Maintenance
fee for the month of May 2025" (~33); "A deducir anticipo recibido según proforma PF25-.. de X EUR. Importe neto a pagar Y";
"Pago a cuenta por plazos según contrato" (instalments); "Abono por anulación total de la factura X". Numbers use locale
formats (`3.871,12`, `3 735,43`, `4,155.37`).

## Remittance advices
index.csv: `file, entity, payer_name, received_date, format`. Layouts:
- CSV (`;` or `,`): line1 `payment_ref;<ref>;payment_date;YYYY-MM-DD;currency;EUR`, header
  `document;doc_date;gross;discount_or_deduction;reason;paid;your_order`, last row `TOTAL;;;;;31264.23;`.
  Deduction negative with reason (`-310.23;DAMAGED GOODS`).
- TXT (e-mail): `Subject: Payment advice <ref>` / `Aviso de pago <ref>`; body "payment of 3,844.74 EUR on 05/29/2025 covering:"
  then rows `RE250205413        28/03/25          758.79` (mixed US/EU date & number formats).
- XLSX layout A (sheet `Remittance`): r1 payer, r2 `('Payment advice', 'REM/2025/0829', '2025-06-02')`, r3
  `Document, Date, Gross, Deduction, Reason, Paid`. **Layout B (new, live, 178 files)**: `Voucher, Invoice, PO, Inv. date,
  Amount, Deduction, Net paid`, invoice number **without prefix** (`100974`), PO, and TOTAL row puts total in Deduction col.
- PDF EN: `REMITTANCE ADVICE / Payment ref. / <ref> / Date / Payee / Amount / Document Date Gross Deduction Paid`;
  PDF ES: `AVISO DE PAGO / Ref. / REM/2025/1161 / Fecha / Beneficiario / Importe / Documento Fecha Bruto Dto./ded. Pagado`.
  Empty deduction cells are absent → use `pdf_rows` x-positions.
Payment refs: 12-digit numbers, `REM/2025/NNNN`, `PAY-2025-NNNNNN`, `9053060237`; they appear in bank text
(camt `SVWZ+<ref>`, remittance_info, mt940 `/REMI/`, `/EREF/PAY-...`).

## 5. Data quality
- Amounts: all 2 dp strings, signed (+credit/−debit, live 124,428 / 78,655, no zeros). Item totals always positive
  (credit notes positive); payment_runs `document_amount` negative for netted credit notes (4,368 live).
- N43 `text` = 38-char fields joined with ` | ` (85.5% of n43 lines); tokens/words are cut at the boundary
  (`PR0101251 | 11010`, `FRAS FV25-031026` in ref2 vs `FV25-0310261` in text). Search on text with ` | ` removed AND kept.
  N43 is ASCII-only uppercase (accents stripped: `CLIMATIZACION`); mt940 21% non-ASCII, camt 4%. No mojibake found.
- ref fields truncated at 16 chars (N43 ref2, mt940 ref1). camt end_to_end_id `NOTPROVIDED` 49k.
- Document numbers NOT unique across entities: train AR 507k dup doc numbers across entities (0 within entity);
  AP supplier_invoice_number unique only per supplier; credit notes 3,062 dup within (entity, side). Key by entity(+party).
- oi has no party names; masters names are current (renamed customers) — printed names live in train item files.
- 4,661 live lines duplicated on (account, date, amount, text) — legit repeats (fees etc.), bank_txn_id is unique.
- Empty: debtor_iban 53.7k DD rows (ACH/BECS), stub_reference 920, beneficiary_iban 98k payment rows, customer_po ~40%.
- Credit notes reasons: AP `Abono de proveedor`; AR `Devolución / rectificación`, `Anulación de factura`,
  `Rappel / bonificación por volumen` (rappel → related_document empty).
- DD schemes live: ACH 44.8k, SEPA CORE 37.6k, BECS 8.9k, BACS 2.5k rows; re_presentation=Y 1,802.
- Raw N43: CRLF, ASCII; Banca Ribagorza raw files overlap months (normalised table already de-duplicated).

## Sample golden snapshot (2025-06)
8,499 lines → 36,574 rows (median 1 row/line, max 585). Non-item lines: BANK_FEES 536, UNIDENTIFIED 218, CASH_POOL 63,
CUSTOMER_REFUND 20, BANK_REVERSAL 18, LEASING 13, others <10. Exceptions: FX_DIFFERENCE 605, PARTIAL 466,
SHORT_DISCOUNT 247, SHORT_DISPUTE 225, UNIDENTIFIED 218, RETURNED 210, DUPLICATE_PAYMENT 35, OVERPAYMENT 24,
SHORT_BANK_CHARGES 9, SHORT_WITHHOLDING 8. Top segments (rows): ar_short_dispute 8,318, ar_ref_noise 5,525,
ar_dd_batch 5,184, ap_payment_multi 3,015, ar_1toN_credit_note 2,459, ar_1toN_text 2,229, ar_card_settlement 1,150.

## 6. Loader timings
live: load_bank 1.2 s (203,083×25) · load_items 5.4 s (1,031,576×21) · load_batches 1.6 s · 16,688 PDFs 13.6 s.
train: load_bank 9.5 s (1,489,931) · load_items 36.4 s (5,644,153) · load_batches 11.0 s · 33,204 PDFs 26.5 s.
Golden train load (usecols) ~10 s. Peak RAM fine on 16 GB with dtype=str.
