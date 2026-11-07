# Audit report — live

- Bank lines: **203,083**
- Lines allocated to items: **178,883**
- Lines sent to review (confidence < 0.75): **31,450** → `submission_live_v4_review_queue.csv`
- Full trail: `submission_live_v4_audit_trail.csv.gz`

## Decisions by rule

| rule | lines | share |
|---|---|---|
| ar:single_amount | 39,252 | 19.3% |
| ar:refs_sum | 34,095 | 16.8% |
| payment_run:e2e | 31,048 | 15.3% |
| payment_run:e2e16 | 24,325 | 12.0% |
| category_rule | 14,987 | 7.4% |
| unidentified | 8,633 | 4.3% |
| remittance_advice | 6,709 | 3.3% |
| ap:name_amount | 5,955 | 2.9% |
| ar:subset_overdue1 | 5,908 | 2.9% |
| return:dd_mandate | 3,478 | 1.7% |
| ar:a4_partial_ratio | 3,390 | 1.7% |
| ar:a4_single_short_discount | 2,572 | 1.3% |
| ar:a4_single_short_discount_ref | 2,307 | 1.1% |
| ap:refs_sum | 2,206 | 1.1% |
| ar:subset1 | 2,083 | 1.0% |
| cheque_deposit | 1,968 | 1.0% |
| dd_batch | 1,590 | 0.8% |
| ar:ref_instalment | 1,353 | 0.7% |
| rec:partial_ratio | 1,324 | 0.7% |
| card_settlement | 1,265 | 0.6% |
| ar:subset_ranked | 1,003 | 0.5% |
| ar:fifo_prefix | 965 | 0.5% |
| ar:a4_single_short_withholding | 659 | 0.3% |
| duplicate | 623 | 0.3% |
| ar:a4_single_short_withholding_ref | 491 | 0.2% |
| ar:a4_refs | 390 | 0.2% |
| category_fallback | 367 | 0.2% |
| payment_run:run_e2e_amount | 328 | 0.2% |
| ar:subset_overdue_oldest | 313 | 0.2% |
| rec:subset1 | 311 | 0.2% |
| ar:a4_multi_partial | 291 | 0.1% |
| ar:a4_single_partial | 240 | 0.1% |
| ar:fx_orig_refs | 224 | 0.1% |
| payment_run:run_e2e_subset | 218 | 0.1% |
| duplicate_refund | 213 | 0.1% |
| ar:a4_overpayment | 207 | 0.1% |
| ar:a4_single_partial_ref | 196 | 0.1% |
| rec:subset_ranked | 196 | 0.1% |
| ar:a4_overpayment_sig | 175 | 0.1% |
| return:payment_run_docs | 170 | 0.1% |
| ar:fx_orig_single | 161 | 0.1% |
| ar:a4_bank_charges | 131 | 0.1% |
| ar:ref_partial | 121 | 0.1% |
| ar:fx_rate | 99 | 0.0% |
| return:cheque | 72 | 0.0% |
| ar:fx_orig_subset | 69 | 0.0% |
| ar:fx_orig_short | 68 | 0.0% |
| ar:fifo_prefix_due | 59 | 0.0% |
| ar:refs_partial_last | 53 | 0.0% |
| ar:fx_orig_partial | 44 | 0.0% |
| payment_run:run_count | 33 | 0.0% |
| return:card | 31 | 0.0% |
| rec:short | 28 | 0.0% |
| rec:refs_sum | 23 | 0.0% |
| ar:a4_refs_cn | 19 | 0.0% |
| ar:single_amount_ref | 18 | 0.0% |
| return:party_amount | 8 | 0.0% |
| ar:a4_prefix | 6 | 0.0% |
| ar:fx_orig_prefix | 6 | 0.0% |
| rec:fifo_partial | 3 | 0.0% |
| ar:ref_overpayment | 2 | 0.0% |
| ap:subset | 1 | 0.0% |

## Exceptions (rows)

| exception | rows |
|---|---|
| PARTIAL | 17,495 |
| FX_DIFFERENCE | 15,249 |
| SHORT_DISCOUNT | 9,346 |
| UNIDENTIFIED | 9,048 |
| RETURNED | 4,113 |
| SHORT_DISPUTE | 2,532 |
| SHORT_WITHHOLDING | 2,110 |
| DUPLICATE_PAYMENT | 623 |
| OVERPAYMENT | 385 |
| SHORT_BANK_CHARGES | 212 |

## Categories (non-item lines)

| category | lines |
|---|---|
| BANK_FEES | 10,018 |
| UNIDENTIFIED | 9,048 |
| CASH_POOL | 1,568 |
| CUSTOMER_REFUND | 751 |
| PAYROLL | 487 |
| INTERCOMPANY | 375 |
| LEASING | 354 |
| BANK_REVERSAL | 332 |
| INTERNAL_TRANSFER | 238 |
| CUSTOMER_ADVANCE | 223 |
| FX_DEAL | 212 |
| TAX_PAYMENT | 124 |
| LOAN_REPAYMENT | 107 |
| SOCIAL_SECURITY | 102 |
| CARD_EXPENSES | 97 |
| SUPPLIER_REFUND | 67 |
| INTEREST_INCOME | 35 |
| INSURANCE | 32 |
| INTEREST_EXPENSE | 28 |
| LOAN_DRAWDOWN | 2 |

## Anomaly flags

- DUPLICATE_PAYMENT rows: 623 (refunds expected later as CUSTOMER/SUPPLIER_REFUND)
- RETURNED rows: 4,113
- UNIDENTIFIED receipts: 9,048 (no open item with name/IBAN/reference evidence — never forced)
- BANK_REVERSAL lines: 332

## Audit flags (not scored)

| flag | severity | n |
|---|---|---|
| UNIDENTIFIED_AGED | high | 4,563 |
| UNIDENTIFIED_AGED | medium | 2,709 |
| RETURN_REASON | low | 1,957 |
| RETURN_REASON | medium | 1,802 |
| AMOUNT_OUTLIER | low | 1,413 |
| DUPLICATE_REFUND | low | 818 |
| DUPLICATE_PAYMENT | high | 623 |
| UNIDENTIFIED_CASH | medium | 254 |
| BANK_ERROR_PAIR | medium | 166 |
| IBAN_CHANGE_NOTICE | low | 79 |
| IBAN_UNSEEN | low | 9 |

Detail: `submission_live_v4_audit_flags.csv`
