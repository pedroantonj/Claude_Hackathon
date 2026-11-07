"""Category classifier for bank lines that settle no item (POLICIES §4).

classify(bank) -> DataFrame[category, conf, final]
  final=True  -> the line is a treasury / non-item line: never sent to the matcher.
  final=False -> fallback category used only if no matcher allocates the line.
Rules = regexes on the upper-cased raw bank `text` + sign. Validated by agent 4 on TRAIN golden
(1.49M lines): precision 1.000 for every rule, recall 1.000 for all treasury categories
(see notes/agent4_accounting.md).
"""
import re

import numpy as np
import pandas as pd

# (category, regex on upper(text), sign '+', '-' or None) -- first match wins. All are final.
RULES = [
    ("BANK_REVERSAL", r"ANULACION OPERACION INDEBIDA", None),
    ("CASH_POOL", r"CASH CONCENTRATION|ZBA SWEEP|CASH POOLING|TRASPASO ENTRE CUENTAS|TRASPASO PROPIO", None),
    ("INTERNAL_TRANSFER", r"TRASPASO (?:A|DE) CTA", None),
    ("INTERCOMPANY", r"VALTERM|INTRAGRUP|DEVOLUCION PRESTAM|TRANSF\. IC|FINANCIACION IC|PRESTAMO IC", None),
    ("FX_DEAL", r"FX DEAL|COMPRA DIVISA|VENTA DIVISA", None),
    ("PAYROLL", r"NOMINA|PAYROLL", "-"),
    ("SOCIAL_SECURITY", r"SOCIAL CHARGES|TGSS|CUOTAS DE LA SEGURIDAD SOCIAL", "-"),
    ("TAX_REFUND", r"DEVOLUCION AEAT", "+"),
    ("TAX_PAYMENT", r"IMPUESTOS MOD|TAX PAYMENT|VAT RETURN|TAX AUTHORITY|DGFIP", "-"),
    ("LOAN_DRAWDOWN", r"DISPOSICION POLIZA", "+"),
    ("LOAN_REPAYMENT", r"AMORTIZACION (?:DE PRESTAMO|POLIZA)|^PRES\.[A-Z]{2}\d", "-"),
    ("LEASING", r"CUOTA (?:LEASING|ARRENDAMIENTO|LSG|RENTING)", "-"),
    ("INSURANCE", r"INSURANCE PREMIUM|PRIMA SEGURO|SEGUROS MULTIRRIESGO", "-"),
    ("CARD_EXPENSES", r"ADEUDO MENSUAL DE TARJETA", "-"),
    ("INTEREST_INCOME", r"^LIQUIDACION INTERESES \dT\d\d", "+"),
    ("INTEREST_EXPENSE", r"^LIQUIDACION INTERESES \dT\d\d", "-"),
    ("BANK_FEES", r"^COMISION (?:TRANSF|DEVOLUCION|REMESA|MANTENIMIENTO)|^COMISIONES SERVICIOS|SERV\. EM\. TRANSF|"
                  r"^FRAIS VIREMENT|^COM\. TRANSF|MAINTENANCE FEE|^LIQUIDACION CTA\.CTO|"
                  r"^LIQUIDACION DE INTERESES, COMISIONES", "-"),
    # party-dependent keyword rules (precision 1.000 on TRAIN over ALL lines, incl. item lines)
    ("CUSTOMER_ADVANCE", r"PROFORMA|\bPF\d\d-", "+"),
    ("UNIDENTIFIED", r"INGRESO EFECTIVO|INGRESO CAJERO|VERSEMENT ESPECES|VERSAMENTO CONTANTI|DEPOSITO EN EFECTIVO|"
                     r"ANTICIPO", "+"),
    ("SUPPLIER_REFUND", r"REEMBOLSO(?! FRA)|ABONO PAGO DOBLE|DEVOLUCION PAGO|PAGO DUPLICADO|DUPLICATE", "+"),
    ("CUSTOMER_REFUND", r"PAGO DUPLICADO|DUPLICATE|DEVOLUCION PAGO", "-"),
]
_COMPILED = [(c, re.compile(p), s) for c, p, s in RULES]


def _reversal_partners(bank, txt, cat):
    """Original line of a bank error: same account, amount = -reversal, booked 0..10 days before,
    text == tail of the reversal text (after 'N <digits>')."""
    rev = bank.index[cat == "BANK_REVERSAL"]
    if len(rev) == 0:
        return []
    amt = bank["amount_f"].round(2)
    bd = pd.to_datetime(bank["booking_date"])
    key = lambda s: re.sub(r"[ |]", "", s)[:12]
    look = {}
    for i in bank.index[cat == ""]:
        look.setdefault((bank.at[i, "account_id"], amt.at[i]), []).append(i)
    out = []
    for r in rev:
        tail = re.sub(r"^.*ANULACION OPERACION INDEBIDA N [\d |]+", "", txt.at[r]).strip()
        for i in look.get((bank.at[r, "account_id"], round(-amt.at[r], 2)), []):
            dd = (bd.at[r] - bd.at[i]).days
            if 0 <= dd <= 10 and key(txt.at[i].strip()) == key(tail):
                out.append(i)
    return out


def classify(bank):
    """bank must contain `text` (or `search`), `amount_f`, `account_id`, `booking_date`."""
    src = bank["text"] if "text" in bank else bank["search"]
    txt = src.fillna("").astype(str).str.upper()
    credit = bank["amount_f"] > 0
    sign = np.where(credit, "+", "-")
    cat = pd.Series("", index=bank.index, dtype=object)
    for c, rx, s in _COMPILED:
        m = (cat == "") & txt.str.contains(rx, regex=True)
        if s:
            m &= (sign == s)
        cat[m] = c
    if {"account_id", "booking_date"} <= set(bank.columns):
        cat[_reversal_partners(bank, txt, cat)] = "BANK_REVERSAL"
    final = cat != ""
    # unmatched fallback: receipts -> UNIDENTIFIED; debits -> CUSTOMER_REFUND (outgoing transfer to a
    # customer with no AP item; on TRAIN all residual non-item debits are CUSTOMER_REFUND)
    fallback = np.where(credit, "UNIDENTIFIED", "CUSTOMER_REFUND")
    return pd.DataFrame({"category": cat.where(final, fallback), "final": final,
                         "conf": np.where(final, 0.95, 0.3)}, index=bank.index)


def postprocess(S, bank):
    """Hook run after the matching pass. Reversal pairing is done up-front in classify().
    Enforces POLICIES: category UNIDENTIFIED always carries exception UNIDENTIFIED.
    Enforces POLICIES §3 FX_DIFFERENCE: an invoice row whose currency differs from the bank-line currency and
    is fully cleared ('' / SHORT_* / OVERPAYMENT) is FX_DIFFERENCE; credit-note rows carry no exception.
    (TRAIN golden: 113,508 FX rows, 0 with same currency; all mismatched CN rows are ''.)"""
    cur = dict(zip(bank["bank_txn_id"], bank["currency"]))
    _partial_from_history(S)
    I = S.I
    full = ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES", "OVERPAYMENT")
    for bid, rows in S.out.items():
        if len(rows) == 1 and rows[0][2] == "UNIDENTIFIED" and rows[0][3] != "UNIDENTIFIED":
            S.out[bid] = [("", "", "UNIDENTIFIED", "UNIDENTIFIED")]
            continue
        c = cur.get(bid)
        if not c or not rows or not rows[0][0]:
            continue
        new, ch = [], False
        for it, amt, cat, exc in rows:
            v = I.get(it)
            if v is not None and v["cur"] != c and exc != "RETURNED":
                if v["k"] in ("ARI", "API") and exc in full:
                    exc, ch = "FX_DIFFERENCE", True
                elif v["k"] in ("ARC", "APC") and exc in ("FX_DIFFERENCE",):
                    exc, ch = "", True
            new.append((it, amt, cat, exc))
        if ch:
            S.out[bid] = new
    return None


def _partial_from_history(S):
    """POLICIES §3 PARTIAL: an invoice row with no exception whose amount is below the item's open balance
    before this line leaves an open balance -> PARTIAL (e.g. payment-run documents paid 30/50/70 %).
    Open balance is replayed from our own allocation log (S.alloc_hist, chronological)."""
    I = S.I
    fix = {}
    for it, hist in S.alloc_hist.items():
        v = I.get(it)
        if v is None or v["k"] not in ("ARI", "API"):
            continue
        o = v.get("open0", v["tot"])
        for bid, _d, c, exc in hist:
            if exc == "RETURNED":
                o += c
                continue
            if exc == "DUPLICATE_PAYMENT":
                continue
            if exc == "" and 0 < c < o - 1:
                fix.setdefault(bid, set()).add(it)
                o -= c
            elif exc in ("", "PARTIAL", "SHORT_DISPUTE"):
                o = max(0, o - c)
            else:
                o = 0
    for bid, its in fix.items():
        rows = S.out.get(bid)
        if rows:
            S.out[bid] = [(it, a, cat, "PARTIAL" if it in its and exc == "" else exc) for it, a, cat, exc in rows]
