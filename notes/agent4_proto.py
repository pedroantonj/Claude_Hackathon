"""Agent 4 prototype: vectorized no-item category classifier + exception helpers.
Usage: python3 -I agent4_proto.py <bank_transactions.csv.gz> [golden.csv.gz]
Only pandas/numpy.
"""
import sys, re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- category rules
# (category, regex on upper(text), sign constraint: '+', '-', or None)
# order matters: first match wins
RULES = [
    ("BANK_REVERSAL",     r"ANULACION OPERACION INDEBIDA", None),
    ("CASH_POOL",         r"CASH CONCENTRATION|ZBA SWEEP|CASH POOLING|TRASPASO ENTRE CUENTAS|TRASPASO PROPIO", None),
    ("INTERNAL_TRANSFER", r"TRASPASO (?:A|DE) CTA", None),
    ("INTERCOMPANY",      r"VALTERM|INTRAGRUP|DEVOLUCION PRESTAM|TRANSF\. IC|FINANCIACION IC|PRESTAMO IC", None),
    ("FX_DEAL",           r"FX DEAL|COMPRA DIVISA|VENTA DIVISA", None),
    ("PAYROLL",           r"NOMINA|PAYROLL", "-"),
    ("SOCIAL_SECURITY",   r"SOCIAL CHARGES|TGSS|CUOTAS DE LA SEGURIDAD SOCIAL", "-"),
    ("TAX_REFUND",        r"DEVOLUCION AEAT", "+"),
    ("TAX_PAYMENT",       r"IMPUESTOS MOD|TAX PAYMENT|VAT RETURN|TAX AUTHORITY|DGFIP", "-"),
    ("LOAN_DRAWDOWN",     r"DISPOSICION POLIZA", "+"),
    ("LOAN_REPAYMENT",    r"AMORTIZACION (?:DE PRESTAMO|POLIZA)|^PRES\.[A-Z]{2}\d", "-"),
    ("LEASING",           r"CUOTA (?:LEASING|ARRENDAMIENTO|LSG|RENTING)", "-"),
    ("INSURANCE",         r"INSURANCE PREMIUM|PRIMA SEGURO|SEGUROS MULTIRRIESGO", "-"),
    ("CARD_EXPENSES",     r"ADEUDO MENSUAL DE TARJETA", "-"),
    ("INTEREST_INCOME",   r"^LIQUIDACION INTERESES \dT\d\d", "+"),
    ("INTEREST_EXPENSE",  r"^LIQUIDACION INTERESES \dT\d\d", "-"),
    ("BANK_FEES",         r"^COMISION (?:TRANSF|DEVOLUCION|REMESA|MANTENIMIENTO)|^COMISIONES SERVICIOS|SERV\. EM\. TRANSF|^FRAIS VIREMENT|^COM\. TRANSF|MAINTENANCE FEE|^LIQUIDACION CTA\.CTO|^LIQUIDACION DE INTERESES, COMISIONES", "-"),
]
# soft (party-dependent) rules: only apply if the line was NOT matched to an item
SOFT = [
    ("CUSTOMER_ADVANCE",  r"PROFORMA|\bPF\d\d-", "+"),
    ("UNIDENTIFIED",      r"INGRESO EFECTIVO|INGRESO CAJERO|VERSEMENT ESPECES|VERSAMENTO CONTANTI|DEPOSITO EN EFECTIVO|ANTICIPO", "+"),
    ("SUPPLIER_REFUND",   r"REEMBOLSO(?! FRA)|ABONO PAGO DOBLE|DEVOLUCION PAGO|PAGO DUPLICADO|DUPLICATE", "+"),
    ("CUSTOMER_REFUND",   r"PAGO DUPLICADO|DUPLICATE|DEVOLUCION PAGO", "-"),
]


def norm_text(b):
    return b["text"].fillna("").astype(str).str.upper()


def classify_hard(b):
    """returns Series category (NaN when no hard rule fires)."""
    t = norm_text(b)
    sgn = np.where(b["amount"] > 0, "+", "-")
    out = pd.Series(np.nan, index=b.index, dtype=object)
    for cat, rx, s in RULES:
        hit = t.str.contains(rx, regex=True) & out.isna()
        if s:
            hit &= (sgn == s)
        out[hit] = cat
    # bank reversal partner: original line = same account, opposite amount, text == tail of the reversal text,
    # booked 0..10 days before the reversal
    rev = b[out == "BANK_REVERSAL"].copy()
    if len(rev):
        rev["tail"] = t[rev.index].str.replace(r"^.*ANULACION OPERACION INDEBIDA N [\d |]+", "", regex=True).str.strip()
        rev["key_amt"] = (-rev["amount"]).round(2)
        cand = b[out.isna()].copy()
        cand["key_amt"] = cand["amount"].round(2)
        cand["ctext"] = t[cand.index].str.strip()
        j = rev[["account_id", "key_amt", "booking_date", "tail"]].merge(
            cand[["account_id", "key_amt", "booking_date", "ctext"]].reset_index(),
            on=["account_id", "key_amt"], suffixes=("_r", "_o"))
        d = (pd.to_datetime(j.booking_date_r) - pd.to_datetime(j.booking_date_o)).dt.days
        j = j[(d >= 0) & (d <= 10)]
        # text check is lenient (N43 splits words with ' | ')
        ok = [a.replace(" | ", "").replace(" ", "")[:12] == c.replace(" | ", "").replace(" ", "")[:12]
              for a, c in zip(j["tail"], j["ctext"])]
        j = j[ok]
        out[j["index"].unique()] = "BANK_REVERSAL"
    return out


def classify_soft(b, hard):
    t = norm_text(b)
    sgn = np.where(b["amount"] > 0, "+", "-")
    out = pd.Series(np.nan, index=b.index, dtype=object)
    for cat, rx, s in SOFT:
        hit = t.str.contains(rx, regex=True) & out.isna() & hard.isna() & (sgn == s)
        out[hit] = cat
    return out


def eval_cats(b, g):
    lab = g.groupby("bank_txn_id").agg(cat=("category", "first"), seg=("segment", "first")).reset_index()
    m = b.merge(lab, on="bank_txn_id", how="left")
    m["cat"] = m["cat"].fillna("ITEM")
    hard = classify_hard(m)
    soft = classify_soft(m, hard)
    m["pred_hard"] = hard
    m["pred_soft"] = soft
    rows = []
    for c in sorted(set(m.cat) - {"ITEM"}):
        gold = m.cat == c
        ph = m.pred_hard == c
        ps = m.pred_soft == c
        p = ph | ps
        rows.append(dict(category=c, n=int(gold.sum()), pred=int(p.sum()), tp=int((p & gold).sum()),
                         prec=round((p & gold).sum() / max(p.sum(), 1), 4), rec=round((p & gold).sum() / max(gold.sum(), 1), 4)))
    res = pd.DataFrame(rows)
    return m, res


if __name__ == "__main__":
    b = pd.read_csv(sys.argv[1], low_memory=False)
    if len(sys.argv) > 2:
        g = pd.read_csv(sys.argv[2])
        m, res = eval_cats(b, g)
        print(res.to_string(index=False))
        nonitem = m.cat != "ITEM"
        pred = m.pred_hard.fillna(m.pred_soft)
        print("hard-rule lines:", m.pred_hard.notna().sum(), "acc on them:", (m.pred_hard == m.cat)[m.pred_hard.notna()].mean())
        print("non-item lines covered (any rule, correct):", ((pred == m.cat) & nonitem).sum(), "/", nonitem.sum())


# ---------------------------------------------------------------- exception decision procedure
FULL_RATIOS = None
PARTIAL_RATIOS = np.array([0.2, 0.25, 0.3, 1/3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9, 0.95, 0.98, 0.99, 1/6])
WHT_RATES = {"MX": [0.0125, 0.10], "BR": [0.015, 0.10], "MA": [0.015, 0.10]}


def hu(x):
    """round half up to cents (golden uses this for discount amounts)"""
    return np.floor(np.asarray(x, dtype=float) * 100 + 0.5 + 1e-9) / 100


def subset_sum(vals, target, tol, max_n=60):
    """smallest-cardinality-first DP over cents; returns list of indices or None"""
    if not vals or len(vals) > max_n or target <= 0:
        return None
    t = int(round(target * 100)); tl = max(1, int(round(tol * 100)))
    reach = {0: ()}
    for i, v in enumerate(vals):
        c = int(round(v * 100))
        for sm, idx in list(reach.items()):
            ns = sm + c
            if ns <= t + tl and ns not in reach:
                reach[ns] = idx + (i,)
    for k in range(t - tl, t + tl + 1):
        if k in reach and reach[k]:
            return list(reach[k])
    return None


def decide(rows, cash, bcur, booking_date, orig_amount=None, orig_currency=None,
           fx_rate=None, eur_rate=1.0, remit_reasons=None):
    """rows: list of dicts with keys item_id, side(ARI/API/ARC/APC), open, total, icur,
    disc_pct (or None), disc_days, issue_date (Timestamp), wht_country (MX/BR/MA or None), wht_rate (known or None).
    cash: abs(bank amount) in bank currency. booking_date: Timestamp.
    fx_rate(icur, bcur, date) -> units of bcur per 1 icur (reference) ; eur_rate: units of icur per EUR.
    remit_reasons: {item_id: paid_amount} for items with a reasoned deduction in the remittance advice.
    returns list of (item_id, amount, exception)."""
    inv = [r for r in rows if r["side"] in ("ARI", "API")]
    cns = [r for r in rows if r["side"] in ("ARC", "APC")]
    icur = inv[0]["icur"] if inv else rows[0]["icur"]
    fx = icur != bcur
    if fx:
        if orig_amount is not None and orig_currency == icur and not pd.isna(orig_amount):
            C, tol_rel, tol_abs = abs(orig_amount), 0.0, 0.011
        else:
            C, tol_rel, tol_abs = cash / fx_rate(icur, bcur, booking_date), 0.03, 0.0
    else:
        C, tol_rel, tol_abs = cash, 0.0, 0.011
    out = {r["item_id"]: [r["open"], None] for r in rows}
    N = sum(r["open"] for r in inv) - sum(r["open"] for r in cns)
    tol = max(tol_abs, tol_rel * abs(N))
    d = C - N

    def fin():
        res = []
        for r in rows:
            a, e = out[r["item_id"]]
            if fx and r["side"] in ("ARI", "API") and e in (None, "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES", "OVERPAYMENT"):
                e = "FX_DIFFERENCE"
            res.append((r["item_id"], round(float(a), 2), e))
        return res

    # remittance-advice reasoned deductions -> SHORT_DISPUTE, amount = paid
    if remit_reasons:
        for iid, paid in remit_reasons.items():
            if iid in out:
                d += out[iid][0] - paid
                out[iid] = [paid, "SHORT_DISPUTE"]
        if abs(d) <= tol:
            return fin()
    if abs(d) <= tol:
        return fin()
    if d > tol:
        if len(inv) == 1:
            out[inv[0]["item_id"]][1] = "OVERPAYMENT"
        return fin()
    # ---- shortfall
    # (a) early-payment discount, per invoice (open == total only)
    dis_in, dis_late = [], []
    for r in inv:
        if r.get("disc_pct") and abs(r["open"] - r["total"]) < 0.005:
            disc = float(hu(r["disc_pct"] / 100 * r["total"]))
            late = (booking_date - r["issue_date"]).days > r["disc_days"] + 3
            (dis_late if late else dis_in).append((r, disc))
    cands = [dis_in + dis_late, dis_in, dis_late]
    sub = subset_sum([x[1] for x in dis_in + dis_late], -d, tol)
    if sub is not None:
        cands.append([(dis_in + dis_late)[k] for k in sub])
    for cand in cands:
        D = sum(x[1] for x in cand)
        if cand and abs(d + D) <= tol:
            for r, disc in cand:
                if (r, disc) in dis_late:
                    out[r["item_id"]] = [r["total"] - disc, "PARTIAL"]
                else:
                    out[r["item_id"]][1] = "SHORT_DISCOUNT"
            return fin()
    # (b) withholding
    wc = {r.get("wht_country") for r in inv} - {None}
    rates = []
    for r in inv:
        if r.get("wht_rate"):
            rates.append(r["wht_rate"])
    for c in wc:
        rates += WHT_RATES.get(c, [])
    for rate in dict.fromkeys(rates):
        W = sum(float(hu(rate * r["total"])) for r in inv)
        if abs(d + W) <= tol:
            for r in inv:
                out[r["item_id"]][1] = "SHORT_WITHHOLDING"
            return fin()
    # (c) bank charges: single invoice, <= 60 EUR eq, <= ~3.3 % (golden goes to 3.15 %, partials start at 3.5 %)
    if len(inv) == 1 and -d / eur_rate <= 60 and -d <= 0.033 * inv[0]["open"]:
        out[inv[0]["item_id"]][1] = "SHORT_BANK_CHARGES"
        return fin()
    # (d) partial on one invoice: prefer the one whose paid ratio is a 'round' fraction
    best = None
    for r in sorted(inv, key=lambda r: r["issue_date"], reverse=True):
        paid = r["open"] + d
        if paid <= 0:
            continue
        ratio_o = paid / r["open"]
        ratio_t = paid / r["total"]
        ok = np.isclose(ratio_o, PARTIAL_RATIOS, atol=6e-4).any() or np.isclose(ratio_t, PARTIAL_RATIOS, atol=6e-4).any()
        if ok:
            best = (r, paid)
            break
        if best is None:
            best = (r, paid)
    if best is not None:
        out[best[0]["item_id"]] = [best[1], "PARTIAL"]
    else:  # shortfall bigger than any single invoice: newest invoices partial in turn
        rem = C + sum(r["open"] for r in cns)
        for r in sorted(inv, key=lambda r: r["issue_date"]):
            pay = min(r["open"], max(rem, 0))
            rem -= pay
            out[r["item_id"]] = [pay, None if abs(pay - r["open"]) < 0.005 else "PARTIAL"]
    return fin()
