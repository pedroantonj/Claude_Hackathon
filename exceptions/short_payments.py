"""Explained differences between cash and open items (POLICIES §3) -- agent 4.

explain_short(t, T, A, pool, refs, S) -> (rows, rule) | None
  t: bank line, T: upper-cased search text, A: |amount| in cents (line currency),
  pool: candidate open AR items (any currency) of the identified parties, refs: referenced items.

Verified on TRAIN golden (notes/agent4_accounting.md):
  SHORT_DISCOUNT   cash = T - round_half_up(pct*T) per invoice; booking <= issue + disc_days + 3; else PARTIAL
                   (amount = cash part).  Multi-invoice: discount taken on a SUBSET of invoices.
  SHORT_WITHHOLDING rate*T per invoice, rate fixed per customer (MX 1.25/10 %, BR & MA 1.5/10 %); only known
                   withholders (learned from golden history) may use 10 % (collides with 10 % retentions).
  SHORT_BANK_CHARGES single invoice, shortfall <= 60 EUR-eq and <= 3.3 % (golden goes to 3.15 %).
  OVERPAYMENT      single invoice, cash > open.
  PARTIAL          paid/open or paid/total is a round fraction (instalments 1/n, retention .9/.95, a cuenta ...).
  FX_DIFFERENCE    invoice currency != account currency; orig_amount (item cur) when given, else ECB +-3 %.
"""
import os
import pickle

from matching.text import cents, fmt, mitm

WHT_BY_COUNTRY = {"MX": (0.0125, 0.10), "BR": (0.015, 0.10), "MA": (0.015, 0.10)}
WHT_BY_ENTITY = {"ES01": (0.10,)}   # EDA C-016: ES01 {10 %}
FX_TOL = 0.03
RATIOS = (0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1 / 6)
MAXPREFIX = 30
SEPA = {"ES", "PT", "FR", "IT", "DE", "NL", "GB", "PL", "CH", "BE", "AT", "IE", "LU", "FI", "SE", "DK", "NO"}
import re as _re
NONSEPA_RX = _re.compile(r"WIRE IN|GASTOS SHA|GASTOS BEN|EN DIV|\bSHA\b|TT IN|SWIFT|DIRECT CREDIT")
REFUND_RX = _re.compile(r"CREDIT NOTE|SALDO A FAVOR|\bABONO\b|REEMBOLSO|\bAB-|REFUND")


# ----------------------------------------------------------------------------- helpers
def hu(x):
    """round half up (cents in, cents out)"""
    return int(x + 0.5 + 1e-9)


def _disc(S, v):
    code = v["terms"]
    if not code or code not in S.terms.index:
        return None
    pct = S.terms.at[code, "early_payment_discount_pct"]
    days = S.terms.at[code, "discount_days"]
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        return None
    if not pct:
        return None
    return pct, int(float(days or 0))


def _plus_days(d, n):
    import datetime as dt
    return (dt.date.fromisoformat(d) + dt.timedelta(days=n)).isoformat()


def _wht_customers(S):
    """customer -> True for customers with SHORT_WITHHOLDING in golden history (no leakage)."""
    if hasattr(S, "_a4_wht"):
        return S._a4_wht
    S._a4_wht = set()
    g = S.ctx.get("golden_hist")
    if g is None or len(g) == 0:
        return S._a4_wht
    ids = set(g.loc[g["exception"] == "SHORT_WITHHOLDING", "item_id"])
    key = f"{S.phase}_{S.start}_{len(ids)}"
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "a4_wht.pkl")
    try:
        with open(cache, "rb") as f:
            k, val = pickle.load(f)
        if k == key:
            S._a4_wht = val
            return val
    except Exception:
        pass
    import pandas as pd
    from data_loader.loader import DATA
    ar = pd.read_csv(os.path.join(DATA, "train", "ar_invoices.csv.gz"), usecols=["item_id", "customer_id"])
    S._a4_wht = set(ar.loc[ar["item_id"].isin(ids), "customer_id"])
    try:
        with open(cache, "wb") as f:
            pickle.dump((key, S._a4_wht), f)
    except Exception:
        pass
    return S._a4_wht


def _wht_rates(S, t, items):
    I = S.I
    rates = []
    known = _wht_customers(S)
    for i in items:
        p = I[i]["p"]
        c = S.cust_country.get(p, "")
        if c not in WHT_BY_COUNTRY:
            c = S.entities.at[t.entity, "country"] if t.entity in S.entities.index else ""
        rs = WHT_BY_COUNTRY.get(c) or WHT_BY_ENTITY.get(t.entity)
        if not rs:
            continue
        if p in known:
            rates += list(rs)
        else:
            rates += [r for r in rs if r < 0.05]
    return list(dict.fromkeys(rates))


def _explain_set(t, S, items, C, tol, eur_rate_cur, allow_single_only=True):
    """Given a candidate item set (same currency) and cash C (cents, item currency), try the
    full-clear explanations. Returns rows or None. Exceptions before FX override."""
    I = S.I
    sg = S.sgn
    inv = [i for i in items if I[i]["k"] in ("ARI", "API")]
    if not inv:
        return None
    N = sum(sg(I[i]["k"]) * I[i]["open"] for i in items)
    d = C - N
    base = {i: [I[i]["open"], ""] for i in items}

    def rows():
        return [(i, fmt(base[i][0]), "", base[i][1]) for i in items]

    if abs(d) <= tol:
        return rows()
    if d > 0:
        return None
    # (a) discount: per invoice, subsets
    elig = []
    for i in inv:
        v = I[i]
        ds = _disc(S, v)
        if ds and v["open"] == v["tot"]:
            pct, days = ds
            late = t.booking_date > _plus_days(v["iss"], days + 3)
            elig.append((i, hu(pct * v["tot"] / 100), late))
    if elig:
        D = [e[1] for e in elig]
        sols = []
        if abs(d + sum(D)) <= tol:
            sols = [list(range(len(elig)))]
        else:
            for dt in range(-tol, tol + 1):
                cnt, sol = mitm(D, -d + dt, cap=1, maxn=22)
                if cnt:
                    sols = [sol]
                    break
        if sols:
            for k in sols[0]:
                i, disc, late = elig[k]
                if late:
                    base[i] = [I[i]["tot"] - disc, "PARTIAL"]
                else:
                    base[i][1] = "SHORT_DISCOUNT"
            return rows()
    # (b) withholding
    for r in _wht_rates(S, t, inv):
        W = sum(hu(r * I[i]["tot"]) for i in inv)
        if abs(d + W) <= max(tol, len(inv)):
            for i in inv:
                base[i][1] = "SHORT_WITHHOLDING"
            return rows()
    # (c) bank charges: single invoice, <= 60 EUR eq, <= 3.3 %
    if len(inv) == 1:
        short = -d
        eur = short / eur_rate_cur if eur_rate_cur else short
        if eur <= 6000 and short <= 0.033 * I[inv[0]]["open"]:
            base[inv[0]][1] = "SHORT_BANK_CHARGES"
            return rows()
    return None


OVER_ABS = (100, 200, 500, 1000, 5000, 10000)
DISPUTE_R = (0.03, 0.05, 0.10, 0.20, 0.30)
DISPUTE_SEARCH = False


def _over_sig(ex, o):
    return ex in OVER_ABS or abs(ex - 0.01 * o) <= 1 or abs(ex - 0.10 * o) <= 1


def _dispute_search(t, S, items, A):
    """cash = sum(open) - sum_k r_k*tot_k over 1..4 invoices, r in {3,5,10,20,30 %} (TRAIN golden: exact set).
    Known withholders: withholding applied to every invoice first. Returns rows or None (unique solution only)."""
    I = S.I
    sg = S.sgn
    inv = [i for i in items if I[i]["k"] in ("ARI", "API") and I[i]["open"] == I[i]["tot"]]
    if not inv or len(inv) > 400:
        return None
    N = sum(sg(I[i]["k"]) * I[i]["open"] for i in items)
    for wr in [0.0] + _wht_rates(S, t, inv):
        W = sum(hu(wr * I[i]["tot"]) for i in inv) if wr else 0
        need = N - W - A
        if need <= 0:
            continue
        tol = 1 + len(inv) // 50
        vals = [(hu(r * I[i]["tot"]), k, r) for k, i in enumerate(inv) for r in DISPUTE_R]
        vals = [v for v in vals if v[0] <= need + tol]
        sol = None
        # 1 or 2 deductions, unique solution only (3-4 deductions give spurious fits on wrong item sets)
        sols = []
        for dt in range(-tol, tol + 1):
            sols += [[v] for v in vals if v[0] == need + dt]
        if not sols:
            by = {}
            for v in vals:
                by.setdefault(v[0], []).append(v)
            for a_ in range(len(vals)):
                for dt in range(-tol, tol + 1):
                    for b_ in by.get(need + dt - vals[a_][0], []):
                        if b_[1] > vals[a_][1]:
                            sols.append([vals[a_], b_])
                if len(sols) > 1:
                    break
        if len(sols) == 1:
            sol = sols[0]
        if sol:
            ded = {inv[k]: v for v, k, r in sol}
            rows = []
            for i in items:
                if i in ded:
                    rows.append((i, fmt(I[i]["open"] - ded[i] - (hu(wr * I[i]["tot"]) if wr else 0)), "", "SHORT_DISPUTE"))
                elif wr and I[i]["k"] in ("ARI", "API"):
                    rows.append((i, fmt(I[i]["open"]), "", "SHORT_WITHHOLDING"))
                else:
                    rows.append((i, fmt(I[i]["open"]), "", ""))
            return rows
    return None


def _round_ratio(paid, v):
    for x in (v["open"], v["tot"]):
        if x <= 0:
            continue
        for r in RATIOS:
            if abs(paid - r * x) <= 1.01:
                return True
    return False


# ----------------------------------------------------------------------------- FX
def explain_fx(t, T, A, pool, refs, S):
    I = S.I
    foreign = [i for i in pool if I[i]["cur"] != t.currency]
    if not foreign:
        return None
    sg = S.sgn

    def fxrows(items, base=None):
        out = []
        for i in items:
            amt, exc = (base or {}).get(i, (I[i]["open"], ""))
            if I[i]["k"] in ("ARI", "API") and exc in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING",
                                                       "SHORT_BANK_CHARGES", "OVERPAYMENT"):
                exc = "FX_DIFFERENCE"
            if I[i]["k"] in ("ARC", "APC"):
                exc = ""
            out.append((i, fmt(amt), "", exc))
        return out

    if t.orig_amount and t.orig_currency:
        target = abs(cents(t.orig_amount))
        cand = [i for i in foreign if I[i]["cur"] == t.orig_currency]
        cand.sort(key=lambda i: (I[i]["iss"], I[i]["dn"]))
        rcand = [i for i in refs if i in cand]
        # exact: refs, single, FIFO prefix, subset
        if rcand and sum(sg(I[i]["k"]) * I[i]["open"] for i in rcand) == target:
            return fxrows(rcand), "fx_orig_refs"
        one = [i for i in cand if I[i]["k"] == "ARI" and I[i]["open"] == target]
        if one:
            pref = [i for i in one if i in refs]
            return fxrows([(pref or one)[0]]), "fx_orig_single"
        acc = 0
        for k, i in enumerate(cand[:MAXPREFIX]):
            acc += sg(I[i]["k"]) * I[i]["open"]
            if acc == target:
                return fxrows(cand[:k + 1]), "fx_orig_prefix"
        if 1 < len(cand) <= 22:
            cnt, sol = mitm([sg(I[i]["k"]) * I[i]["open"] for i in cand], target)
            if cnt >= 1:
                return fxrows([cand[k] for k in sol]), "fx_orig_subset"
        # withholding / discount hidden inside an FX payment
        eur = S.fx_rate(t.orig_currency, t.booking_date)
        sets = ([rcand] if rcand else []) + [[i] for i in cand if I[i]["k"] == "ARI"] + \
               [cand[:k + 1] for k in range(1, min(len(cand), MAXPREFIX))]
        for items in sets:
            rows = _explain_set(t, S, items, target, 1, eur)
            if rows:
                return fxrows(items, {i: (cents(a), e) for i, a, _, e in rows}), "fx_orig_short"
        # partial on a referenced foreign item
        if len(rcand) == 1 and I[rcand[0]]["k"] == "ARI" and target < I[rcand[0]]["open"]:
            return [(rcand[0], fmt(target), "", "PARTIAL")], "fx_orig_partial"
    # convert at reference rate, accept within FX_TOL
    best = None
    by_cur = {}
    for i in foreign:
        by_cur.setdefault(I[i]["cur"], []).append(i)
    for cur, items in by_cur.items():
        items.sort(key=lambda i: (I[i]["iss"], I[i]["dn"]))
        conv_A = S.to_cur(A, t.currency, cur, t.booking_date)
        if not conv_A:
            continue
        cands = [[i] for i in items if I[i]["k"] == "ARI"]
        rc = [i for i in refs if i in items]
        if rc:
            cands.insert(0, rc)
        cands += [items[:k + 1] for k in range(1, min(len(items), MAXPREFIX))]
        for c in cands:
            N = sum(sg(I[i]["k"]) * I[i]["open"] for i in c)
            if N <= 0:
                continue
            diff = abs(conv_A - N) / N
            pri = 0 if (c is rc or any(i in refs for i in c)) else 1
            if diff <= FX_TOL and (best is None or (pri, diff) < best[0]):
                best = ((pri, diff), c)
    if best:
        return fxrows(best[1]), "fx_rate"
    return None


# ----------------------------------------------------------------------------- main
def explain_short(t, T, A, pool, refs, S):
    I = S.I
    res = explain_fx(t, T, A, pool, refs, S)
    if res:
        return res
    sg = S.sgn
    same = [i for i in pool if I[i]["cur"] == t.currency and I[i]["k"] in ("ARI", "ARC", "API", "APC")]
    same.sort(key=lambda i: (I[i]["iss"], I[i]["dn"]))
    if not same:
        return None
    eur = S.fx_rate(t.currency, t.booking_date)
    rsame = [i for i in refs if i in same]
    inv = [i for i in same if I[i]["k"] in ("ARI", "API")]
    # 1. full-clear explanations: refs set, then single invoices (refs first), then FIFO prefixes
    sets = []
    if rsame:
        sets.append(("refs", rsame))
        cns = [i for i in same if I[i]["k"] in ("ARC", "APC") and i not in rsame]
        if cns:
            sets.append(("refs_cn", rsame + cns))
    singles = [i for i in rsame if I[i]["k"] in ("ARI", "API")] + [i for i in inv if i not in rsame]
    for kind in ("single",):
        hits = []
        for i in singles:
            rows = _explain_set(t, S, [i], A, 1, eur)
            if rows and rows[0][3] in ("SHORT_DISCOUNT", "SHORT_WITHHOLDING", "PARTIAL"):
                if i in rsame:
                    return rows, "a4_single_" + rows[0][3].lower() + "_ref"
                hits.append(rows)
        if len(hits) >= 1:
            return hits[0], "a4_single_" + hits[0][0][3].lower()
    for name, items in sets:
        rows = _explain_set(t, S, items, A, 1, eur)
        if rows:
            return rows, "a4_" + name
    for k in range(1, min(len(same), MAXPREFIX)):
        items = same[:k + 1]
        N = sum(sg(I[i]["k"]) * I[i]["open"] for i in items)
        if N < A:
            continue
        if N - A > 0.11 * N:
            break
        rows = _explain_set(t, S, items, A, 1, eur)
        if rows and any(r[3] for r in rows) and not any(r[3] == "SHORT_BANK_CHARGES" for r in rows):
            return rows, "a4_prefix"
    # 2. bank charges on a single invoice (referenced or unique candidate)
    sbc = []
    for i in singles:
        rows = _explain_set(t, S, [i], A, 1, eur)
        if rows and rows[0][3] == "SHORT_BANK_CHARGES":
            sbc.append((i, rows))
    # POLICIES: cross-border transfer from outside SEPA (SHA/BEN). Without a reference only accept a unique
    # candidate in a small pool (avoid unrelated look-alike invoices).
    # TRAIN golden SBC lines: 85 % 'WIRE IN' / 'TRANSFER. EN DIV ... GASTOS SHA' / AU 'DIRECT CREDIT'; the rest are
    # EUR SEPA credits from payers outside SEPA. Domestic ACH / PIX / SPEI shortfalls are never bank charges.
    def _xborder(i):
        if NONSEPA_RX.search(T):
            return True
        return t.currency == "EUR" and S.cust_country.get(I[i]["p"], "") not in SEPA
    sbc = [x for x in sbc if _xborder(x[0])] if not REFUND_RX.search(T) else []
    if sbc and sbc[0][0] not in rsame and any(A < I[i]["open"] and _round_ratio(A, I[i]) for i in singles):
        sbc = []          # cash is an instalment / retention / a-cuenta share of another invoice
    if sbc and (sbc[0][0] in rsame or len(sbc) == 1):
        return sbc[0][1], "a4_bank_charges"
    # 3. overpayment: referenced invoice, or unique invoice with excess <= 10 %
    over = [i for i in rsame if I[i]["k"] in ("ARI", "API") and I[i]["open"] < A <= 1.2 * I[i]["open"]]   # cap kept: uncapped gave P .447 on 2025-01..04
    if len(over) == 1 and len(rsame) == 1:
        i = over[0]
        return [(i, fmt(I[i]["open"]), "", "OVERPAYMENT")], "a4_overpayment"
    # unreferenced overpayment: TRAIN golden excess is 1/2/5/10/50/100 (61 %) or 1 % / 10 % of the open amount
    sig = [i for i in singles if I[i]["open"] < A and _over_sig(A - I[i]["open"], I[i]["open"])]
    if len(sig) == 1 and not rsame:
        i = sig[0]
        return [(i, fmt(I[i]["open"]), "", "OVERPAYMENT")], "a4_overpayment_sig"
    # disputes without advice: referenced multi-invoice set, 1-4 invoices short by 3/5/10/20/30 %
    if DISPUTE_SEARCH and len(rsame) >= 2:     # off: no TP gain on dev windows, adds false positives
        res = _dispute_search(t, S, rsame, A)
        if res:
            return res, "a4_dispute_search"
    # 4. partial: single invoice whose paid share is a round fraction of open/total
    part = [i for i in singles if A < I[i]["open"] and _round_ratio(A, I[i])]
    if part:
        pref = [i for i in part if i in rsame]
        if pref or (len(part) == 1 and len(singles) <= 5):
            i = (pref or part)[0]
            return [(i, fmt(A), "", "PARTIAL")], "a4_partial_ratio"
    # 5. partial inside a referenced multi-invoice set: one invoice partial with a round ratio
    if len(rsame) > 1:
        N = sum(sg(I[i]["k"]) * I[i]["open"] for i in rsame)
        d = A - N
        if d < 0:
            for i in sorted(rsame, key=lambda i: I[i]["iss"], reverse=True):
                if I[i]["k"] not in ("ARI", "API"):
                    continue
                paid = I[i]["open"] + d
                if paid > 0 and _round_ratio(paid, I[i]):
                    return [(j, fmt(paid if j == i else I[j]["open"]), "", "PARTIAL" if j == i else "")
                            for j in rsame], "a4_multi_partial"
    return None
