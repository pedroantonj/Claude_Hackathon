"""Party + amount matching for AR receipts and AP debits not covered by files.

Cascade (stop at first hit): refs summing to cash -> single exact amount -> FIFO prefix -> subset-sum ->
short-payment explanations (exceptions module) -> single referenced item PARTIAL/OVERPAYMENT.
Never accepts an amount-only match without a name / IBAN / reference (unidentified look-alikes).
"""
import datetime as dt
import itertools
import re

from rapidfuzz.distance import DamerauLevenshtein as DL

from exceptions.short_payments import explain_short
from .text import cents, fmt, mitm, mitm_all, nz

MONTHS = {"ENE": 1, "JAN": 1, "ENERO": 1, "JANUARY": 1, "JANVIER": 1, "FEB": 2, "FEBRERO": 2, "FEBRUARY": 2, "FEV": 2,
          "MAR": 3, "MARZO": 3, "MARCH": 3, "MARS": 3, "ABR": 4, "APR": 4, "ABRIL": 4, "APRIL": 4, "AVR": 4,
          "MAY": 5, "MAYO": 5, "MAI": 5, "JUN": 6, "JUNIO": 6, "JUNE": 6, "JUIN": 6, "JUL": 7, "JULIO": 7, "JULY": 7,
          "JUIL": 7, "AGO": 8, "AUG": 8, "AGOSTO": 8, "AUGUST": 8, "AOUT": 8, "SEP": 9, "SEPT": 9, "SEPTIEMBRE": 9,
          "SEPTEMBER": 9, "OCT": 10, "OCTUBRE": 10, "OCTOBER": 10, "OKT": 10, "NOV": 11, "NOVIEMBRE": 11,
          "NOVEMBER": 11, "DIC": 12, "DEC": 12, "DICIEMBRE": 12, "DECEMBER": 12, "DEZ": 12}


STALE_DAYS = 365
INSTAL_RX = re.compile(r"INSTAL|PLAZO|VENCIMIENTO \d|\b\d/\d\b|A CUENTA|ON ACCOUNT|ACOMPTE|ACCONTO|PARCIAL|PARTIAL|ABSCHLAG|RATA|PRESTA")


def _minus_days(d, n):
    return (dt.date.fromisoformat(d) - dt.timedelta(days=n)).isoformat()


def _days(a, b):
    try:
        return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days
    except ValueError:
        return 9999


def service_month(T, d=None):
    """'MANT. MAYO 25' / 'SERVICE OCTOBER 20' (year truncated) / 'SERVICIOS MARZO' -> (year, month) or None.
    The year is taken from the text only when plausible (within a year of the booking date), else the latest such
    month not after the booking month."""
    m = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s*'?(\d{2,4})?\b", T)
    if not m:
        return None
    mon = MONTHS[m.group(1)]
    by, bm = (int(d[:4]), int(d[5:7])) if d else (None, None)
    y = None
    if m.group(2):
        g = m.group(2)
        y = int(g) if len(g) == 4 else 2000 + int(g[-2:])
        if by is not None and abs(y - by) > 1:
            y = None
    if y is None:
        if by is None:
            return None
        y = by if mon <= bm else by - 1
    return y, mon


def find_refs(T, ent, parties, S, kinds=("ARI", "ARC")):
    I = S.I
    refs, unresolved = [], []
    for tok in sorted(set(re.findall(r"[A-Z0-9\-/]{5,}", T))):
        it = None
        c = [i for i in S.full.get((ent, nz(tok)), []) if I[i]["k"] in kinds]
        if c:
            c2 = [i for i in c if I[i]["open"] > 0] or c
            it = c2[0]
        elif re.fullmatch(r"\d{5,8}", tok) and kinds[0] == "ARI":
            c = [i for i in S.suf.get((ent, tok.rjust(6, "0")[-6:]), [])
                 if I[i]["open"] > 0 and (not parties or I[i]["p"] in parties)]
            it = c[0] if len(c) == 1 else None
        if it is None and kinds[0] == "ARI":
            c = [i for i in S.po.get((ent, nz(tok)), []) if I[i]["open"] > 0]
            it = c[0] if len(c) == 1 else None
        if it and I[it]["open"] > 0:
            refs.append(it)
        elif it is None and parties and kinds[0] == "ARI" and 5 <= sum(ch.isdigit() for ch in tok) <= 11:
            unresolved.append(tok)
    if unresolved:
        refs += fuzzy_refs(unresolved, parties, S, set(refs))
    return list(dict.fromkeys(refs))


def fuzzy_refs(toks, parties, S, taken):
    """typos / transposed digits (RE252008072 -> RE250208072, 'RE 108031' tails): Damerau-Levenshtein <= 1
    against the open AR documents of the candidate parties only; unique hit required."""
    I = S.I
    docs = [i for p in parties for i in S.byparty.get(p, ()) if I[i]["open"] > 0 and I[i]["k"] == "ARI" and i not in taken]
    if not docs or len(docs) > 400:
        return []
    keys = [(i, nz(I[i]["dn"]), re.sub(r"\D", "", I[i]["dn"])) for i in docs]
    out = []
    for tok in sorted(toks):
        k = nz(tok)
        if k.isdigit():
            if len(k) < 6:
                continue
            hits = {i for i, _, dd in keys if len(dd) >= len(k) and DL.distance(dd[-len(k):], k) <= 1}
        else:
            if len(k) < 8:
                continue
            hits = {i for i, kk, _ in keys if abs(len(kk) - len(k)) <= 1 and DL.distance(kk, k) <= 1}
        if len(hits) == 1:
            out.append(hits.pop())
    return out


def ar_parties(t, T, S):
    ent = t.entity
    parties = S.CX.query(ent, T)
    if t.counterparty_account:
        p = S.iban_party.get(nz(t.counterparty_account))
        if p:
            parties.add(p)
        p = S.learned_iban.get(nz(t.counterparty_account)) if hasattr(S, "learned_iban") else None
        if p:
            parties.add(p)
    for p in sorted(parties):
        parties |= S.children.get(p, set())
    return parties


def _combo(same, A, refs, d, S):
    I, sg = S.I, S.sgn
    vals = [sg(I[i]["k"]) * I[i]["open"] for i in same]
    acc = list(itertools.accumulate(vals))
    if A in acc:
        k = acc.index(A)
        return [(i, fmt(I[i]["open"]), "", "") for i in same[:k + 1]], "fifo_prefix"
    bydue = sorted(same, key=lambda i: (I[i]["due"] or I[i]["iss"], I[i]["dn"]))
    acc = list(itertools.accumulate(sg(I[i]["k"]) * I[i]["open"] for i in bydue))
    if A in acc:
        k = acc.index(A)
        return [(i, fmt(I[i]["open"]), "", "") for i in bydue[:k + 1]], "fifo_prefix_due"
    if len(same) < 2:
        return None
    forced = [i for i in refs if i in same]
    rest = [i for i in same if i not in forced]
    target = A - sum(sg(I[i]["k"]) * I[i]["open"] for i in forced)
    # EDA C-028: payments without reference settle OVERDUE invoices (97 %): subset-sum N<=25 over them first;
    # unique -> assign (precision .997), several -> the oldest invoices; none -> fall through to the ranker.
    od = sorted([i for i in rest if I[i]["k"] == "ARI" and I[i]["due"] and I[i]["due"] <= d],
                key=lambda i: (I[i]["iss"], I[i]["dn"]))
    if 1 <= len(od) <= 25:
        sols = mitm_all([I[i]["open"] for i in od], target, cap=64, maxn=25) if target > 0 else []
        if sols:
            best = sols[0] if len(sols) == 1 else min(sols, key=lambda sl: sorted(I[od[k]]["iss"] for k in sl))
            chosen = forced + [od[k] for k in best]
            return [(i, fmt(I[i]["open"]), "", "") for i in chosen], \
                ("subset_overdue1" if len(sols) == 1 else "subset_overdue_oldest")
    W = _weights(rest, d, S)
    if len(rest) > 22:
        # keep the most plausible items for a payment booked on d (learned payment-delay hazard; credit notes kept)
        if W is not None:
            rest = sorted(rest, key=lambda i: (I[i]["k"] != "ARC", -W[i], I[i]["iss"]))[:22]
        else:
            rest = sorted(rest, key=lambda i: abs(_days(I[i]["due"] or I[i]["iss"], d)))[:22]
        rest.sort(key=lambda i: (I[i]["iss"], I[i]["dn"]))
    vals = [sg(I[i]["k"]) * I[i]["open"] for i in rest]
    if W is None:
        cnt, sol = mitm(vals, target)
        if cnt >= 1:
            chosen = forced + [rest[k] for k in sol]
            return [(i, fmt(I[i]["open"]), "", "") for i in chosen], f"subset{min(cnt, 2)}"
        return None
    sols = mitm_all(vals, target, cap=64)
    if not sols:
        return None
    if len(sols) == 1:
        best, rule = sols[0], "subset1"
    else:
        best = max(sols, key=lambda sl: (sum(W[rest[k]] for k in sl), -len(sl)))
        rule = "subset_ranked"
    chosen = forced + [rest[k] for k in best]
    return [(i, fmt(I[i]["open"]), "", "") for i in chosen], rule


def _weights(items, d, S):
    """log-odds that each open item is paid by a line booked on d (DelayModel from golden history)."""
    dm = getattr(S, "delay", None)
    if dm is None:
        return None
    I = S.I
    return {i: (0.0 if I[i]["k"] == "ARC" else dm.weight(I[i], d)) for i in items}


def match_ar(t, T, S):
    I = S.I
    A = abs(cents(t.amount))
    ent, d = t.entity, t.booking_date
    parties = ar_parties(t, T, S)
    refs = find_refs(T, ent, parties, S)
    if not parties and refs:
        parties = {I[refs[0]]["p"]}
        for p in sorted(parties):
            parties |= S.children.get(p, set())
    if not parties:
        return False
    pool = [i for p in sorted(parties) for i in S.byparty[p]
            if I[i]["open"] > 0 and I[i]["iss"] <= d and I[i]["k"] in ("ARI", "ARC")]
    same_all = [i for i in pool if I[i]["cur"] == t.currency]
    same_all.sort(key=lambda i: (I[i]["iss"], I[i]["dn"]))
    # items older than ~1 year are almost never paid by amount-only combinatorics (0.4 % of paid AR items):
    # keep them for exact single-amount / references, drop them from FIFO / subset-sum (they are old unpaid noise)
    cutoff = _minus_days(d, STALE_DAYS)
    same = [i for i in same_all if I[i]["iss"] >= cutoff or i in refs]
    sg = S.sgn
    rows, rule = None, ""
    # a. references summing to cash
    if refs and sum(sg(I[i]["k"]) * I[i]["open"] for i in refs if I[i]["cur"] == t.currency) == A:
        rows = [(i, fmt(I[i]["open"]), "", "") for i in refs]
        rule = "refs_sum"
    # b. single exact amount
    if rows is None:
        c = [i for i in same_all if I[i]["k"] == "ARI" and I[i]["open"] == A]
        if c:
            pref = [i for i in c if i in refs]
            if not pref and len(c) > 1:
                sm = service_month(T, d)
                if sm:
                    y, m = sm
                    ny, nm = (y + (m == 12), m % 12 + 1)
                    tag = f"{ny:04d}-{nm:02d}"
                    pref = [i for i in c if I[i]["iss"][:7] == tag]
            i = (pref or c)[0]
            rows = [(i, fmt(A), "", "")]
            rule = "single_amount" + ("_ref" if i in refs else "")
    # c/d. FIFO prefix (issue / due order) and subset-sum, per candidate party first (same-name customers make the
    #      union pool large and ambiguous), then on the union
    if rows is None and same:
        groups = []
        byp = {}
        for i in same:
            byp.setdefault(I[i]["p"], []).append(i)
        if len(byp) > 1:
            groups = list(byp.values())
        res = [r for r in (_combo(g, A, refs, d, S) for g in groups) if r]
        good = [r for r in res if r[1] not in ("subset2", "subset_ranked")]
        if len(good) == 1:
            rows, rule = good[0]
        elif not good:
            r = _combo(same, A, refs, d, S)
            if r and not (len(res) > 1 and r[1] in ("subset2", "subset_ranked")):
                rows, rule = r
    # e0. explicit instalment / on-account payment of one referenced invoice -> PARTIAL (before short explanations)
    if rows is None and len(refs) == 1 and INSTAL_RX.search(T):
        i = refs[0]
        if I[i]["cur"] == t.currency and A < I[i]["open"]:
            rows, rule = [(i, fmt(A), "", "PARTIAL")], "ref_instalment"
    # e. short payments / FX / partial explanations
    if rows is None:
        res = explain_short(t, T, A, pool, refs, S)
        if res:
            rows, rule = res
    if rows is None and len(refs) == 1:
        i = refs[0]
        o = I[i]["open"]
        if I[i]["cur"] == t.currency:
            if A < o:
                rows, rule = [(i, fmt(A), "", "PARTIAL")], "ref_partial"
            elif A <= o * 1.15:   # cap kept (uncapped OVERPAYMENT P .447 on 2025-01..04)
                # small excess on a single referenced invoice; larger excess = the line pays more invoices
                rows, rule = [(i, fmt(o), "", "OVERPAYMENT")], "ref_overpayment"
    # f. several referenced invoices, cash short of their total: oldest first in full, remainder PARTIAL on the next
    if rows is None and len(refs) >= 2:
        rs = sorted([i for i in refs if I[i]["cur"] == t.currency and I[i]["k"] == "ARI"], key=lambda i: (I[i]["iss"], I[i]["dn"]))
        cn = sum(I[i]["open"] for i in refs if I[i]["k"] == "ARC" and I[i]["cur"] == t.currency)
        left = A + cn
        out = [(i, fmt(I[i]["open"]), "", "") for i in refs if I[i]["k"] == "ARC" and I[i]["cur"] == t.currency]
        for i in rs:
            o = I[i]["open"]
            if left >= o:
                out.append((i, fmt(o), "", ""))
                left -= o
            elif left > 0:
                out.append((i, fmt(left), "", "PARTIAL"))
                left = 0
        if left == 0 and len(out) >= 2:
            rows, rule = out, "refs_partial_last"
    # g. unreferenced on-account payment: oldest open invoice of the (name-identified) party larger than the cash
    if False and rows is None and not refs and same and len(parties) <= 3:   # disabled: 21 % accurate on dev
        big = [i for i in same if I[i]["k"] == "ARI" and I[i]["open"] > A]
        if big and A >= 0.2 * I[big[0]]["open"]:
            rows, rule = [(big[0], fmt(A), "", "PARTIAL")], "fifo_partial"
    if rows:
        conf = {"refs_sum": 0.99, "single_amount_ref": 0.99, "single_amount": 0.95, "fifo_prefix": 0.9,
                "subset1": 0.95, "subset2": 0.75}.get(rule, 0.7)
        S.take(t, rows, "ar:" + rule, conf)
        return True
    return False


def match_ap(t, T, S):
    """Supplier debit not in payment runs (supplier DD, manual transfers) or supplier refund of an APC."""
    I = S.I
    A = abs(cents(t.amount))
    d = t.booking_date
    debit = cents(t.amount) < 0
    sp = S.SX.query(t.entity, T)
    kinds = ("API",) if debit else ("APC",)
    ents = {t.entity} | ({"ES01", "ES02", "PT01"} if t.entity == "ES00" else set())
    refs = []
    for e in ents:
        refs += find_refs(T, e, sp, S, kinds=kinds)
    if not sp and refs:
        sp = {I[refs[0]]["p"]}
    if not sp:
        return False
    if refs and sum(I[i]["open"] for i in refs) == A:
        S.take(t, [(i, fmt(I[i]["open"]), "", "") for i in refs], "ap:refs_sum", 0.97)
        return True
    c = [i for p in sp for i in S.byparty[p] if I[i]["k"] in kinds and I[i]["open"] == A and I[i]["iss"] <= d]
    if c:
        c.sort(key=lambda i: (I[i]["due"], I[i]["iss"]))
        pref = [i for i in c if i in refs]
        S.take(t, [((pref or c)[0], fmt(A), "", "")], "ap:name_amount", 0.9)
        return True
    if debit:
        pool = [i for p in sp for i in S.byparty[p] if I[i]["k"] in ("API", "APC") and I[i]["open"] > 0 and I[i]["iss"] <= d]
        pool.sort(key=lambda i: (I[i]["due"], I[i]["iss"]))
        if 1 < len(pool) <= 22:
            cnt, sol = mitm([S.sgn(I[i]["k"]) * I[i]["open"] for i in pool], A)
            if cnt == 1:
                S.take(t, [(pool[k], fmt(I[pool[k]]["open"]), "", "") for k in sol], "ap:subset", 0.85)
                return True
    return False
