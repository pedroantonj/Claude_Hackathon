"""Relaxed recovery for receipts of an identifiable payer that the main cascade could not allocate (owner: agent 2).

recover(t, T, S, p_unid) -> bool (allocates via S.take when it finds a plausible explanation).
Called only for credits nothing else matched and whose P(truly unidentified) is low (anomaly_detection.unidentified).

Cascade (stop at first hit; rule names "rec:<rule>" for per-rule precision):
  party evidence : PrefixIndex / IBAN (party.ar_parties) -> rapidfuzz payer-name fallback -> parent / group siblings
  refs           : full / tail numbers resolved against the PARTY's open items only (not unique entity-wide)
  refs_sum       : referenced items sum to the cash
  subset         : exact subset over ALL open items of the party (no age cut), ranked by the learned DelayModel;
                   pools > 26 pruned by delay weight
  short          : exceptions.short_payments.explain_short on the full pool
  fifo_partial   : referenced invoices (>= 2) oldest first in full, remainder PARTIAL on the next
  partial_ref    : one referenced invoice, cash < open -> PARTIAL (cash)
  partial_ratio  : one invoice where cash / open is a round fraction (1/n, k/n, .9, .95 ...) -> PARTIAL
  partial_text   : instalment / on-account / retention text, best-ranked invoice with open > cash -> PARTIAL
Weak evidence -> False (stays UNIDENTIFIED).
"""
import re
from collections import defaultdict

from exceptions.short_payments import explain_short
from .learned import payer_names
from .party import INSTAL_RX, _minus_days, _weights, ar_parties, find_refs
from .text import cents, fmt, mitm_all, nz

try:
    from rapidfuzz import fuzz, process
except Exception:   # pragma: no cover
    process = None

RATIOS = (1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.05)
STALE_DAYS = 365


def _names(S, ent):
    cache = S.__dict__.setdefault("_rec_names", {})
    if ent not in cache:
        rows = S.cust[S.cust["entity"] == ent]
        names, ids = [], []
        for cid, n, tn in zip(rows["customer_id"], rows["name"], rows["trade_name"]):
            for x in (n, tn):
                if x:
                    names.append(nz(x))
                    ids.append(cid)
        cache[ent] = (names, ids)
    return cache[ent]


def _groups(S):
    g = S.__dict__.get("_rec_groups")
    if g is None:
        groups, cg = defaultdict(set), {}
        for cid, gid in zip(S.cust["customer_id"], S.cust["group_id"]):
            if gid:
                groups[gid].add(cid)
                cg[cid] = gid
        S._rec_groups = g = (groups, cg)
    return g


def parties_relaxed(t, T, S):
    ps = set(ar_parties(t, T, S))
    how = "name"
    if not ps and process is not None:
        names, ids = _names(S, t.entity)
        if names:
            for cand in payer_names(T, (t.counterparty_name or "").upper()):
                k = nz(cand)
                if len(k) < 6:
                    continue
                hits = process.extract(k, names, scorer=fuzz.partial_ratio, limit=5, score_cutoff=92)
                if hits:
                    top = hits[0][1]
                    ps |= {ids[j] for _, sc, j in hits if sc >= top}
                    how = "fuzzy"
                    break
    if ps and len(ps) <= 3:
        groups, cg = _groups(S)
        for p in sorted(ps):
            gid = cg.get(p)
            if gid and len(groups[gid]) <= 12:
                ps |= groups[gid]
    return ps, how


def tail_refs(T, pool, S):
    """5-7 digit tails / numbers without prefix resolved against the party's open items only."""
    I = S.I
    idx = defaultdict(list)
    for i in pool:
        dd = re.sub(r"\D", "", I[i]["dn"])
        if len(dd) >= 5:
            idx[dd[-6:]].append(i)
            idx[dd[-5:]].append(i)
    out = []
    for tok in sorted(set(re.findall(r"\d{5,12}", T))):
        for k in (tok[-6:], tok[-5:]):
            c = idx.get(k, [])
            if len(c) == 1:
                out.append(c[0])
                break
    return out


from anomaly_detection.unidentified import UNID_THRESHOLD  # noqa: E402

GATED_RULES = set()   # gating by p_unid measured: 93.02/91.33 gated vs 93.04/91.38 ungated -> off
DROP_RULES = {"partial_ref", "short:fx_rate", "short:fx_rate_single", "short:a4_bank_charges"}


def recover(t, T, S, p_unid):
    I = S.I
    A = abs(cents(t.amount))
    d = t.booking_date
    if A <= 0:
        return False
    parties, how = parties_relaxed(t, T, S)
    if not parties or len(parties) > 40:
        return False
    pool = [i for p in sorted(parties) for i in S.byparty.get(p, ())
            if I[i]["open"] > 0 and I[i]["iss"] <= d and I[i]["k"] in ("ARI", "ARC")]
    if not pool:
        return False
    refs = [i for i in find_refs(T, t.entity, parties, S) if i in pool]
    cutoff = _minus_days(d, STALE_DAYS)
    # old open items of a party are mostly never-paid noise: keep them only when referenced
    fresh = [i for i in pool if I[i]["iss"] >= cutoff]
    refs += [i for i in tail_refs(T, [i for i in pool if I[i]["cur"] == t.currency], S) if i not in refs]
    same = sorted([i for i in fresh if I[i]["cur"] == t.currency] + [i for i in refs if I[i]["iss"] < cutoff],
                  key=lambda i: (I[i]["iss"], I[i]["dn"]))
    sg = S.sgn
    W = _weights(same, d, S) or {i: 0.0 for i in same}
    rows, rule = None, ""

    # refs summing to the cash
    if refs and sum(sg(I[i]["k"]) * I[i]["open"] for i in refs if I[i]["cur"] == t.currency) == A:
        rows, rule = [(i, fmt(I[i]["open"]), "", "") for i in refs], "refs_sum"

    # single invoice paid by a round fraction of its open amount (instalments, retentions, 50 % advances)
    if rows is None:
        big = [i for i in same if I[i]["k"] == "ARI" and I[i]["open"] > A]
        rr = [i for i in big if any(abs(A - x * I[i]["open"]) <= 1 for x in RATIOS)]
        if rr:
            i = max(rr, key=lambda i: (i in refs, W[i]))
            rows, rule = [(i, fmt(A), "", "PARTIAL")], "partial_ratio"

    # exact subset over all open items of the party (refs forced in), ranked by payment-delay likelihood
    if rows is None and same:
        forced = [i for i in refs if i in same]
        # unreferenced credit notes make almost any amount reachable: leave them out
        rest = [i for i in same if i not in forced and I[i]["k"] == "ARI"]
        target = A - sum(sg(I[i]["k"]) * I[i]["open"] for i in forced)
        if len(rest) > 26:
            rest = sorted(rest, key=lambda i: (I[i]["k"] != "ARC", -W[i], I[i]["iss"]))[:26]
        vals = [sg(I[i]["k"]) * I[i]["open"] for i in rest]
        sols = mitm_all(vals, target, cap=64, maxn=26) if target != 0 else []
        if 0 < len(sols) <= 3:
            best = max(sols, key=lambda sl: (sum(W[rest[k]] for k in sl), -len(sl)))
            chosen = forced + [rest[k] for k in best]
            rows = [(i, fmt(I[i]["open"]), "", "") for i in chosen]
            rule = "subset1" if len(sols) == 1 else "subset_ranked"

    # short-payment explanations (discount, withholding, bank charges, FX, dispute) on the full pool
    if rows is None:
        res = explain_short(t, T, A, pool, refs, S)
        if res:
            rows, rule = res
            rule = "short:" + rule

    # several referenced invoices: oldest in full, remainder PARTIAL on the next
    if rows is None and len(refs) >= 2:
        rs = sorted([i for i in refs if I[i]["cur"] == t.currency and I[i]["k"] == "ARI"], key=lambda i: (I[i]["iss"], I[i]["dn"]))
        cn = [i for i in refs if I[i]["k"] == "ARC" and I[i]["cur"] == t.currency]
        left = A + sum(I[i]["open"] for i in cn)
        out = [(i, fmt(I[i]["open"]), "", "") for i in cn]
        for i in rs:
            o = I[i]["open"]
            if left >= o:
                out.append((i, fmt(o), "", ""))
                left -= o
            elif left > 0:
                out.append((i, fmt(left), "", "PARTIAL"))
                left = 0
        if left == 0 and len(out) >= 2:
            rows, rule = out, "fifo_partial"

    # one referenced invoice, cash short of it
    if rows is None:
        rb = [i for i in same if i in refs and I[i]["k"] == "ARI" and I[i]["open"] > A]
        if len(rb) == 1:
            rows, rule = [(rb[0], fmt(A), "", "PARTIAL")], "partial_ref"

    if not rows:
        return False
    # gating (coordinator, measured on lite 2024-11..12 / 2025-05..06):
    #  - strong rules (line-exact >= .86): always
    #  - subset rules (line-exact .21-.67): only when P(truly unidentified) is low
    #  - rules at 0 % line precision on both windows: never
    if rule in DROP_RULES:
        return False
    if rule in GATED_RULES and p_unid >= UNID_THRESHOLD:
        return False
    S.take(t, rows, f"rec:{rule}", 0.6, f"p_unid={p_unid:.2f};party={how}")
    return True
