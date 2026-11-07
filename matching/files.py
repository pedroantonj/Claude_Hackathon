"""Deterministic file-based matchers: payment runs (AP), direct-debit batches, card settlements, cheque deposits.

Each matcher: match_xxx(t, T, S) -> True if it allocated the line (writes via S.take).
"""
import glob
import itertools
import os
import re
from collections import defaultdict

import pandas as pd

from data_loader.loader import _read
from .text import cents, fmt, mitm, nz

PAYMENT_FACTORY = ("ES01", "ES02", "PT01", "ES00")


def _dd(a, b):
    """|days| between ISO dates"""
    import datetime as _dt
    try:
        return abs((_dt.date.fromisoformat(b) - _dt.date.fromisoformat(a)).days)
    except ValueError:
        return 999


def _months_ok(fname, S):
    """Keep batch files whose YYYYMM is within [start-2m, end+1m] (train window); all files otherwise."""
    if S.phase != "train" or S.start is None:
        return True
    m = re.search(r"_(\d{6})(?:_|\.)", fname)
    if not m:
        return True
    ym = pd.Timestamp(m.group(1)[:4] + "-" + m.group(1)[4:] + "-01")
    return (S.start - pd.DateOffset(months=2)) <= ym <= (S.end + pd.DateOffset(months=1))


def load_dir(S, d):
    fs = [f for f in sorted(glob.glob(os.path.join(S.data_dir, d, "*.csv"))) if _months_ok(os.path.basename(f), S)]
    return pd.concat([_read(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


class FileMatchers:
    def __init__(self, S):
        self.S = S
        I = S.I
        # ---------------- payment runs
        pr = load_dir(S, "payment_runs")
        self.e2e = defaultdict(list)
        self.run = defaultdict(list)
        self.e2e_info = {}
        self.e2e_acct = defaultdict(set)
        self.e2e_a = defaultdict(list)
        self.used_e2e = set()
        self.n_supkey = 0
        self.run_a = defaultdict(list)
        self.run_acct = defaultdict(set)
        for r in pr.itertuples(index=False):
            ent = S.acct_ent.get(r.debtor_account_id, "")
            ents = {ent} | (set(PAYMENT_FACTORY) if ent == "ES00" else set())
            cands = [i for e in ents for i in S.full.get((e, nz(r.document_number)), []) if I[i]["k"] in ("API", "APC")]
            da = cents(r.document_amount)
            kind = "APC" if da < 0 else "API"
            c2 = [i for i in cands if I[i]["k"] == kind and I[i]["tot"] == abs(da)] or \
                 [i for i in cands if I[i]["k"] == kind] or cands
            if len(c2) > 1:
                bn = nz(r.beneficiary)[:8]
                c3 = [i for i in c2 if I[i]["open"] > 0] or c2
                c3 = [i for i in c3 if nz(I[i]["pn"]).startswith(bn)] or c3
                c2 = [i for i in c3 if I[i]["e"] == ent] or c3
            it = c2[0] if c2 else None
            # AP key = (supplier, supplier invoice number): beneficiary IBAN / name -> supplier
            sp = []
            if r.beneficiary_iban and nz(r.beneficiary_iban) in S.sup_iban:
                sp = [S.sup_iban[nz(r.beneficiary_iban)]]
            else:
                sp = sorted(S.SX.query(ent, r.beneficiary.upper()))
            sc = [i for p in sp for i in S.supdoc.get((p, nz(r.document_number)), ()) if I[i]["k"] == kind]
            if sc:
                sc2 = [i for i in sc if I[i]["tot"] == abs(da)] or sc
                sc2 = [i for i in sc2 if I[i]["open"] > 0] or sc2
                it = sc2[0]
                self.n_supkey += 1
            row = (it, abs(da), cents(r.transfer_amount), r.currency, r.end_to_end_id, r.execution_date)
            # e2e / run ids collide across entities (same PR<entity-less> numbering): key by debtor account
            self.e2e_a[(r.debtor_account_id, r.end_to_end_id)].append(row)
            self.run_a[(r.debtor_account_id, r.run_id)].append(row)
            self.e2e[r.end_to_end_id].append(row)     # flat (used by anomaly_detection.returns)
            self.run[r.run_id].append(row)
            self.e2e_acct[r.end_to_end_id].add(r.debtor_account_id)
            self.run_acct[r.run_id].add(r.debtor_account_id)
            self.e2e_info[r.end_to_end_id] = (r.run_type, r.currency)
        # e2e prefix (16 char truncation in ref1/ref2)
        self.e2e16 = defaultdict(set)
        for (acc, k) in self.e2e_a:
            self.e2e16[(acc, k[:16])].add(k)
        # ---------------- direct debits
        dd = load_dir(S, "direct_debit_batches")
        self.dd = defaultdict(list)
        self.dd_mandate = defaultdict(list)   # mandate_id -> [(collection_date, amount_c, item, batch)]
        for r in dd.itertuples(index=False):
            ent = S.acct_ent.get(r.account_id, "")
            a = cents(r.amount)
            self.dd[r.batch_id].append((ent, r.remittance_info, a, r.mandate_id, r.debtor_name))
        self.dd_rows = dd
        self.mandate_party = {r.sepa_mandate_id: r.customer_id for r in S.cust.itertuples() if r.sepa_mandate_id}
        # mandate -> [(collection_date, amount_c, item)] for linking DD returns
        self.mandate_items = defaultdict(list)
        for r in dd.itertuples(index=False):
            ent = S.acct_ent.get(r.account_id, "")
            a = cents(r.amount)
            self.mandate_items[r.mandate_id].append((r.collection_date, a, self.resolve_ar(ent, r.remittance_info, a, self.mandate_party.get(r.mandate_id))))
        # ---------------- card
        cs = load_dir(S, "card_settlements")
        self.card = defaultdict(list)
        net = defaultdict(int)
        for r in cs.itertuples(index=False):
            self.card[(r.merchant_id, r.settlement_id)].append((S.acct_ent.get(r.account_id, ""), r.order_number,
                                                                 cents(r.gross), r.card, r.operation_date))
            net[(r.merchant_id, r.settlement_id)] += cents(r.net)
        self.card_key = defaultdict(list)
        for (mid, sid), n in net.items():
            self.card_key[(mid, n)].append(sid)
        # ---------------- cheques
        ch = load_dir(S, "cheque_deposits")
        self.chq = defaultdict(list)
        for r in ch.itertuples(index=False):
            self.chq[r.deposit_id].append((S.acct_ent.get(r.account_id, ""), r.stub_reference, cents(r.amount),
                                           r.drawer, r.cheque_number))

    # ------------------------------------------------------------------ AR reference resolution
    def resolve_ar(self, ent, ref, amt_c=None, party=None):
        S, I = self.S, self.S.I
        c = [i for i in S.full.get((ent, nz(ref)), []) if I[i]["k"] in ("ARI", "ARC")]
        if len(c) == 1:
            return c[0]
        if len(c) > 1:
            c2 = [i for i in c if I[i]["open"] > 0] or c
            c2 = [i for i in c2 if I[i]["p"] == party] or c2
            return c2[0]
        dd = re.sub(r"\D", "", ref)
        if len(dd) >= 5:
            c = S.suf.get((ent, dd[-6:].rjust(6, "0")), [])
            c2 = [i for i in c if amt_c is None or I[i]["open"] == amt_c] or c
            c2 = [i for i in c2 if I[i]["open"] > 0] or c2
            c2 = [i for i in c2 if I[i]["p"] == party] or c2
            if c2:
                return c2[-1]
        if party and amt_c:
            # no usable reference: open invoice of the mandate's customer with that amount (oldest first)
            c = [i for i in S.byparty.get(party, ()) if I[i]["k"] == "ARI" and I[i]["open"] == amt_c]
            if c:
                return c[0]
        return None

    # ------------------------------------------------------------------ AP payment runs
    def match_payment_run(self, t, T, S):
        A = abs(cents(t.amount))
        hit, rule = None, ""
        toks = set(re.findall(r"(?:PR|CF)\d{12}-\d{4}", T))
        if t.end_to_end_id:
            toks.add(t.end_to_end_id)
        acc = t.account_id
        # amount in the transfer currency when the bank converted (FX): /OCMT/USD7637,88/ or orig_amount
        alts = {A}
        m = re.search(r"/OCMT/([A-Z]{3})([\d.,]+)", T)
        if m:
            try:
                alts.add(abs(cents(m.group(2).replace(".", "").replace(",", ".") if "," in m.group(2) else m.group(2))))
            except ValueError:
                pass
        if getattr(t, "orig_amount", "") not in ("", None) and t.orig_amount == t.orig_amount:
            try:
                alts.add(abs(cents(t.orig_amount)))
            except (TypeError, ValueError):
                pass

        def key(d, accs, tok):
            if (acc, tok) in d:
                return (acc, tok)
            if len(accs.get(tok, ())) == 1:     # other account of the same entity (payment factory etc.)
                return (next(iter(accs[tok])), tok)
            return None
        for tok in sorted(toks):
            k = key(self.e2e_a, self.e2e_acct, tok)
            if k:
                hit, rule = self.e2e_a[k], "e2e"
                break
        if hit is None:
            # truncated e2e (16 chars) -> candidates with same transfer amount
            for tok in sorted(set(re.findall(r"(?:PR|CF)\d{12}-?\d{0,3}", T))):
                ks = [k for k in sorted(self.e2e16.get((acc, tok[:16]), ())) if self.e2e_a[(acc, k)][0][2] in alts]
                if len(ks) == 1:
                    hit, rule = self.e2e_a[(acc, ks[0])], "e2e16"
                    break
        if hit is None:
            for run in sorted(set(re.findall(r"(?:PR|CF)\d{12}", T))):
                k = key(self.run_a, self.run_acct, run)
                rr = self.run_a.get(k) if k else None
                if not rr:
                    continue
                by_e2e = defaultdict(list)
                for row in rr:
                    by_e2e[row[4]].append(row)
                tot = sum(v[0][2] for v in by_e2e.values())
                if tot in alts:
                    hit, rule = rr, "run_total"
                    break
                # aggregated / bulk file in another currency: "415 TRANSFERS" == number of beneficiaries in the run
                m = re.search(r"(\d+) (?:TRANSFERS|TRANSFERENCIAS|VIREMENTS|PAYMENTS|PAGOS|BONIFICI|UEBERWEISUNGEN)", T)
                if m and int(m.group(1)) == len(by_e2e) and len(by_e2e) > 1:
                    hit, rule = rr, "run_count"
                    break
                ok = [v for v in by_e2e.values() if v[0][2] in alts]
                if len(ok) == 1:
                    hit, rule = ok[0], "run_e2e_amount"
                    break
                # confirming: one debit per maturity date -> group by execution date
                by_d = defaultdict(list)
                for row in rr:
                    by_d[row[5]].append(row)
                ok = [v for v in by_d.values() if sum({r[4]: r[2] for r in v}.values()) == A]
                if len(ok) == 1:
                    hit, rule = ok[0], "run_date_group"
                    break
                # confirming: debit = some beneficiaries (e2e groups) of the run ("N PROVEEDORES")
                groups = [(e, v) for e, v in by_e2e.items() if (k, e) not in self.used_e2e]
                m = re.search(r"(\d+) PROVEEDOR", T)
                n = int(m.group(1)) if m else None
                sol = None
                # one debit per maturity date: beneficiaries whose documents fall due around the booking date
                near = []
                for gi, (e, v) in enumerate(groups):
                    dues = [S.I[r[0]]["due"] for r in v if r[0] and S.I[r[0]]["due"]]
                    if dues and _dd(max(dues), t.booking_date) <= 4:
                        near.append(gi)
                if near and sum(groups[i][1][0][2] for i in near) == A:
                    sol = near
                elif 1 < len(near) <= 24:
                    cnt, sl = mitm([groups[i][1][0][2] for i in near], A)
                    if cnt == 1:
                        sol = [near[i] for i in sl]
                if sol:
                    pass
                elif n and n <= 2:
                    for combo in itertools.combinations(range(len(groups)), n):
                        if sum(groups[i][1][0][2] for i in combo) == A:
                            sol = list(combo)
                            break
                elif len(groups) <= 24:
                    cnt, sl = mitm([g[1][0][2] for g in groups], A)
                    if cnt == 1:
                        sol = sl
                if sol:
                    hit = [row for i in sol for row in groups[i][1]]
                    for i in sol:
                        self.used_e2e.add((k, groups[i][0]))
                    rule = "run_e2e_subset"
                    break
        if not hit:
            return False
        rows = []
        for it, da, *_ in hit:
            if not it:
                continue
            v = S.I[it]
            # document paid for less than its open balance -> PARTIAL, amount = what the run paid
            if v["k"] == "API" and 0 < da < v["open"] - 1:
                exc = "PARTIAL"
            elif v["k"] == "API" and (v["cur"] != t.currency or (hit[0][3] and hit[0][3] != t.currency)):
                exc = "FX_DIFFERENCE"   # fully cleared invoice paid from an account in another currency
            else:
                exc = ""
            rows.append((it, fmt(da), "", exc))
        if not rows:
            return False
        S.take(t, rows, "payment_run:" + rule, 0.97)
        return True

    # ------------------------------------------------------------------ AR batches
    def match_dd(self, t, T, S):
        for tok in re.findall(r"[A-Z]*\d{8,}", T):
            if tok in self.dd:
                rows = []
                used = set()
                for e, ref, a_, mandate, debtor in self.dd[tok]:
                    party = self.mandate_party.get(mandate)
                    if party is None:
                        ps = S.CX.query(e, debtor)
                        party = next(iter(ps)) if len(ps) == 1 else None
                    it = self.resolve_ar(e, ref, a_, party)
                    if it in used:
                        it = None
                    used.add(it)
                    if it:
                        rows.append((it, fmt(a_), "", ""))
                if rows:
                    S.take(t, rows, "dd_batch", 0.96)
                    return True
        return False

    def match_cheque(self, t, T, S):
        for tok in re.findall(r"[A-Z]*\d{6,}", T):
            if tok in self.chq:
                rows = []
                for e, ref, a_, drawer, num in self.chq[tok]:
                    it = self.resolve_ar(e, ref, a_) if ref else None
                    if not it:
                        ps = S.CX.query(e, drawer)
                        c = [i for p in sorted(ps) for i in S.byparty[p] if S.I[i]["open"] == a_ and S.I[i]["k"] == "ARI"]
                        it = c[0] if c else None
                    if it:
                        rows.append((it, fmt(a_), "", ""))
                if rows:
                    S.take(t, rows, "cheque_deposit", 0.95)
                    return True
        return False

    def match_card(self, t, T, S):
        A = abs(cents(t.amount))
        for tok in sorted(set(re.findall(r"\d{9}", T))):
            sids = self.card_key.get((tok, A))
            if sids:
                rows = []
                for e, order, g_, card, od in self.card[(tok, sids[0])]:
                    c = S.full.get((e, nz(order)), [])
                    if c:
                        rows.append((c[0], fmt(g_), "", ""))
                if rows:
                    S.take(t, rows, "card_settlement", 0.97)
                    return True
        return False
