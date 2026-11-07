"""RETURNED detection and linking (agent 3).

Detection (TRAIN P=R=1.000): bank codes, or the text regex below.
Linking (validated on TRAIN golden):
  * rejected supplier transfer: payment-run end_to_end_id `PR\\d{12}-\\d{4}` in text -> every document of that transfer
  * SEPA/BACS/ACH DD return: mandate `VTM-XX-nnnnnnn` + amount -> most recent DD collection 1-140 days before
    (same account preferred) -> its remittance (our document number) -> item  (~85 % exact item)
  * card chargeback: card last-4 + gross amount -> card_settlements order -> item (282/289 unique)
  * returned cheque: cheque number + amount -> cheque_deposits stub reference -> item (460/538 unique)
Collections made BEFORE the window/phase are handled: batch files and invoice tables are read over a wider time
range (and from train/ for live/final); items closed before the window (absent from S.I) are still returned by id.

is_return(t, T) -> bool ; match_return(t, T, S, FM) -> bool (allocates the re-opened items).
history_tables(S) -> cached helper shared with duplicates.py / flags.py.
"""
import datetime as dt
import glob
import os
import re
from collections import defaultdict

import pandas as pd

from data_loader.loader import DATA, _read
from matching.text import cents, fmt, nz

RET_CODES = {"PMNT/RDDT/UPDD", "PMNT/ICDT/RRTN", "PMNT/CCRD/CHRG", "PMNT/RCHQ/UPCQ", "NRTI"}
RET_N43 = {("14", "021"), ("14", "110"), ("14", "370"), ("14", "908")}
RET_TEXT = re.compile(r"^RETURN |SDD RETURN|RUECKLASTSCHRIFT|^DEVOLUCION RECIBO|^DEVOL\.RECIBOS|"
                      r"^DEVOLUCION TRANSFERENCIA|CHARGEBACK|CONTRACARGO|RETURNED ITEM|IMPAYE")
MANDATE = re.compile(r"VTM-[A-Z]{2}-\d{7}")
LOOKBACK_DAYS = 140


def is_return(t, T):
    if t.bank_code in RET_CODES:
        return True
    if (t.bank_code, t.bank_subcode) in RET_N43:
        return True
    return "COMISION" not in T and bool(RET_TEXT.search(t.text.upper()))


# ---------------------------------------------------------------------------------------------- history tables
def _dirs(S):
    d = [S.data_dir]
    if S.phase in ("live", "final"):
        d.append(os.path.join(DATA, "train"))
        if S.phase == "final":
            d.append(os.path.join(DATA, "live"))
    return d


def _window(S):
    b = S.ctx["bank"]["booking_date"]
    return b.min(), b.max()


def _batch_files(S, sub, months_back=6):
    lo, hi = _window(S)
    lo = (lo - pd.DateOffset(months=months_back)).strftime("%Y%m")
    hi = hi.strftime("%Y%m")
    out = []
    for d in _dirs(S):
        for f in sorted(glob.glob(os.path.join(d, sub, "*.csv"))):
            m = re.search(r"_(\d{6})(?:_|\.)", os.path.basename(f))
            if not m or lo <= m.group(1) <= hi:
                out.append(f)
    return out


class HistTables:
    """Invoice tables incl. items already closed, batch files over a wider window, golden-history facts."""

    def __init__(self, S):
        self.S = S
        lo, hi = _window(S)
        self.lo, self.hi = lo, hi
        parts = []
        min_iss = (lo - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        max_iss = hi.strftime("%Y-%m-%d")
        for d in _dirs(S):
            for fn, cols, kind in (("ar_invoices.csv.gz", ["item_id", "entity", "document_number", "customer_id",
                                                            "issue_date", "total_amount"], "ARI"),
                                   ("ap_invoices.csv.gz", ["item_id", "entity", "supplier_invoice_number",
                                                            "supplier_id", "issue_date", "total_amount"], "API")):
                p = os.path.join(d, fn)
                if not os.path.exists(p):
                    continue
                x = _read(p, usecols=cols)
                x = x[cols]
                x.columns = ["item_id", "entity", "doc", "party", "iss", "tot"]
                x = x[(x.iss >= min_iss) & (x.iss <= max_iss)]
                x["k"] = kind
                parts.append(x)
            p = os.path.join(d, "credit_notes.csv.gz")
            if os.path.exists(p):
                x = _read(p, usecols=["item_id", "side", "entity", "document_number", "party_id", "issue_date",
                                      "total_amount"])
                x = x[x.side == "AP"][["item_id", "entity", "document_number", "party_id", "issue_date", "total_amount"]]
                x.columns = ["item_id", "entity", "doc", "party", "iss", "tot"]
                x = x[(x.iss >= min_iss) & (x.iss <= max_iss)]
                x["k"] = "APC"
                parts.append(x)
        it = pd.concat(parts, ignore_index=True).drop_duplicates("item_id")
        it["totc"] = (pd.to_numeric(it.tot, errors="coerce").fillna(0) * 100).round().astype("int64")
        it["dk"] = it.doc.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        self.items = it.set_index("item_id")
        self.info = dict(zip(it.item_id, zip(it.k, it.party, it.totc)))
        ar = it[it.k == "ARI"]
        self.docmap = dict(zip(zip(ar.entity, ar.dk), ar.item_id))
        ap = it[it.k.isin(["API", "APC"])]
        self.apdoc = defaultdict(list)
        for e, k, i, c, kd in zip(ap.entity, ap.dk, ap.item_id, ap.totc, ap.k):
            self.apdoc[(e, k)].append((i, c, kd))
        dg = ar.dk.str.replace(r"\D", "", regex=True).str[-6:]
        self.suf = defaultdict(list)
        for e, s, i, c in zip(ar.entity, dg, ar.item_id, ar.totc):
            if len(s) >= 5:
                self.suf[(e, s)].append((i, c))
        self.n_by_total = it.groupby(["party", "totc"]).size().to_dict()
        self.by_party_total = defaultdict(list)
        for i, p, c, k, iss in zip(ar.item_id, ar.party, ar.totc, ar.k, ar.iss):
            self.by_party_total[(p, c)].append((iss, i))
        gh = S.ctx.get("golden_hist")
        self.hist_returned = set()
        if gh is not None and len(gh):
            self.hist_returned = set(gh.loc[gh["exception"] == "RETURNED", "item_id"])
        self._dd = self._card = self._chq = self._pr = None

    def item_info(self, it):
        """(kind, party, total_cents, open_cents). Items not in S.I were closed at window start -> open 0."""
        v = self.S.I.get(it)
        if v is not None:
            return v["k"], v["p"], v["tot"], v["open"]
        x = self.info.get(it)
        if x is None:
            return None
        return x[0], x[1], int(x[2]), 0

    def resolve_ar(self, ent, ref, amt_c=None):
        it = self.docmap.get((ent, nz(ref)))
        if it:
            return it
        d = re.sub(r"\D", "", ref)
        if len(d) >= 5:
            c = self.suf.get((ent, d[-6:].rjust(6, "0")), [])
            c2 = [i for i, tc in c if amt_c is None or tc == amt_c] or [i for i, _ in c]
            if c2:
                return c2[-1]
        return None

    def dd(self):
        if self._dd is None:
            self._dd = defaultdict(list)
            for f in _batch_files(self.S, "direct_debit_batches"):
                x = _read(f, usecols=["collection_date", "account_id", "mandate_id", "amount", "remittance_info"])
                for r in x.itertuples(index=False):
                    self._dd[r.mandate_id].append((r.collection_date, cents(r.amount), r.account_id, r.remittance_info))
        return self._dd

    def pr(self):
        """payment-run e2e -> [(entity, doc_number, document_type, doc_amount_cents)] over a wide window."""
        if self._pr is None:
            self._pr = defaultdict(list)
            for f in _batch_files(self.S, "payment_runs", months_back=4):
                x = _read(f, usecols=["debtor_account_id", "end_to_end_id", "document_number", "document_amount"])
                for r in x.itertuples(index=False):
                    self._pr[r.end_to_end_id].append((self.S.acct_ent.get(r.debtor_account_id, ""), r.document_number,
                                                      cents(r.document_amount)))
        return self._pr

    def resolve_ap(self, ent, doc, amt_c):
        """API (amt>0) / APC (amt<0) by document number; payment factory ES00 pays ES01/ES02/PT01."""
        kind = "APC" if amt_c < 0 else "API"
        ents = [ent] + (["ES01", "ES02", "PT01"] if ent == "ES00" else [])
        c = [x for e in ents for x in self.apdoc.get((e, nz(doc)), ()) if x[2] == kind]
        if not c:
            return None
        c2 = [x for x in c if x[1] == abs(amt_c)] or c
        c3 = [x for x in c2 if self.item_info(x[0])[3] == 0] or c2      # prefer already-paid items
        return c3[-1][0]

    def card(self):
        if self._card is None:
            self._card = defaultdict(list)
            for f in _batch_files(self.S, "card_settlements"):
                x = _read(f, usecols=["settlement_date", "account_id", "order_number", "card", "gross"])
                for r in x.itertuples(index=False):
                    self._card[(r.card[-4:], cents(r.gross))].append((r.settlement_date, r.account_id, r.order_number))
        return self._card

    def chq(self):
        if self._chq is None:
            self._chq = defaultdict(list)
            for f in _batch_files(self.S, "cheque_deposits"):
                x = _read(f, usecols=["deposit_date", "account_id", "cheque_number", "amount", "drawer",
                                      "stub_reference"])
                for r in x.itertuples(index=False):
                    self._chq[(r.cheque_number.lstrip("0"), cents(r.amount))].append(
                        (r.deposit_date, r.account_id, r.stub_reference, r.drawer))
        return self._chq


def history_tables(S):
    h = getattr(S, "_a3_hist", None)
    if h is None:
        h = HistTables(S)
        S._a3_hist = h
    return h


# ---------------------------------------------------------------------------------------------- linking
def _gap(d_line, d_other):
    return (dt.date.fromisoformat(d_line[:10]) - dt.date.fromisoformat(d_other[:10])).days


def _take(S, t, it, A, rule, conf):
    S.take(t, [(it, fmt(A), "", "RETURNED")], rule, conf)
    return True


def _party_fallback(t, T, S, A, name=None):
    """Most recently cleared ARI of the named party with total == amount (engine state, then history)."""
    I = S.I
    parties = S.CX.query(t.entity, name or T)
    c = [i for p in parties for i in S.byparty[p] if I[i]["k"] == "ARI" and I[i]["tot"] == A and I[i]["open"] == 0]
    if c:
        c.sort(key=lambda i: (S.alloc_hist[i][-1][1] if S.alloc_hist[i] else "", I[i]["iss"]))
        return c[-1]
    H = history_tables(S)
    c = [x for p in parties for x in H.by_party_total.get((p, A), ()) if x[1] not in I or I[x[1]]["open"] == 0]
    if c:
        return max(c)[1]
    return None


def match_return(t, T, S, FM):
    A = abs(cents(t.amount))
    d = t.booking_date
    # 1. rejected supplier transfer: payment-run e2e reference -> documents of that transfer
    H = history_tables(S)
    for tok in set(re.findall(r"(?:PR|CF)\d{12}-\d{4}", T)):
        docs = H.pr().get(tok)
        if docs:
            out = []
            for ent, doc, da in docs:
                it = H.resolve_ap(ent, doc, da)
                if it:
                    out.append((it, fmt(abs(da)), "", "RETURNED"))
            if out:
                S.take(t, out, "return:payment_run_docs", 0.95)
                return True
        rows = FM.e2e.get(tok)
        if rows:
            out = [(it, fmt(da), "", "RETURNED") for it, da, *_ in rows if it]
            if out:
                S.take(t, out, "return:payment_run", 0.95)
                return True
    # 2. DD return: mandate + amount -> most recent collection 1..140 days before (same account first)
    m = MANDATE.findall(T)
    if m:
        cands = []
        for cd, a, acc, rem in H.dd().get(m[0], ()):
            if a == A and 1 <= _gap(d, cd) <= LOOKBACK_DAYS:
                cands.append((acc == t.account_id, cd, rem, acc))
        for same_acc, cd, rem, acc in sorted(cands, reverse=True):
            it = H.resolve_ar(S.acct_ent.get(acc, t.entity), rem, A)
            if it:
                return _take(S, t, it, A, "return:dd_mandate", 0.85)
    # 3. card chargeback: last-4 + gross
    mc = re.search(r"\*{4}(\d{4})", T)
    if mc:
        c = [x for x in H.card().get((mc.group(1), A), ()) if 0 <= _gap(d, x[0]) <= 400]
        if c:
            sd, acc, order = max(c)
            ent = S.acct_ent.get(acc, t.entity)
            it = H.docmap.get((ent, nz(order))) or next(iter(S.full.get((ent, nz(order)), [])), None)
            if it:
                return _take(S, t, it, A, "return:card", 0.9)
    # 4. returned cheque: cheque number + amount
    mq = re.search(r"(?:CHK|CHQ|CHEQUE(?: IMPAYE| DEVUELTO| RETURNED)?)\.? ?N?[O°]?\.? ?(\d{5,})", T)
    if mq:
        c = [x for x in H.chq().get((mq.group(1).lstrip("0"), A), ()) if 0 <= _gap(d, x[0]) <= 400]
        if c:
            _, acc, stub, drawer = max(c)
            ent = S.acct_ent.get(acc, t.entity)
            it = (H.resolve_ar(ent, stub, A) if stub else None) or _party_fallback(t, T, S, A, drawer)
            if it:
                return _take(S, t, it, A, "return:cheque", 0.85)
    # 5. fallback: party name + already-cleared item with the same total
    it = _party_fallback(t, T, S, A)
    if it:
        return _take(S, t, it, A, "return:party_amount", 0.6)
    return False
