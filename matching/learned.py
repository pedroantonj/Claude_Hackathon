"""Lookups learned from golden HISTORY only (ctx['golden_hist']: rows booked before the window / phase).

learn(S, ctx) attaches
  S.learned_iban : nz(payer IBAN) -> party  (unambiguous in history)
  S.CX           : PrefixIndex rebuilt with payer-name aliases seen in history bank texts (renamed customers, trade
                   names, payers whose printed name differs from the master), keyed to the party that was paid.
Never uses golden of the evaluated window (golden_hist is cut at the window start by data_loader.history).
"""
import datetime as dt
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from data_loader.loader import DATA, _read
from .text import PrefixIndex, nz

# payer-name extraction per bank format (group 1 = printed payer name, possibly truncated)
NAME_RX = [re.compile(p) for p in (
    r"TRANSFERENCIAS (.+?) \|",
    r"\d{8}-(.+?)(?: \||$)",
    r"TRANSFERENCIA DE (.+?) \| (?:CONCEPTO|ORDENANTE)",
    r"TRANSFERENCIA DE (.+? \| .+?) \| CONCEPTO",
    r"/DE (.+?) /MOTIF",
    r"/ORDP/(.+?)/",
    r"BONIFICO DA (.+?) CAUSALE",
    r"PRZELEW PRZYCH\. (.+?)(?: TYT| FV| FAKT| ZAPL|$)",
    r"TT IN (.+?)(?: INV| PAYMENT| NONREF| REMIT|$)",
    r"DIRECT CREDIT (.+?)(?:  | INV| PAYMENT|$)",
    r"ACH CREDIT (.+?) CCD",
    r"WIRE IN /ORG/(.+?) /",
    r"PIX RECEBIDO (.+?)(?: PAGAMENTO| PAYMENT| NF| NONREF|$)",
    r"^FP (.+?)(?:  | INVOICE| PAYMENT| REF|$)",
    r"ORD\. (.+?)(?:  |$)",
)]


def payer_names(text, cp=""):
    out = [cp] if cp else []
    for rx in NAME_RX:
        m = rx.search(text)
        if m:
            out.append(m.group(1).replace(" | ", ""))
    return [n for n in out if len(nz(n)) >= 6]


def _item_party():
    m = {}
    for f, col in (("ar_invoices.csv.gz", "customer_id"), ("credit_notes.csv.gz", "party_id")):
        d = _read(os.path.join(DATA, "train", f), usecols=["item_id", col])
        m.update(zip(d["item_id"], d[col]))
    return m


def learn(S, ctx, log=print):
    try:
        learn_delays(S, ctx, log)
    except Exception as ex:   # never break the pipeline on the optional ranker
        S.delay = None
        log(f"[learned] delay model failed: {type(ex).__name__}: {ex}")
    S.learned_iban = {}
    S.n_alias = 0
    hist = ctx.get("golden_hist")
    if hist is None or len(hist) == 0:
        return
    h = hist[hist["item_id"].str[:2].eq("AR")]
    if "booking_date" in h.columns and h["booking_date"].notna().any():
        h = h[h["booking_date"] >= h["booking_date"].max() - pd.Timedelta(days=550)]
    if len(h) == 0:
        return
    ip = {i: v["p"] for i, v in S.I.items()}
    if set(h["item_id"]) - set(ip):
        ip.update(_item_party())
    h = h.assign(party=h["item_id"].map(ip)).dropna(subset=["party"])
    g = h.groupby("bank_txn_id")["party"]
    lp = g.first()[g.nunique() == 1]
    b = _read(os.path.join(DATA, "train", "bank_transactions.csv.gz"),
              usecols=["bank_txn_id", "entity", "text", "counterparty_name", "counterparty_account"])
    b = b[b["bank_txn_id"].isin(lp.index)]
    b = b.assign(party=b["bank_txn_id"].map(lp))
    ib = defaultdict(Counter)
    for acc, p in zip(b["counterparty_account"], b["party"]):
        if acc:
            ib[nz(acc)][p] += 1
    for acc, c in ib.items():
        p, n = c.most_common(1)[0]
        if n >= 0.8 * sum(c.values()):
            S.learned_iban[acc] = p
    alias = set()
    for e, txt, cp, p in zip(b["entity"], b["text"], b["counterparty_name"], b["party"]):
        for n in payer_names(txt.upper(), cp.upper()):
            alias.add((e, p, n))
    add = [(e, p, n) for e, p, n in alias if p not in S.CX.query(e, n)]
    S.n_alias = len(add)
    if add:
        names = [(r.entity, r.customer_id, r.name) for r in S.cust.itertuples()]
        names += [(r.entity, r.customer_id, r.trade_name) for r in S.cust.itertuples() if r.trade_name]
        seen = set()
        for v in S.I.values():
            if v["k"] in ("ARI", "ARC") and v["pn"] and (v["p"], v["pn"]) not in seen:
                seen.add((v["p"], v["pn"]))
                names.append((v["e"], v["p"], v["pn"]))
        S.CX = PrefixIndex(names + add)
    log(f"[learned] {len(S.learned_iban):,} payer IBANs, {S.n_alias:,} name aliases from {len(b):,} history lines")


# ---------------------------------------------------------------------------- payment-delay model (history only)
import numpy as np  # noqa: E402

LAG_LO, LAG_HI, LAG_W = -120, 480, 4
NB = (LAG_HI - LAG_LO) // LAG_W + 1


def _bin(lag):
    return (min(max(lag, LAG_LO), LAG_HI) - LAG_LO) // LAG_W


class DelayModel:
    """P(booking_date - due_date) per customer (>= 12 paid items) -> customer_type x payment_method -> method -> global.
    weight(item, d) = log pmf(lag) - log survival(lag): log-odds that the item is paid by a line booked on d rather than
    still open. Ranking subsets by sum of weights reproduces FIFO / due-date behaviour learned from history."""

    def __init__(self, counts, ctype):
        self.ctype = ctype
        g = counts.get(("g",), np.ones(NB))
        self.tables = {}
        gp = (g + 1.0) / (g.sum() + NB)
        self.tables[("g",)] = gp
        for k, c in counts.items():
            if k[0] == "m":
                self.tables[k] = (c + 20 * gp) / (c.sum() + 20)
        for k, c in counts.items():
            if k[0] == "t":
                par = self.tables.get(("m", k[2]), gp)
                self.tables[k] = (c + 20 * par) / (c.sum() + 20)
        for k, c in counts.items():
            if k[0] == "c" and c.sum() >= 12:
                par = self.tables.get(("t", ctype.get(k[1], ""), k[2]), self.tables.get(("m", k[2]), gp))
                self.tables[k] = (c + 10 * par) / (c.sum() + 10)
        self.logp, self.logs = {}, {}
        for k, p in self.tables.items():
            sf = np.clip(1.0 - np.cumsum(p) + p, 1e-6, 1.0)   # P(lag >= bin)
            self.logp[k] = np.log(p)
            self.logs[k] = np.log(sf)

    def key(self, party, method):
        for k in (("c", party, method), ("t", self.ctype.get(party, ""), method), ("m", method)):
            if k in self.logp:
                return k
        return ("g",)

    def weight(self, v, d):
        ref = v["due"] or v["iss"]
        try:
            lag = (dt.date.fromisoformat(d) - dt.date.fromisoformat(ref)).days
        except ValueError:
            return 0.0
        if lag > LAG_HI:
            return -8.0
        k = self.key(v["p"], v.get("pm") or "")
        b = _bin(lag)
        return float(self.logp[k][b] - self.logs[k][b])


def learn_delays(S, ctx, log=print):
    S.delay = None
    hist = ctx.get("golden_hist")
    if hist is None or len(hist) == 0:
        return
    h = hist[hist["item_id"].str[:3].eq("ARI") & ~hist["exception"].isin(["RETURNED", "DUPLICATE_PAYMENT"])]
    if "booking_date" not in h.columns:
        return
    h = h[h["booking_date"] >= h["booking_date"].max() - pd.Timedelta(days=730)]
    d = _read(os.path.join(DATA, "train", "ar_invoices.csv.gz"),
              usecols=["item_id", "customer_id", "due_date", "payment_method"])
    d = d[d["item_id"].isin(set(h["item_id"]))].set_index("item_id")
    h = h[h["item_id"].isin(d.index)]
    due = pd.to_datetime(pd.Series(d.loc[h["item_id"], "due_date"].values), errors="coerce")
    bk = pd.to_datetime(pd.Series(h["booking_date"].values))
    lag = ((bk - due) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    ok = ~np.isnan(lag)
    party = d.loc[h["item_id"], "customer_id"].values[ok]
    meth = d.loc[h["item_id"], "payment_method"].values[ok]
    bins = ((np.clip(lag[ok], LAG_LO, LAG_HI) - LAG_LO) // LAG_W).astype(int)
    ctype = dict(zip(S.cust["customer_id"], S.cust["customer_type"]))
    counts = defaultdict(lambda: np.zeros(NB))
    for p, m_, b in zip(party, meth, bins):
        counts[("g",)][b] += 1
        counts[("m", m_)][b] += 1
        counts[("t", ctype.get(p, ""), m_)][b] += 1
        counts[("c", p, m_)][b] += 1
    S.delay = DelayModel(dict(counts), ctype)
    log(f"[learned] delay model: {int(counts[('g',)].sum()):,} paid AR items, "
        f"{sum(1 for k in S.delay.tables if k[0] == 'c'):,} customer-specific profiles")
