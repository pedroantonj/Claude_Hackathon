"""Engine state: open items (cents), lookup indexes, masters, allocation log.

Every matcher reads/writes through State so balances evolve consistently with our own decisions.
"""
import os
import re
from collections import defaultdict

import pandas as pd

from data_loader.loader import DATA, MASTERS, _read
from .text import PrefixIndex, cents, nz

CREDIT = ("ARC", "APC")


class State:
    def __init__(self, ctx):
        self.ctx = ctx
        self.phase = ctx["phase"]
        self.data_dir = os.path.join(DATA, ctx["phase"])
        self.start = ctx.get("start")
        self.end = ctx.get("end")
        self.out = {}            # bank_txn_id -> list of (item_id, amount_str, category, exception)
        self.why = {}            # bank_txn_id -> (rule, confidence, note)  (audit trail)
        self.alloc_hist = defaultdict(list)   # item_id -> [(bank_txn_id, booking_date, cents, exc)]
        self._masters()
        self._items(ctx["items"])

    # ------------------------------------------------------------------ masters
    def _masters(self):
        self.cust = _read(os.path.join(MASTERS, "customers.csv"))
        self.sup = _read(os.path.join(MASTERS, "suppliers.csv"))
        self.terms = _read(os.path.join(MASTERS, "payment_terms.csv")).set_index("code")
        acc = _read(os.path.join(MASTERS, "bank_accounts.csv"))
        self.accounts = acc.set_index("account_id")
        self.acct_ent = acc.set_index("account_id")["entity"].to_dict()
        self.entities = _read(os.path.join(MASTERS, "entities.csv")).set_index("entity")
        self.cust_country = self.cust.set_index("customer_id")["country"].to_dict()
        fx = _read(os.path.join(MASTERS, "fx_rates.csv"))
        fx["units_per_eur"] = fx["units_per_eur"].astype(float)
        self.fx = {c: g.set_index("date")["units_per_eur"].sort_index() for c, g in fx.groupby("currency")}
        self.children = defaultdict(set)
        for r in self.cust.itertuples():
            if r.parent_customer_id:
                self.children[r.parent_customer_id].add(r.customer_id)
        self.iban_party = {}
        for r in self.cust.itertuples():
            if r.iban:
                self.iban_party[nz(r.iban)] = r.customer_id
        for r in self.sup.itertuples():
            if r.iban:
                self.iban_party.setdefault(nz(r.iban), r.supplier_id)

    def fx_rate(self, cur, date):
        """units of `cur` per 1 EUR at date (last available on/before)."""
        if cur == "EUR":
            return 1.0
        s = self.fx.get(cur)
        if s is None:
            return None
        i = s.index.searchsorted(date, side="right") - 1
        return float(s.iloc[max(i, 0)])

    def to_cur(self, amount, from_cur, to_cur, date):
        if from_cur == to_cur:
            return amount
        a, b = self.fx_rate(from_cur, date), self.fx_rate(to_cur, date)
        if a is None or b is None:
            return None
        return amount / a * b

    # ------------------------------------------------------------------ items
    def _items(self, df):
        I = {}
        cols = ["item_id", "kind", "entity", "party_id", "party_name", "doc_number", "issue_date", "due_date",
                "currency", "total", "open_amount", "payment_terms", "po", "related_document", "payment_method"]
        d = df[cols].copy()
        d["issue_date"] = d["issue_date"].dt.strftime("%Y-%m-%d").fillna("")
        d["due_date"] = d["due_date"].dt.strftime("%Y-%m-%d").fillna("")
        for r in d.itertuples(index=False):
            k = r.kind if r.kind in ("ARI", "ARC", "API", "APC") else r.item_id[:3]
            tot = cents(r.total) if r.total == r.total else 0
            op = cents(r.open_amount) if r.open_amount == r.open_amount else tot
            I[r.item_id] = dict(k=k, e=r.entity, p=r.party_id, pn=r.party_name, dn=r.doc_number, iss=r.issue_date,
                                due=r.due_date, cur=r.currency, tot=tot, open=op, open0=op, terms=r.payment_terms,
                                po=r.po, rel=r.related_document, pm=r.payment_method)
        self.I = I
        full = defaultdict(list)    # (entity, nz docno) -> items
        anyent = defaultdict(list)  # nz docno -> items (any entity)
        suf = defaultdict(list)     # (entity, last 6 digits) -> AR items
        po = defaultdict(list)      # (entity, nz po) -> AR items
        byparty = defaultdict(list)
        supdoc = defaultdict(list)  # (supplier_id, nz supplier invoice number) -> AP items (API/APC)
        for i, v in I.items():
            key = nz(v["dn"])
            full[(v["e"], key)].append(i)
            anyent[key].append(i)
            if v["k"] in ("ARI", "ARC"):
                dd = re.sub(r"\D", "", v["dn"])
                if len(dd) >= 5:
                    suf[(v["e"], dd[-6:])].append(i)
                if v["po"] and v["k"] == "ARI":
                    po[(v["e"], nz(v["po"]))].append(i)
            else:
                supdoc[(v["p"], key)].append(i)
            byparty[v["p"]].append(i)
        for p in byparty:
            byparty[p].sort(key=lambda i: (I[i]["iss"], I[i]["dn"]))
        self.full, self.anyent, self.suf, self.po, self.byparty = full, anyent, suf, po, byparty
        self.supdoc = supdoc
        self.sup_iban = {nz(r.iban): r.supplier_id for r in self.sup.itertuples() if r.iban}
        names = [(r.entity, r.customer_id, r.name) for r in self.cust.itertuples()]
        names += [(r.entity, r.customer_id, r.trade_name) for r in self.cust.itertuples() if r.trade_name]
        seen = set()
        for v in I.values():
            if v["k"] in ("ARI", "ARC") and v["pn"] and (v["p"], v["pn"]) not in seen:
                seen.add((v["p"], v["pn"]))
                names.append((v["e"], v["p"], v["pn"]))
        self.CX = PrefixIndex(names)
        snames = [(r.entity, r.supplier_id, r.name) for r in self.sup.itertuples()]
        # payment factory: ES00 pays suppliers of ES01/ES02/PT01 -> index those under ES00 too
        snames += [("ES00", r.supplier_id, r.name) for r in self.sup.itertuples() if r.entity in ("ES01", "ES02", "PT01")]
        self.SX = PrefixIndex(snames)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def sgn(k):
        return -1 if k in CREDIT else 1

    def take(self, t, rows, rule, conf=1.0, note=""):
        """rows: list of (item_id, amount_str, category, exception). Updates open balances."""
        clean, seen = [], set()
        for it, amt, cat, exc in rows:
            if it:
                if it in seen:
                    continue
                seen.add(it)
            clean.append((it, amt, cat, exc))
        # guard: an item cannot be paid by a bank line booked before the item was issued (0 cases in 2025 golden)
        ok = [r for r in clean if not r[0] or r[0] not in self.I or r[3] in ("RETURNED", "DUPLICATE_PAYMENT")
              or not self.I[r[0]]["iss"] or self.I[r[0]]["iss"] <= t.booking_date]
        if ok and len(ok) < len(clean):
            note = (note + " " if note else "") + f"dropped {len(clean) - len(ok)} item(s) issued after booking"
            clean = ok
        self.out[t.bank_txn_id] = clean
        self.why[t.bank_txn_id] = (rule, conf, note)
        for it, amt, cat, exc in clean:
            if not it or it not in self.I:
                continue
            c = cents(amt)
            if exc == "RETURNED":
                self.I[it]["open"] += c
            elif exc == "DUPLICATE_PAYMENT":
                pass
            elif exc in ("PARTIAL", "SHORT_DISPUTE"):
                self.I[it]["open"] = max(0, self.I[it]["open"] - c)
            else:
                self.I[it]["open"] = 0
            self.alloc_hist[it].append((t.bank_txn_id, t.booking_date, c, exc))

    def category(self, t, cat, exc="", rule="category", conf=0.5, note=""):
        self.out[t.bank_txn_id] = [("", "", cat, exc)]
        self.why[t.bank_txn_id] = (rule, conf, note)
