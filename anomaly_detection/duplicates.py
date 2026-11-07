"""DUPLICATE_PAYMENT detection + structural refund pairing (agent 3; TRAIN P 0.9998 / R 0.926 with entity key).

A line L is a duplicate of item X when
  1. an earlier line E of the same entity with the same signed amount was booked 1-60 days before L;
  2. E settled exactly one ARI/API item X (exception '' or FX_DIFFERENCE) and |amount| == X.total;
  3. X's party has a single invoice with that total (kills fixed monthly fees);
  4. name-token Jaccard(E.text, L.text) >= 0.5 (bank boilerplate removed) or identical texts;
  5. X was not RETURNED (re-collections / re-paid rejected transfers are normal payments);
  6. no open item matched L (the engine only calls match() for otherwise unmatched lines).
Earlier lines come from our own allocations in the window AND from golden_hist lines booked in the 60 days before
the window (cross-boundary duplicates).
Refunds: a line of opposite sign, same account, same |amount|, 5-90 days after a duplicate -> CUSTOMER_REFUND
(debit) / SUPPLIER_REFUND (credit) (TRAIN P 0.997 / R 0.983 incl. keyword rule).
"""
import datetime as dt
import os
import re
from collections import defaultdict

import pandas as pd

from data_loader.loader import DATA, _read
from matching.text import cents, fmt
from .returns import history_tables

STOP = set("ACH CREDIT DEBIT CCD PAYMENT SEPA UEBERWEISUNG EREF NOTPROVIDED SVWZ TRANSFER TRANSFERENCIA TRANSFERENCIAS "
           "TRANSF FAVOR INMEDIATA DIRECT TRTP OVERBOEKING NAME REMI ORDP BONIFICO CAUSALE PRZELEW PRZYCH TYTUL VIR "
           "RECU EMIS MOTIF REF INV FRA FRAS PAGO FACTURA INVOICE ORDER RMR VENDOR PMT OUT SPEI RECIBIDO ENVIADO "
           "CONCEPTO PIX RECEBIDO GUTSCHRIFT".split())
MAX_GAP, REFUND_MIN, REFUND_MAX = 60, 5, 90


def toks(text):
    return {w for w in re.findall(r"[A-Z]{3,}", text.upper()) if w not in STOP}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def _d(s):
    return dt.date.fromisoformat(str(s)[:10])


class DuplicateDetector:
    def __init__(self, S):
        self.S = S
        self.recent = defaultdict(list)    # (entity, signed cents) -> [(date, item, tokens, text)]
        self.dups = defaultdict(list)      # (account, signed cents of the duplicate) -> [date]
        self.flagged = []                  # (bank_txn_id, item, gap, date) for the audit report
        self.refunds = []
        self._seeded = False

    # ---------------------------------------------------------------- seed from golden history (before window)
    def _seed(self):
        self._seeded = True
        S = self.S
        gh = S.ctx.get("golden_hist")
        if gh is None or not len(gh) or "booking_date" not in gh:
            return
        H = history_tables(S)
        lo = H.lo
        g = gh[(gh["booking_date"] >= lo - pd.Timedelta(days=REFUND_MAX)) & (gh["booking_date"] < lo)]
        if not len(g):
            return
        path = os.path.join(DATA, "train", "bank_transactions.csv.gz")
        b = _read(path, usecols=["bank_txn_id", "account_id", "entity", "booking_date", "amount", "text",
                                 "counterparty_name"])
        b = b[b.bank_txn_id.isin(set(g.bank_txn_id))].set_index("bank_txn_id")
        n = g.groupby("bank_txn_id").size()
        for r in g.itertuples(index=False):
            if r.bank_txn_id not in b.index:
                continue
            bl = b.loc[r.bank_txn_id]
            sc = cents(bl.amount)
            if r.exception == "DUPLICATE_PAYMENT":
                self.dups[(bl.account_id, sc)].append(bl.booking_date)
            if n[r.bank_txn_id] != 1 or not r.item_id or r.exception not in ("", "FX_DIFFERENCE"):
                continue
            if r.booking_date < lo - pd.Timedelta(days=MAX_GAP):
                continue
            inf = H.item_info(r.item_id)
            if inf and inf[0] in ("ARI", "API") and abs(sc) == inf[2]:
                txt = bl.text + " " + bl.counterparty_name
                self.recent[(bl.entity, sc)].append((bl.booking_date, r.item_id, toks(txt), txt))

    # ---------------------------------------------------------------- engine hooks
    def record(self, t, T):
        rows = self.S.out.get(t.bank_txn_id, [])
        if len(rows) != 1 or not rows[0][0] or rows[0][3] not in ("", "FX_DIFFERENCE"):
            return
        it = rows[0][0]
        v = self.S.I.get(it)
        if v and v["k"] in ("ARI", "API") and abs(cents(t.amount)) == v["tot"]:
            txt = t.text + " " + t.counterparty_name
            self.recent[(t.entity, cents(t.amount))].append((t.booking_date, it, toks(txt), txt))

    def _returned(self, it):
        return any(h[3] == "RETURNED" for h in self.S.alloc_hist.get(it, [])) or \
            it in history_tables(self.S).hist_returned

    def match(self, t, T):
        if not self._seeded:
            self._seed()
        return self._match_dup(t) or self._match_refund(t) or self._match_refund_text(t)

    def _match_refund_text(self, t):
        """Unmatched credit 'REEMBOLSO <ref>' (not 'REEMBOLSO FRA', which settles an APC) -> SUPPLIER_REFUND
        (TRAIN P=R=1.000 with the keyword set below)."""
        if cents(t.amount) <= 0:
            return False
        T = (t.text + " " + t.remittance_info).upper()
        if re.search(r"REEMBOLSO|DEVOLUCION PAGO|DUPLICAD|PAGO DOBLE", T) and "REEMBOLSO FRA" not in T:
            self.S.category(t, "SUPPLIER_REFUND", "", "supplier_refund_text", 0.85)
            return True
        return False

    def _match_dup(self, t):
        S = self.S
        sc = cents(t.amount)
        cands = self.recent.get((t.entity, sc))
        if not cands:
            return False
        H = history_tables(S)
        d = _d(t.booking_date)
        txt = t.text + " " + t.counterparty_name
        tk = toks(txt)
        for date, it, tk0, txt0 in reversed(cands):
            gap = (d - _d(date)).days
            if gap < 1 or gap > MAX_GAP:
                continue
            inf = H.item_info(it)
            if inf is None or inf[3] != 0:
                continue
            if H.n_by_total.get((inf[1], inf[2]), 1) != 1:
                continue
            if self._returned(it):
                continue
            j = jac(tk, tk0)
            if txt == txt0 or j >= 0.5:
                S.take(t, [(it, fmt(abs(sc)), "", "DUPLICATE_PAYMENT")], "duplicate", 0.9,
                       f"gap={gap}d jaccard={j:.2f}")
                self.dups[(t.account_id, sc)].append(t.booking_date)
                self.flagged.append((t.bank_txn_id, it, gap, t.booking_date))
                return True
        return False

    def _match_refund(self, t):
        sc = cents(t.amount)
        ds = self.dups.get((t.account_id, -sc))
        if not ds:
            return False
        d = _d(t.booking_date)
        for x in ds:
            gap = (d - _d(x)).days
            if REFUND_MIN <= gap <= REFUND_MAX:
                cat = "CUSTOMER_REFUND" if sc < 0 else "SUPPLIER_REFUND"
                self.S.category(t, cat, "", "duplicate_refund", 0.85, f"refund of duplicate booked {x}")
                ds.remove(x)
                self.refunds.append((t.bank_txn_id, cat, gap))
                return True
        return False
