"""Phase / dev-window context: bank lines + open items as they were at the window start.

Leakage rule: in dev mode (phase=train with a window), open balances and every golden-derived
artefact are built ONLY from golden rows whose bank line was booked before `start`.
"""
import gc
import os
import pickle

import numpy as np
import pandas as pd

from .loader import (CACHE, DATA, _p, _read, _num, _date, load_bank, load_items, set_phase_masters,  # noqa: F401
                     _period_items, _ITEM_COLS)

CTX_VERSION = "v3"   # bump when the context layout changes (cache key)
FULL_CLEAR_EXC = {"", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "FX_DIFFERENCE", "OVERPAYMENT",
                  "PARTIAL", "SHORT_DISPUTE"}


def load_golden(phase="train", before=None, with_dates=True, booking=None):
    """Golden rows (+ booking_date). `before`: keep only rows of bank lines booked < before.
    booking: optional Series bank_txn_id -> booking_date (avoids re-reading the bank file)."""
    g = _read(_p(phase, "golden_allocations.csv.gz"),
              usecols=["bank_txn_id", "item_id", "amount", "category", "exception", "segment"])
    g["amount_f"] = _num(g["amount"])
    g["amount"] = g["amount_f"]   # float (was str): kept for API compatibility, saves ~0.3 GB
    for c in ["category", "exception", "segment"]:
        g[c] = g[c].astype("category")
    if with_dates or before is not None:
        if booking is None:
            bt = _read(_p(phase, "bank_transactions.csv.gz"), usecols=["bank_txn_id", "booking_date"])
            booking = pd.Series(_date(bt["booking_date"]).values, index=bt["bank_txn_id"].values)
        g["booking_date"] = g["bank_txn_id"].map(booking)
        if before is not None:
            g = g[g["booking_date"] < pd.Timestamp(before)].reset_index(drop=True)
    return g


def _cleared(golden_before):
    """item_id -> cleared amount: normal/partial rows clear, RETURNED re-opens, DUPLICATE does not clear."""
    gi = golden_before[golden_before["item_id"] != ""]
    exc = gi["exception"].astype(str).values
    sign = np.where(exc == "RETURNED", -1.0, np.where(exc == "DUPLICATE_PAYMENT", 0.0, 1.0))
    return pd.Series(gi["amount_f"].values * sign, index=gi["item_id"].values).groupby(level=0).sum()


def open_items_from_history(items, golden_before):
    """open = total - cleared (normal/partial rows) + reopened (RETURNED rows). DUPLICATE does not clear."""
    cleared = _cleared(golden_before)
    items = items.copy()
    items["open_amount"] = (items["total"] - items["item_id"].map(cleared).fillna(0.0)).round(2)
    return items


def build_context(phase, start=None, end=None, cache=True):
    """Cached wrapper (pickle in outputs/cache) around _build_context. Also points MASTERS to the phase masters."""
    set_phase_masters(phase)
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    code = hashlib.md5(b"".join(open(os.path.join(here, f), "rb").read() for f in ("loader.py", "history.py"))).hexdigest()[:8]
    key = os.path.join(CACHE, f"ctx_{CTX_VERSION}_{code}_{phase}_{start}_{end}.pkl")   # code hash: no stale contexts
    if cache and os.path.exists(key):
        gc.disable()
        try:
            with open(key, "rb") as fh:
                return pickle.load(fh)
        finally:
            gc.enable()
    ctx = _build_context(phase, start, end)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        tmp = f"{key}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(ctx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, key)
    return ctx


def _build_context(phase, start=None, end=None):
    """Returns dict(bank, items, golden_hist, phase, start, end).

    - live/final/sample: bank lines of the phase, items = loader items (open_at_start + period),
      golden_hist = ALL train golden (+ booking_date) — no golden exists for live/final.
    - train + window [start, end]: bank lines in window; items issued <= end with open balance at `start`
      derived from golden before start; golden_hist = golden before start (for learned lookups).
    """
    if phase != "train":
        bank = load_bank(phase)
        items = load_items(phase)
        hist = None
        if phase != "sample" and os.path.exists(_p("train", "golden_allocations.csv.gz")):
            hist = load_golden("train")
        return dict(bank=bank, items=items, golden_hist=hist, phase=phase, start=start, end=end)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    bank = load_bank("train")
    booking = pd.Series(bank["booking_date"].values, index=bank["bank_txn_id"].values)
    hist = load_golden("train", before=start, booking=booking)
    del booking
    bank = bank[(bank["booking_date"] >= start) & (bank["booking_date"] <= end)].reset_index(drop=True)
    cleared = _cleared(hist)
    end_s = end.strftime("%Y-%m-%d")

    def keep(d):   # per-file filter: issued <= end and still open at start (cuts 5.6M rows to ~0.5M early)
        tot = pd.to_numeric(d["total"], errors="coerce")
        op = (tot - d["item_id"].map(cleared).fillna(0.0)).round(2)
        return (d["issue_date"] <= end_s) & (op > 0.005)

    items = load_items("train", keep=keep)
    items = open_items_from_history(items, hist)
    items = items[items["open_amount"] > 0.005].reset_index(drop=True)
    items["source"] = pd.Categorical(np.where(items["issue_date"] < start, "open_at_start", "period"))
    gc.collect()
    return dict(bank=bank, items=items, golden_hist=hist, phase="train", start=start, end=end)
