"""Leakage guards: dev-window context must only use golden booked before the window start."""
import glob
import os
import re
import time

import pandas as pd

from _util import LOCK, ROOT, skip

WINDOW = os.environ.get("BANKREC_LEAK_WINDOW", "2025-05-01,2025-06-30").split(",")


def test_only_history_module_reads_golden():
    """Static guard: golden_allocations may only be opened by data_loader/history.py (+ evaluation/tests)."""
    offenders = []
    for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        rel = os.path.relpath(p, ROOT)
        if rel.split(os.sep)[0] in ("evaluation", "tests", "notes", "outputs") or rel == os.path.join("data_loader", "history.py"):
            continue
        if re.search(r"golden_allocations", open(p, encoding="utf-8", errors="ignore").read()):
            offenders.append(rel)
    assert not offenders, offenders


def test_open_items_from_history_semantics():
    from data_loader.history import open_items_from_history
    items = pd.DataFrame({"item_id": ["A", "B", "C", "D"], "total": [100.0, 100.0, 100.0, 100.0]})
    g = pd.DataFrame({"item_id": ["A", "B", "B", "C", "D", ""],
                      "amount_f": [40.0, 100.0, 100.0, 100.0, 100.0, float("nan")],
                      "exception": ["PARTIAL", "", "RETURNED", "", "", "UNIDENTIFIED"]})
    g = pd.concat([g, pd.DataFrame({"item_id": ["C"], "amount_f": [100.0], "exception": ["DUPLICATE_PAYMENT"]})])
    o = open_items_from_history(items, g).set_index("item_id")["open_amount"].to_dict()
    assert o == {"A": 60.0, "B": 100.0, "C": 0.0, "D": 0.0}


def _with_lock(fn, wait=float(os.environ.get("BANKREC_LOCK_WAIT", "120"))):
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            if time.time() - t0 > wait:
                skip("dev lock busy")
            time.sleep(5)
    try:
        return fn()
    finally:
        os.rmdir(LOCK)


def test_dev_context_has_no_window_golden():
    start, end = WINDOW
    from data_loader.history import CTX_VERSION
    cache = os.path.join(ROOT, "outputs", "cache", f"ctx_{CTX_VERSION}_train_{start}_{end}.pkl")
    if not os.path.exists(cache) and not os.environ.get("BANKREC_HEAVY"):
        skip(f"no cached context for {start}..{end} (set BANKREC_HEAVY=1 to build it, ~7 GB RAM)")
    from data_loader.history import build_context
    ctx = _with_lock(lambda: build_context("train", start, end))
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    h = ctx["golden_hist"]
    assert h["booking_date"].max() < s
    assert not h["booking_date"].isna().any()
    assert ctx["bank"]["booking_date"].between(s, e).all()
    assert not set(h["bank_txn_id"]) & set(ctx["bank"]["bank_txn_id"])
    assert (ctx["items"]["issue_date"] <= e).all()
    assert (ctx["items"]["open_amount"] > 0).all()
