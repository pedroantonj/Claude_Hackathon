"""Submission writer invariants on the real pipeline output (sample phase) + unit test of main.write_submission."""
import csv
import gzip
import os
import re
from collections import defaultdict
from types import SimpleNamespace

from _util import DATA, EXC, TMP, phase_txn_ids, read_rows, sample_submission

AMT = re.compile(r"^\d+\.\d{2}$")


def _cats():
    return {r["category"] for r in csv.DictReader(open(os.path.join(DATA, "masters", "categories.csv"), encoding="utf-8"))}


def _by_txn(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["bank_txn_id"]].append(r)
    return d


def test_header_and_coverage():
    sub = sample_submission()
    with gzip.open(sub, "rt", encoding="utf-8") as fh:
        assert fh.readline().strip() == "bank_txn_id,item_id,amount,category,exception"
    d = _by_txn(read_rows(sub))
    ids = phase_txn_ids("sample")
    assert set(d) == set(ids), "every bank line exactly covered, nothing extra"


def test_row_level_invariants():
    cats = _cats()
    bad = []
    for r in read_rows(sample_submission()):
        it, a, c, e = r["item_id"], r["amount"], r["category"], r["exception"]
        assert e in EXC or e == "", r
        if it:
            if not (re.fullmatch(r"(ARI|ARC|API|APC)-\d{4}-\d{7}", it) and AMT.match(a) and c == ""):
                bad.append(r)
            if e == "UNIDENTIFIED":
                bad.append(r)
            if e == "FX_DIFFERENCE" and it[2] == "C":
                bad.append(r)  # POLICIES §3: credit-note rows stay empty
        else:
            if a or c not in cats or e not in ("", "UNIDENTIFIED") or ((c == "UNIDENTIFIED") != (e == "UNIDENTIFIED")):
                bad.append(r)
    assert not bad, bad[:5]


def test_line_level_invariants():
    for t, rows in _by_txn(read_rows(sample_submission())).items():
        items = [r for r in rows if r["item_id"]]
        assert not (items and len(items) != len(rows)), f"{t} mixes item and category rows"
        assert items or len(rows) == 1, f"{t} has {len(rows)} no-item rows"
        assert len({r["item_id"] for r in items}) == len(items), f"{t} duplicate item"


def test_amounts_positive():
    zero = [r for r in read_rows(sample_submission()) if r["item_id"] and float(r["amount"]) <= 0]
    assert not zero, zero[:5]


def test_content_deterministic_across_hash_seeds():
    a = sorted(map(tuple, (r.values() for r in read_rows(sample_submission("0")))))
    b = sorted(map(tuple, (r.values() for r in read_rows(sample_submission("12345")))))
    assert a == b


def test_bytes_deterministic_across_hash_seeds():
    assert gzip.decompress(open(sample_submission("0"), "rb").read()) == \
        gzip.decompress(open(sample_submission("12345"), "rb").read())


def test_write_submission_unit():
    from main import write_submission
    S = SimpleNamespace(out={
        "BT-1": [("ARI-2025-0000001", "10.00", "", ""), ("", "", "BANK_FEES", "")],   # mixed -> items only
        "BT-2": [("", "", "PAYROLL", ""), ("", "", "BANK_FEES", "")],                 # 2 no-item -> first
        # BT-3 missing -> default UNIDENTIFIED
    })
    bank = {"bank_txn_id": ["BT-3", "BT-1", "BT-2"]}
    p = os.path.join(TMP, "unit_write.csv")
    n = write_submission(S, bank, p)
    rows = [tuple(r.values()) for r in read_rows(p)]
    assert n == 3
    assert rows == [("BT-1", "ARI-2025-0000001", "10.00", "", ""), ("BT-2", "", "", "PAYROLL", ""),
                    ("BT-3", "", "", "UNIDENTIFIED", "UNIDENTIFIED")]
