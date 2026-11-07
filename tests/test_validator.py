"""validate.py must reject exactly what the server rejects (crafted bad files on the sample phase)."""
import gzip
import os
import subprocess
import sys

from _util import HEADER, ROOT, TMP, phase_txn_ids, sample_submission

T = None


def _t(i=0):
    global T
    if T is None:
        T = phase_txn_ids("sample")
    return T[i]


def _validate(name, content, binary=False):
    p = os.path.join(TMP, name)
    with open(p, "wb") as f:
        f.write(content if binary else content.encode("utf-8"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate.py"), p, "--phase", "sample"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _rejects(name, content, msg, binary=False):
    rc, out = _validate(name, content, binary)
    assert rc == 1, f"{name} should be rejected:\n{out}"
    assert msg in out, f"{name}: expected '{msg}' in:\n{out}"


def test_accepts_pipeline_output():
    rc, out = _validate("ok.csv.gz", open(sample_submission(), "rb").read(), binary=True)
    assert rc == 0, out
    assert "missing_bank_line" not in out


def test_rejects_unknown_txn():
    _rejects("bad_txn.csv", HEADER + "BT-1999-0000001,,,BANK_FEES,\n", "is not a bank transaction of this phase")


def test_rejects_mixed_item_and_category():
    _rejects("mixed.csv", HEADER + f"{_t()},ARI-2025-0000001,10.00,,\n{_t()},,,BANK_FEES,\n", "both item rows and a no-item")


def test_rejects_duplicate_pair():
    _rejects("dup.csv", HEADER + f"{_t()},ARI-2025-0000001,10.00,,\n{_t()},ARI-2025-0000001,10.00,,\n", "duplicate row")


def test_rejects_two_noitem_rows():
    _rejects("dup_noitem.csv", HEADER + f"{_t()},,,BANK_FEES,\n{_t()},,,PAYROLL,\n", "duplicate no-item row")


def test_rejects_negative_amount():
    _rejects("neg.csv", HEADER + f"{_t()},ARI-2025-0000001,-10.00,,\n", "invalid amount")


def test_rejects_comma_decimal_with_comma_delimiter():
    _rejects("commadec.csv", HEADER + f'{_t()},ARI-2025-0000001,"10,00",,\n', "invalid amount")


def test_rejects_unknown_category_and_exception():
    _rejects("cat.csv", HEADER + f"{_t()},,,FEES,\n", "unknown category")
    _rejects("exc.csv", HEADER + f"{_t()},,,BANK_FEES,SHORT\n", "unknown exception")


def test_rejects_bad_item_id_and_category_on_item_row():
    _rejects("item.csv", HEADER + f"{_t()},ari-2025-0000001,1.00,,\n", "is not a valid id")
    _rejects("itemcat.csv", HEADER + f"{_t()},ARI-2025-0000001,1.00,BANK_FEES,\n", "category must be empty")


def test_rejects_amount_on_noitem_row_and_extra_column():
    _rejects("amt.csv", HEADER + f"{_t()},,5.00,BANK_FEES,\n", "amount must be empty")
    _rejects("extra.csv", "bank_txn_id,item_id,amount,category,exception,note\n" + f"{_t()},,,BANK_FEES,,x\n",
             "unexpected column")


def test_rejects_fake_gzip_name():
    _rejects("plain.csv.gz", HEADER + f"{_t()},,,BANK_FEES,\n", "not gzip")


def test_accepts_semicolon_comma_decimals_bom_crlf():
    body = "﻿bank_txn_id;item_id;amount;category;exception\r\n" + f" {_t()} ;ARI-2025-0000001;10,50;;\r\n"
    rc, out = _validate("semi.csv", body)
    assert rc == 0, out
    assert "missing_bank_line" in out  # only one line covered -> warning, not an error


def test_gzip_roundtrip_accepted():
    rc, out = _validate("g.csv.gz", gzip.compress((HEADER + f"{_t()},,,BANK_FEES,\n").encode()), binary=True)
    assert rc == 0, out
