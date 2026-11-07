"""Shared helpers for the test-suite (works under pytest or plain `python3 tests/run_tests.py`)."""
import csv
import functools
import gzip
import io
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BANKREC_DATA", os.path.join(os.path.dirname(ROOT), "data"))
TMP = os.environ.get("BANKREC_TEST_TMP", os.path.join(tempfile.gettempdir(), "bankrec_tests"))
LOCK = "/tmp/bankrec_dev.lock"
for p in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(DATA, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(TMP, exist_ok=True)

EXC = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
       "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]
HEADER = "bank_txn_id,item_id,amount,category,exception\n"


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def xfail(reason):
    """Known bug: test documents desired behaviour; failure is expected (XFAIL), a pass is reported as XPASS."""
    def deco(fn):
        try:
            import pytest
            return pytest.mark.xfail(reason=reason, strict=False)(fn)
        except ImportError:
            fn._xfail = reason
            return fn
    return deco


def read_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def phase_txn_ids(phase):
    with gzip.open(os.path.join(DATA, phase, "bank_transactions.csv.gz"), "rt", encoding="utf-8") as fh:
        return [r["bank_txn_id"] for r in csv.DictReader(fh)]


def run_main_sample(out, seed="0"):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--phase", "sample", "--out", out],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    return out


@functools.lru_cache(maxsize=None)
def sample_submission(seed="0"):
    """Run the pipeline on the sample phase once per session (~6 s)."""
    return run_main_sample(os.path.join(TMP, f"sample_seed{seed}.csv.gz"), seed)


@functools.lru_cache(maxsize=None)
def sample_golden():
    """Official Golden object for the sample phase (same construction as starter/score_train.py)."""
    import bankrec_score as S
    from evaluate import load_categories
    tp = os.path.join(TMP, "sample_txns.csv")
    with open(tp, "w") as f:
        f.write("bank_txn_id\n" + "\n".join(phase_txn_ids("sample")) + "\n")
    return S.load_golden(os.path.join(DATA, "sample", "golden_allocations.csv.gz"), tp, EXC, load_categories(DATA))


def golden_as_submission_bytes(phase="sample"):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(HEADER.strip().split(","))
    with gzip.open(os.path.join(DATA, phase, "golden_allocations.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            w.writerow([r["bank_txn_id"], r["item_id"], r["amount"], r["category"], r["exception"]])
    return out.getvalue().encode()
