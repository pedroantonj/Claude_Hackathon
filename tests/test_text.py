"""Unit tests for matching/text.py helpers."""
from _util import ROOT  # noqa: F401  (sys.path)
from matching.text import PrefixIndex, cents, fmt, mitm, nz, pnum


def test_pnum_formats():
    assert pnum("1.234,56") == 1234.56
    assert pnum("1,234.56") == 1234.56
    assert pnum("1234,56") == 1234.56
    assert pnum("1234.56") == 1234.56
    assert pnum("-1.234.567,89") == -1234567.89
    assert pnum("12.345.678,00") == 12345678.0


def test_cents_fmt_roundtrip():
    for s in ("0.01", "1210.00", "99999999.99", "0.29", "4.35", "1.10"):
        assert fmt(cents(s)) == s
    assert cents("1.005") in (100, 101)   # float half-cent: either side is within scorer tolerance
    assert cents("-12.34") == -1234


def test_nz_folds_accents_and_punctuation():
    assert nz("Pooltechnik Weißenau GmbH & Co. KG")[:10] == "POOLTECHNI"
    assert nz("Ibérica, S.A.U.") == "IBERICASAU"


def test_mitm_unique_and_ambiguous():
    cnt, sol = mitm([100, 250, 400, 1000], 650)
    assert cnt == 1 and sorted(sol) == [1, 2]
    cnt, sol = mitm([100, 100, 300], 400)      # two solutions (either 100)
    assert cnt == 2
    cnt, sol = mitm([100, 200], 50)
    assert cnt == 0 and sol is None


def test_mitm_credit_notes_negative_and_limits():
    cnt, sol = mitm([1000, -200, 500], 800)    # invoice - credit note
    assert cnt == 1 and sorted(sol) == [0, 1]
    assert mitm([], 0) == (0, None)
    assert mitm([1] * 23, 5) == (0, None)        # n > maxn guard
    assert mitm([5, -5], 0)[0] == 1              # non-empty subset summing to 0 only


def test_mitm_large_uses_both_halves():
    vals = [10 ** i for i in range(12)]
    target = 10 ** 0 + 10 ** 7 + 10 ** 11
    cnt, sol = mitm(vals, target)
    assert cnt == 1 and sorted(sol) == [0, 7, 11]


def test_prefix_index_truncation_and_entity():
    px = PrefixIndex([("ES01", "C1", "Gartenbau Márquez GmbH"), ("ES01", "C2", "Gartenbau Martinez SL"),
                      ("DE01", "C3", "Gartenbau Márquez GmbH"), ("ES01", "C4", "ABC")])
    assert px.query("ES01", "TRANSFERENCIA DE GARTENBAU MARQUEZ GM") == {"C1"}       # truncated name
    assert px.query("DE01", "SEPA GARTENBAU MARQUEZ GMBH") == {"C3"}                 # entity isolation
    assert px.query("FR01", "GARTENBAU MARQUEZ GMBH") == set()                       # unknown entity
    assert px.query("ES01", "ABC SA") == set()                                       # short names not indexed
    s, l = px.query("ES01", "PAGO GARTENBAU", return_len=True)   # shared prefix < 60 % of both names
    assert l == 9 and s == set()
