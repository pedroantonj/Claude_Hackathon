"""Unit tests for exceptions/short_payments.py against POLICIES §3 (synthetic State)."""
import os
from types import SimpleNamespace

import pandas as pd

from _util import DATA
from data_loader.loader import _read
from exceptions.short_payments import explain_short
from matching.state import State

M = os.path.join(DATA, "masters")


class FakeS:
    fx_rate = State.fx_rate
    to_cur = State.to_cur
    sgn = staticmethod(State.sgn)

    def __init__(self, items, cust_country=None, fx=None):
        self.I = items
        self.terms = _read(os.path.join(M, "payment_terms.csv")).set_index("code")
        self.entities = _read(os.path.join(M, "entities.csv")).set_index("entity")
        self.cust_country = cust_country or {}
        self.fx = {c: pd.Series(r) for c, r in (fx or {}).items()}
        self.ctx = {"golden_hist": None}      # no history -> only low withholding rates (< 5 %) allowed
        self.phase, self.start, self.end = "sample", None, None


def item(k="ARI", tot=100000, open_=None, cur="EUR", iss="2025-06-01", terms="T30", p="C1"):
    return dict(k=k, e="ES01", p=p, pn="X", dn="D", iss=iss, due=iss, cur=cur, tot=tot,
                open=tot if open_ is None else open_, open0=tot, terms=terms, po="", rel="", pm="")


def line(date="2025-06-05", cur="EUR", orig_amount="", orig_currency="", entity="ES01"):
    return SimpleNamespace(booking_date=date, currency=cur, orig_amount=orig_amount, orig_currency=orig_currency,
                           entity=entity)


def run(S, A, t, refs=(), T=""):
    return explain_short(t, T, A, list(S.I), list(refs), S)


# ---------------------------------------------------------------- SHORT_DISCOUNT
def test_discount_within_terms_plus_grace():
    S = FakeS({"I1": item(terms="D2N30")})                       # 2 % within 10 days
    rows, rule = run(S, 98000, line("2025-06-14"))                # 1 Jun + 10 + 3 grace
    assert rows == [("I1", "1000.00", "", "SHORT_DISCOUNT")], rule


def test_discount_after_grace_is_partial_with_cash_amount():
    S = FakeS({"I1": item(terms="D2N30")})
    rows, rule = run(S, 98000, line("2025-06-15"))
    assert rows == [("I1", "980.00", "", "PARTIAL")]


def test_discount_half_cent_rounding_both_ways():
    # 2 % of 125.25 = 2.505 -> cash 122.74 or 122.75 depending on rounding convention: both accepted
    for A in (12274, 12275):
        S = FakeS({"I1": item(tot=12525, terms="D2N30")})
        rows, _ = run(S, A, line("2025-06-03"))
        assert rows[0][3] == "SHORT_DISCOUNT" and rows[0][1] == "125.25"


def test_discount_needs_terms_with_discount_and_untouched_item():
    assert run(FakeS({"I1": item(terms="T30")}), 98000, line()) is None or \
        run(FakeS({"I1": item(terms="T30")}), 98000, line())[0][0][3] != "SHORT_DISCOUNT"
    S = FakeS({"I1": item(terms="D2N30", open_=50000)})
    res = run(S, 98000, line())
    assert res is None or res[0][0][3] != "SHORT_DISCOUNT"


# ---------------------------------------------------------------- SHORT_WITHHOLDING
def test_withholding_mx_customer_full_amount():
    S = FakeS({"I1": item(tot=100000)}, cust_country={"C1": "MX"})
    rows, rule = run(S, 98750, line())                            # MX: 1.25 %
    assert rows == [("I1", "1000.00", "", "SHORT_WITHHOLDING")]
    S = FakeS({"I1": item(tot=100000)}, cust_country={"C1": "BR"})
    rows, rule = run(S, 98500, line())                            # BR: 1.5 %
    assert rows == [("I1", "1000.00", "", "SHORT_WITHHOLDING")]


def test_withholding_10pct_only_for_known_withholders():
    S = FakeS({"I1": item(tot=100000)}, cust_country={"C1": "MX"})   # no history -> 10 % (retention look-alike) refused
    res = run(S, 90000, line())
    assert res is None or res[0][0][3] != "SHORT_WITHHOLDING"


def test_no_withholding_for_spanish_customer():
    S = FakeS({"I1": item(tot=100000)}, cust_country={"C1": "ES"})
    res = run(S, 90000, line())                                   # 10 % short, ES customer, ES entity
    assert res is None or res[0][0][3] != "SHORT_WITHHOLDING"


# ---------------------------------------------------------------- FX_DIFFERENCE
FX = {"USD": {"2025-06-02": 1.10, "2025-06-04": 1.12}}


def test_fx_orig_amount_single_full_amount_item_currency():
    S = FakeS({"I1": item(tot=100000, cur="USD")}, fx=FX)
    rows, rule = run(S, 89000, line(orig_amount="1000.00", orig_currency="USD"))
    assert rows == [("I1", "1000.00", "", "FX_DIFFERENCE")] and rule.startswith("fx_orig"), rule


def test_fx_subset_credit_note_row_has_no_exception():
    S = FakeS({"I1": item(tot=100000, cur="USD"), "C1": item(k="ARC", tot=20000, cur="USD"),
               "I2": item(tot=33333, cur="USD")}, fx=FX)
    rows, rule = run(S, 71000, line(orig_amount="800.00", orig_currency="USD"))
    assert rule.startswith("fx_orig"), rule
    assert sorted(rows) == [("C1", "200.00", "", ""), ("I1", "1000.00", "", "FX_DIFFERENCE")]


def test_fx_reference_rate_last_available_and_tolerance():
    S = FakeS({"I1": item(tot=110000, cur="USD")}, fx=FX)
    # weekend 2025-06-03 uses 2025-06-02 rate 1.10 -> 1000 EUR; bank 995 EUR (0.5 % spread) -> FX
    rows, rule = run(S, 99500, line("2025-06-03"))
    assert rows == [("I1", "1100.00", "", "FX_DIFFERENCE")] and rule.startswith("fx_rate"), (rows, rule)
    res = run(FakeS({"I1": item(tot=110000, cur="USD")}, fx=FX), 90000, line("2025-06-03"))   # 10 % off
    assert res is None or res[0][0][3] != "FX_DIFFERENCE"


# ---------------------------------------------------------------- SHORT_BANK_CHARGES
def test_bank_charges_non_sepa_small_shortfall():
    S = FakeS({"I1": item(tot=100000, cur="USD")}, cust_country={"C1": "US"}, fx=FX)
    # cross-border signal required (85 % of golden SBC lines read WIRE IN / EN DIV / GASTOS SHA; domestic ACH excluded)
    rows, _ = run(S, 98500, line(cur="USD", entity="US01"), T="WIRE IN /ORG/LAKEVIEW POOLS /OBI/")   # 15 USD short (1.5 %)
    assert rows == [("I1", "1000.00", "", "SHORT_BANK_CHARGES")]


def test_no_bank_charges_for_domestic_ach_without_wire_signal():
    S = FakeS({"I1": item(tot=100000, cur="USD")}, cust_country={"C1": "US"}, fx=FX)
    res = run(S, 98500, line(cur="USD", entity="US01"), T="ACH CREDIT CCD LAKEVIEW POOLS")
    assert res is None or res[0][0][3] != "SHORT_BANK_CHARGES"


def test_bank_charges_rejected_above_threshold():
    # merged rule: <= 60 EUR-eq and <= 3.3 % (golden max 3.15 %); 4 % short must not be bank charges
    S = FakeS({"I1": item(tot=100000, cur="USD")}, cust_country={"C1": "US"}, fx=FX)
    res = run(S, 96000, line(cur="USD", entity="US01"))
    assert res is None or res[0][0][3] != "SHORT_BANK_CHARGES"


def test_no_bank_charges_for_domestic_sepa_eur():
    S = FakeS({"I1": item(tot=100000)}, cust_country={"C1": "ES"})
    res = run(S, 99000, line())                                   # 10 EUR short, ES customer, EUR SEPA
    assert res is None or res[0][0][3] != "SHORT_BANK_CHARGES"
