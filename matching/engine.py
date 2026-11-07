"""Reconciliation engine: one chronological pass over bank lines with stateful open balances.

Per line (first hit wins):
  1. treasury / non-item category with high confidence (exceptions.categories, vectorised up-front)
  2. RETURNED (anomaly_detection.returns)
  3. file-based: AP payment runs | DD batches, cheques, card settlements
  4. remittance advices
  5. party + amount (AR receipts / AP debits)
  6. DUPLICATE_PAYMENT (anomaly_detection.duplicates)
  7. UNIDENTIFIED resolver: P(unidentified) model (anomaly_detection.unidentified) -> relaxed recovery
     (matching.recovery) when the payer is identifiable, else UNIDENTIFIED
  8. fallback category (receipts -> UNIDENTIFIED, debits -> classifier guess)
"""
import time

import pandas as pd

from anomaly_detection.duplicates import DuplicateDetector
from anomaly_detection.returns import is_return, match_return
from exceptions.categories import classify
from exceptions.categories import postprocess
from .files import FileMatchers
from .learned import learn
from .party import match_ap, match_ar
from .recovery import recover
from .remittance import Remittance
from anomaly_detection.unidentified import UnidentifiedModel, UNID_THRESHOLD
from .state import State


def run(ctx, log=print):
    t0 = time.time()
    S = State(ctx)
    log(f"[engine] state: {len(S.I):,} items ({time.time() - t0:.1f}s)")
    FM = FileMatchers(S)
    RA = Remittance(S)
    DUP = DuplicateDetector(S)
    learn(S, ctx)
    UNID = UnidentifiedModel(S, ctx)
    log(f"[engine] files + {len(RA.txt):,} remittance advices loaded ({time.time() - t0:.1f}s)")

    bank = ctx["bank"].copy()
    bank["booking_date"] = bank["booking_date"].dt.strftime("%Y-%m-%d")
    for c in ["text", "ref1", "ref2", "remittance_info", "end_to_end_id", "counterparty_name", "counterparty_account",
              "orig_amount", "orig_currency", "bank_code", "bank_subcode"]:
        bank[c] = bank[c].fillna("")
    bank["T"] = (bank["text"] + " " + bank["text_join"] + " " + bank["ref1"] + " " + bank["ref2"] + " "
                 + bank["remittance_info"] + " " + bank["end_to_end_id"] + " " + bank["counterparty_name"]).str.upper()
    bank["search"] = bank["T"]
    cats = classify(bank)
    bank["cat"], bank["cat_final"] = cats["category"], cats["final"]
    bank = bank.sort_values(["booking_date", "bank_txn_id"])

    for t in bank.itertuples(index=False):
        T = t.T
        credit = t.amount_f > 0
        if t.cat_final:
            # supplier refund of a credit note ("REEMBOLSO FRA") is an item, not a category
            S.category(t, t.cat, "", "category_rule", 0.9)
            continue
        if is_return(t, T):
            if match_return(t, T, S, FM):
                continue
        if not credit:
            if DUP.match(t, T):   # before payment runs: re-executed references would otherwise grab it (agent 3)
                continue
            if FM.match_payment_run(t, T, S) or match_ap(t, T, S):
                DUP.record(t, T)
                continue
            S.category(t, t.cat, "", "category_fallback", 0.3)
            continue
        if FM.match_dd(t, T, S) or FM.match_cheque(t, T, S) or FM.match_card(t, T, S) or RA.match(t, T, S):
            continue
        if DUP.match(t, T):       # before party matching: near-amount rules would otherwise grab duplicates
            continue
        if match_ar(t, T, S) or match_ap(t, T, S):
            DUP.record(t, T)
            continue
        if t.cat == "UNIDENTIFIED":
            # UNIDENTIFIED resolver: only receipts whose P(truly unidentified) is low go to relaxed recovery
            p_unid = UNID.prob(t, T)
            if recover(t, T, S, p_unid):   # strong rules always; weak rules gated by p_unid inside recover()
                DUP.record(t, T)
                continue
            S.category(t, "UNIDENTIFIED", "UNIDENTIFIED", "unidentified", 0.5, f"p_unid={p_unid:.2f}")
        else:
            S.category(t, t.cat, "", "category_fallback", 0.3)
    postprocess(S, bank)
    log(f"[engine] matched {len(bank):,} lines ({time.time() - t0:.1f}s)")
    S.DUP = DUP
    return S, bank
