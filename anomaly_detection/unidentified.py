"""P(truly UNIDENTIFIED) for receipts the matcher could not allocate (owner: agent 3).

UnidentifiedModel(S, ctx).prob(t, T) -> float in [0, 1].
Logistic regression (numpy, L2, Newton) fitted ONLY on ctx['golden_hist']: credit lines booked in the FIT_DAYS before
the window/phase (label = golden category UNIDENTIFIED vs line settling items; other categories dropped).
Features (bank line + masters only):
  payer-name evidence : PrefixIndex LCP vs entity customer names, party found, company suffix, person-name pattern,
                        initials ("B.H."), first-name dictionary (person customers in masters)
  account evidence    : counterparty IBAN present / known in masters / IBAN quoted in text
  wording             : cash deposit, ANTICIPO, generic remittance only, invoice-like reference, 4+ digit token
  amount              : round 100 / round 10, log10 amount
  format              : n43 / mt940 / camt053
State factor (outside the regression): party identified but holding no open AR item -> odds x OPEN0_ODDS
(measured: no ranking gain on 2024-11..12, kept at 1.0).
Measured on the lines the merged pipeline predicted UNIDENTIFIED (g_int outputs, LITE data):
  2024-11..12 (tune):    n=3175, truly UNID 1440, AUC 0.971; thr 0.13 -> UNID recall 0.976, 1363 lines to recovery
                         of which 97.4 % truly have items
  2025-05..06 (confirm): n=4046, truly UNID 1677, AUC 0.972; thr 0.13 -> UNID recall 0.979, 1875 to recovery (98.1 %)
prob() ~0.1 ms/line.
"""
import math
import os
import pickle
import re

import numpy as np
import pandas as pd

# The regression is fitted on all credit lines (UNIDENTIFIED prior ~4 %); prob() is evaluated on lines the matcher
# could not allocate (prior ~45 %): PRIOR_SHIFT moves the intercept (log-odds) to that population.
# Threshold tuned on TRAIN 2024-11..12 for true-UNIDENTIFIED recall >= 0.97 (raw p 0.0075 == shifted 0.13).
PRIOR_SHIFT = 3.0
UNID_THRESHOLD = 0.13
FIT_DAYS = 270
OPEN0_ODDS = 1.0

_SUFFIX = re.compile(r"\b(?:S\.?L\.?U?|S\.?A\.?U?|SARL|SAS|EURL|GMBH|LTD|LLC|INC|CORP|CO\.|PTY|B\.V\.|BV|S\.?P\.?A|"
                     r"S\.?R\.?L|LDA|S\.?COOP|Z O\.?O|S\.?N\.?C|KG|E\.K\.|LTDA|GROUP|C\.V\.|LIMITED|COMPANY|SA DE)\b")
_INIT = re.compile(r"(?:^|[\s/|:-])((?:[A-Z]\.(?:&\.)?){2,4})(?=[\s/|\\]|$)")
_NOTINIT = {"S.A.", "S.L.", "B.V.", "N.V.", "E.K.", "S.A.S.", "S.P.A.", "S.R.L.", "C.V.", "S.L.U.", "S.N.C.", "S.C."}
_CASH = re.compile(r"ESPECES|INGRESO CAJERO|INGRESO EFECTIVO|CASH DEPOSIT|BAREINZAHLUNG|CONTANTI|GOTOWK")
_ANT = re.compile(r"ANTICIPO|ADVANCE|ACOMPTE|ANZAHLUNG")
_INVREF = re.compile(r"\b(?:INV|FV|FA|FAC|FT|RE|FRA|FACT|FACTURA|INVOICE|NF|VTN|RECIBO)[ .:#-]*\d{3,}|\b\d{2,4}[/-]\d{3,}")
_DIG = re.compile(r"\b\d{4,}\b")
_BOILER = re.compile(r"PAYMENT ID:\d+|HPNB\d+|WATT\d+|RHEI\d+|NOTPROVIDED|NONREF|[A-Z]{2}\d{12,}|\b\d{8}-")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{12,30}\b")
_GENERIC = {"PAYMENT", "ORDER", "INVOICE", "TRANSFER", "FACTURA", "PAGO", "ENVIO", "REF", "TRANSFERENCIA"}

FEATS = ["bias", "lcp", "lcp_ge8", "lcp_ge12", "party", "suffix", "initials", "firstname", "cp_iban", "iban_known",
         "iban_text", "cash", "anticipo", "generic", "invref", "digits", "round100", "round10", "logamt", "n43",
         "mt940", "camt"]


def _first_names(S):
    fn = set()
    for n in S.cust["name"]:
        u = str(n).upper()
        if not _SUFFIX.search(u):
            w = re.findall(r"[A-Z]{3,}", u)
            if 2 <= len(w) <= 4:
                fn.add(w[0])
    return fn


class UnidentifiedModel:
    def __init__(self, S, ctx):
        self.S = S
        self.fnames = _first_names(S)
        self.w = None
        try:
            self._fit(ctx)
        except Exception as e:  # never break the pipeline: fall back to "keep UNIDENTIFIED"
            print(f"[unid] fit failed: {e!r}")
            self.w = None

    # ------------------------------------------------------------------ features
    def feats(self, entity, T, text, cp_name, cp_acc, amount, fmt):
        S = self.S
        parties, lcp = S.CX.query(entity, T, return_len=True)
        u = text.upper() + " " + cp_name.upper()
        inits = [m for m in _INIT.findall(u) if m not in _NOTINIT]
        words = set(re.findall(r"[A-Z]{3,}", u))
        acc = cp_acc.replace(" ", "")
        ib_txt = _IBAN.findall(T)
        known = bool(acc and acc in S.iban_party) or any(i in S.iban_party for i in ib_txt)
        Tn = _BOILER.sub(" ", T)
        rem = re.sub(r"[^A-Z ]", " ", T).split()
        a = abs(amount)
        x = [1.0, min(lcp, 30) / 10.0, float(lcp >= 8), float(lcp >= 12), float(bool(parties)),
             float(bool(_SUFFIX.search(u))), float(bool(inits)), float(bool(words & self.fnames)),
             float(bool(acc)), float(known), float(bool(ib_txt)), float(bool(_CASH.search(T))),
             float(bool(_ANT.search(T)) and "PROFORMA" not in T), float(bool(rem) and rem[-1] in _GENERIC),
             float(bool(_INVREF.search(Tn))), float(bool(_DIG.search(Tn))), float(round(a * 100) % 10000 == 0),
             float(round(a * 100) % 1000 == 0), math.log10(a + 1.0) / 3.0,
             float(fmt == "n43"), float(fmt == "mt940"), float(fmt == "camt053")]
        return x, parties

    # ------------------------------------------------------------------ fit
    def _fit(self, ctx):
        gh = ctx.get("golden_hist")
        if gh is None or not len(gh) or "booking_date" not in gh:
            return
        S = self.S
        lo = pd.Timestamp(ctx["bank"]["booking_date"].min())
        from data_loader.history import CACHE
        key = os.path.join(CACHE, f"unid_w_{S.phase}_{lo.date()}_{FIT_DAYS}.pkl")
        if os.path.exists(key):
            with open(key, "rb") as fh:
                self.w, self.mu, self.sd = pickle.load(fh)
            return
        g = gh[(gh["booking_date"] >= lo - pd.Timedelta(days=FIT_DAYS)) & (gh["booking_date"] < lo)]
        lab = g.groupby("bank_txn_id").agg(cat=("category", "first"), item=("item_id", "first"))
        lab = lab[(lab["cat"] == "UNIDENTIFIED") | (lab["item"] != "")]
        from data_loader.loader import _read, _p
        b = _read(_p("train", "bank_transactions.csv.gz"),
                  usecols=["bank_txn_id", "entity", "statement_format", "amount", "text", "ref1", "ref2",
                           "remittance_info", "end_to_end_id", "counterparty_name", "counterparty_account"])
        b = b[b.bank_txn_id.isin(lab.index)]
        b = b[~b.amount.str.startswith("-")]
        b = b.join(lab, on="bank_txn_id")
        y = (b["cat"] == "UNIDENTIFIED").values
        neg = np.where(~y)[0]
        nneg = min(len(neg), 80000)
        rng = np.random.RandomState(0)
        keep = np.concatenate([np.where(y)[0], rng.choice(neg, nneg, replace=False)])
        b = b.iloc[keep]
        y = y[keep].astype(float)
        tj = b.text.str.replace(" | ", "", regex=False)
        T = (b.text + " " + tj + " " + b.ref1 + " " + b.ref2 + " " + b.remittance_info + " " + b.end_to_end_id + " "
             + b.counterparty_name).str.upper()
        X = np.array([self.feats(e, t_, tx, cn, ca, float(a), f)[0] for e, t_, tx, cn, ca, a, f in
                      zip(b.entity, T, b.text, b.counterparty_name, b.counterparty_account, b.amount,
                          b.statement_format)])
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        self.mu[0], self.sd[0] = 0.0, 1.0
        Z = (X - self.mu) / self.sd
        sw = np.where(y > 0, 1.0, len(neg) / max(1, nneg))   # re-weight the negative subsample
        w = np.zeros(Z.shape[1])
        lam = 1.0
        for _ in range(30):   # Newton-Raphson
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            gr = Z.T @ (sw * (p - y)) + lam * w
            H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(w))
            step = np.linalg.solve(H, gr)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        try:
            with open(key, "wb") as fh:
                pickle.dump((self.w, self.mu, self.sd), fh)
        except OSError:
            pass

    # ------------------------------------------------------------------ predict
    def logit(self, entity, T, text, cp_name, cp_acc, amount, fmt):
        x, parties = self.feats(entity, T, text, cp_name, cp_acc, amount, fmt)
        return float(((np.asarray(x) - self.mu) / self.sd) @ self.w), parties

    def open_ar(self, parties):
        I = self.S.I
        return sum(1 for p in parties for i in self.S.byparty.get(p, ()) if I[i]["k"] == "ARI" and I[i]["open"] > 0)

    def prob(self, t, T):
        if self.w is None:
            return 1.0
        z, parties = self.logit(t.entity, T, t.text, t.counterparty_name, t.counterparty_account, t.amount_f,
                                t.statement_format)
        z += PRIOR_SHIFT
        if OPEN0_ODDS != 1.0 and parties and self.open_ar(parties) == 0:
            z += math.log(OPEN0_ODDS)
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
