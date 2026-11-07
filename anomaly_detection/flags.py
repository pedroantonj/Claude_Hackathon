"""Non-scored audit flags (agent 3).  audit_flags(S, bank) -> DataFrame

Columns: flag, severity (high/medium/low), bank_txn_id, booking_date, account_id, amount, currency, party, detail.
Flags:
  DUPLICATE_PAYMENT      line flagged as a duplicate (+ whether a refund was seen)        high
  DUPLICATE_REFUND       refund paired structurally with a duplicate                       low
  UNIDENTIFIED_AGED      unidentified receipt, age (days to window end) > 30               medium (>90: high)
  UNIDENTIFIED_CASH      unidentified cash deposit                                         medium
  RETURN_REASON          returned collection/payment with its reason code                  low (MD01/MD06/AC04/AC06/
                                                                                           SL01/MS02/MS03: medium)
  BANK_ERROR_PAIR        "ANULACION OPERACION INDEBIDA" + the wrong original (days open)   medium
  BANK_ERROR_UNPAIRED    a reversal without its original in the window                     high
  IBAN_UNSEEN            camt counterparty IBAN not in masters and not announced           low
  IBAN_CHANGE_NOTICE     bank-account change notice in pdfs/ (new/old IBAN, effective)     low
  AMOUNT_OUTLIER         receipt > 10x the payer's median (payers with >= 6 receipts)       low (soft)
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from data_loader.loader import MASTERS, _read
from matching.text import nz

RISKY_REASONS = {"MD01", "MD06", "AC04", "AC06", "SL01", "MS02", "MS03", "AC01", "BE04"}
REASON = re.compile(r"\b(AC\d\d|AG\d\d|AM\d\d|BE\d\d|MD\d\d|MS\d\d|RC\d\d|RR\d\d|SL\d\d|FF\d\d|R\d\d|NSF|"
                    r"REFER TO MAKER|SIN FONDOS|STOP PAYMENT)\b")
ANUL = re.compile(r"^ANULACION OPERACION INDEBIDA N [\d |]+")
IBAN = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b")


def _pdf_text(path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, here)
    try:
        from data_loader.loader import extract_pdf_text  # noqa
        return extract_pdf_text(path)
    except Exception:
        import base64
        import zlib
        data = open(path, "rb").read()
        out = []
        for m in re.finditer(rb"/Filter\s*\[?([^\]>]*)\]?[^>]*>>\s*stream\r?\n", data):
            f = m.group(1)
            s = data[m.end():data.find(b"endstream", m.end())].strip()
            try:
                if b"ASCII85" in f:
                    s = base64.a85decode(s[:-2] if s.endswith(b"~>") else s)
                if b"Flate" in f:
                    s = zlib.decompress(s)
            except Exception:
                continue
            out += [t.decode("latin-1") for t in re.findall(rb"\(((?:\\.|[^\\)])*)\)\s*Tj", s)]
        return "\n".join(out)


def iban_notices(S):
    """Parse bank_account_notice PDFs of the phase -> DataFrame(party, doc_date, effective, new, old, kind)."""
    rows = []
    idx = os.path.join(S.data_dir, "pdfs", "index.csv")
    if not os.path.exists(idx):
        return pd.DataFrame(columns=["party", "doc_date", "effective", "new", "old", "kind", "file"])
    ix = _read(idx)
    ix = ix[ix.document_type == "bank_account_notice"]
    for f, pid, dd in zip(ix.file, ix.party_id, ix.document_date):
        p = os.path.join(S.data_dir, "pdfs", f)
        if not os.path.exists(p):
            continue
        t = _pdf_text(p).replace("\n", " ")
        ib = IBAN.findall(t)
        ds = re.findall(r"\d\d/\d\d/\d{4}", t)
        kind = "dd_mandate" if re.search(r"domicili|direct debit|mandat", t, re.I) else "payer_account"
        rows.append((pid, dd, ds[1] if len(ds) > 1 else "", ib[0] if ib else "", ib[1] if len(ib) > 1 else "", kind, f))
    return pd.DataFrame(rows, columns=["party", "doc_date", "effective", "new", "old", "kind", "file"])


def audit_flags(S, bank, DUP=None):
    out = []
    b = bank.copy()
    b["T"] = b["text"].fillna("").str.upper()
    end = pd.Timestamp(b["booking_date"].max())

    def add(flag, sev, r, party="", detail=""):
        out.append((flag, sev, r.bank_txn_id, r.booking_date, r.account_id, r.amount, r.currency, party, detail))

    res = {t: S.out.get(t, []) for t in b.bank_txn_id}
    # --- duplicates / refunds / returns / unidentified (from the engine output)
    refunds = {tid for tid, rows in res.items() if rows and not rows[0][0] and rows[0][2] in
               ("CUSTOMER_REFUND", "SUPPLIER_REFUND")}
    for r in b.itertuples(index=False):
        rows = res.get(r.bank_txn_id) or []
        excs = {x[3] for x in rows}
        if "DUPLICATE_PAYMENT" in excs:
            it = rows[0][0]
            party = S.I[it]["pn"] if it in S.I else ""
            add("DUPLICATE_PAYMENT", "high", r, party, f"item {it}; {S.why.get(r.bank_txn_id, ('', 0, ''))[2]}")
        if "RETURNED" in excs:
            m = REASON.search(r.T)
            code = m.group(1) if m else "?"
            add("RETURN_REASON", "medium" if code in RISKY_REASONS else "low", r, "",
                f"reason {code}; items {','.join(x[0] for x in rows if x[0])}")
        if rows and not rows[0][0] and rows[0][2] == "UNIDENTIFIED":
            age = (end - pd.Timestamp(r.booking_date)).days
            if re.search(r"ESPECES|INGRESO CAJERO|INGRESO EFECTIVO|CASH DEPOSIT", r.T):
                add("UNIDENTIFIED_CASH", "medium", r, "", "cash deposit without payer")
            if age > 30:
                add("UNIDENTIFIED_AGED", "high" if age > 90 else "medium", r, r.counterparty_name or "",
                    f"age {age}d at window end")
        if r.bank_txn_id in refunds:
            add("DUPLICATE_REFUND", "low", r, "", rows[0][2])
    # --- bank error pairs
    an = b[b["T"].str.match(ANUL)]
    for r in an.itertuples(index=False):
        suf = ANUL.sub("", r.T).strip()
        a = -float(r.amount)
        c = b[(b.account_id == r.account_id) & (np.isclose(b.amount.astype(float), a)) &
              (b["T"].str.strip() == suf) & (b.booking_date <= r.booking_date)]
        if len(c):
            o = c.iloc[-1]
            days = (pd.Timestamp(r.booking_date) - pd.Timestamp(o.booking_date)).days
            add("BANK_ERROR_PAIR", "medium", r, "", f"reverses {o.bank_txn_id} ({suf}) after {days}d")
        else:
            add("BANK_ERROR_UNPAIRED", "high", r, "", f"original '{suf}' not found in window")
    # --- IBANs
    known = set(S.iban_party)
    notices = iban_notices(S)
    announced = set(notices.new) | set(notices.old)
    for r in notices.itertuples(index=False):
        out.append(("IBAN_CHANGE_NOTICE", "low", "", r.doc_date, "", "", "", r.party,
                    f"{r.kind}: new {r.new} (old {r.old}) effective {r.effective}; keep both valid"))
    cp = b[(b.counterparty_account.fillna("") != "")]
    seen = set()
    for r in cp.itertuples(index=False):
        k = nz(r.counterparty_account)
        if k in known or r.counterparty_account in announced or k in seen:
            continue
        seen.add(k)
        add("IBAN_UNSEEN", "low", r, r.counterparty_name, f"first use of {r.counterparty_account}")
    # --- amount outliers (soft)
    cr = b[(b.amount.astype(float) > 0) & (b.counterparty_name.fillna("") != "")].copy()
    if len(cr):
        cr["la"] = np.log10(cr.amount.astype(float))
        g = cr.groupby("counterparty_name")["la"]
        cr["med"], cr["n"] = g.transform("median"), g.transform("size")
        for r in cr[(cr.n >= 6) & (cr.la - cr.med > 1)].itertuples(index=False):
            add("AMOUNT_OUTLIER", "low", r, r.counterparty_name, f"{10 ** (r.la - r.med):.0f}x payer median")
    return pd.DataFrame(out, columns=["flag", "severity", "bank_txn_id", "booking_date", "account_id", "amount",
                                      "currency", "party", "detail"])
