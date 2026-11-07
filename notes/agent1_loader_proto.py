"""Agent 1 prototype loaders (stdlib + pandas only).

Usage:
    from agent1_loader_proto import load_bank, load_items, load_batches, extract_pdf_text, pdf_rows
    bt = load_bank("live"); items = load_items("live"); b = load_batches("live")
phase in {"sample", "train", "live", "final"}.
"""
import base64
import glob
import os
import re
import time
import zlib

import numpy as np
import pandas as pd

DATA = os.environ.get("BANKREC_DATA", "/Users/pedroantonperez/Desktop/Claude Hackathon/data")
MASTERS = os.path.join(DATA, "masters")


def _p(phase, *parts):
    return os.path.join(DATA, phase, *parts)


def _read(path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)


def _num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _date(s):
    return pd.to_datetime(s.replace("", np.nan), format="%Y-%m-%d", errors="coerce")


# --------------------------------------------------------------------------- bank
def load_bank(phase):
    """Normalised bank lines. Adds amount_f (float), dates, and `text_join`:
    N43 text with the ' | ' 38-char field separators removed (tokens such as run ids
    are split across the boundary, e.g. 'PR0101251 | 11010')."""
    df = _read(_p(phase, "bank_transactions.csv.gz"))
    df["amount_f"] = _num(df["amount"])
    df["orig_amount_f"] = _num(df["orig_amount"])
    df["booking_date"] = _date(df["booking_date"])
    df["value_date"] = _date(df["value_date"])
    df["text_join"] = df["text"].str.replace(" | ", "", regex=False)
    df["search"] = (df["ref1"] + " " + df["ref2"] + " " + df["text_join"] + " "
                    + df["end_to_end_id"] + " " + df["remittance_info"])
    return df


# --------------------------------------------------------------------------- items
_ITEM_COLS = ["item_id", "kind", "entity", "party_id", "party_name", "doc_number", "issue_date",
              "due_date", "currency", "total", "open_amount", "net_amount", "tax_amount",
              "payment_terms", "payment_method", "po", "related_document", "reason", "source"]


def _party_names():
    c = _read(os.path.join(MASTERS, "customers.csv"), usecols=["customer_id", "name"])
    s = _read(os.path.join(MASTERS, "suppliers.csv"), usecols=["supplier_id", "name"])
    return pd.concat([c.set_index("customer_id")["name"], s.set_index("supplier_id")["name"]])


def _period_items(phase):
    ar = _read(_p(phase, "ar_invoices.csv.gz"))
    ar = ar.rename(columns={"customer_id": "party_id", "customer_name": "party_name",
                            "document_number": "doc_number", "customer_po": "po",
                            "total_amount": "total"})
    ar["kind"] = "ARI"
    ap = _read(_p(phase, "ap_invoices.csv.gz"))
    ap = ap.rename(columns={"supplier_id": "party_id", "supplier_name": "party_name",
                            "supplier_invoice_number": "doc_number", "our_po": "po",
                            "total_amount": "total"})
    ap["kind"] = "API"
    cn = _read(_p(phase, "credit_notes.csv.gz"))
    cn = cn.rename(columns={"document_number": "doc_number", "total_amount": "total"})
    cn["kind"] = np.where(cn["side"] == "AR", "ARC", "APC")
    cn["due_date"] = cn["issue_date"]
    df = pd.concat([ar, ap, cn.drop(columns=["side"])], ignore_index=True)
    df["source"] = "period"
    df["open_amount"] = df["total"]
    return df


def load_items(phase, include_period=True):
    """Unified item table. LIVE/FINAL = open_items_at_start (open_amount given; party_name
    from masters = CURRENT name) + items issued in the phase (open_amount = total).
    TRAIN/sample = the phase item files (open_amount = total, i.e. no history applied)."""
    parts = []
    oi_path = _p(phase, "open_items_at_start.csv.gz")
    if os.path.exists(oi_path):
        oi = _read(oi_path).rename(columns={"item_type": "kind", "document_number": "doc_number",
                                             "total_amount": "total"})
        oi["party_name"] = oi["party_id"].map(_party_names()).fillna("")
        oi["source"] = "open_at_start"
        parts.append(oi)
    if include_period:
        parts.append(_period_items(phase))
    df = pd.concat(parts, ignore_index=True)
    for c in _ITEM_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[_ITEM_COLS].copy()
    df = df.fillna("")
    for c in ["total", "open_amount", "net_amount", "tax_amount"]:
        df[c] = _num(df[c])
    df["issue_date"] = _date(df["issue_date"])
    df["due_date"] = _date(df["due_date"])
    df["side"] = df["kind"].str[:2]          # AR / AP
    df["is_credit"] = df["kind"].str[2] == "C"
    return df


# --------------------------------------------------------------------------- batches
_BATCH_NUM = {
    "direct_debit_batches": ["batch_total", "amount"],
    "card_settlements": ["gross", "fee", "net"],
    "cheque_deposits": ["deposit_total", "amount"],
    "payment_runs": ["transfer_amount", "document_amount"],
}
_BATCH_DATE = {
    "direct_debit_batches": ["collection_date"],
    "card_settlements": ["settlement_date", "operation_date"],
    "cheque_deposits": ["deposit_date"],
    "payment_runs": ["execution_date"],
}


def load_batches(phase, kinds=None):
    """dict kind -> DataFrame (all monthly files concatenated, `file` column added)."""
    out = {}
    for kind in kinds or _BATCH_NUM:
        files = sorted(glob.glob(_p(phase, kind, "*.csv")))
        frames = [_read(f).assign(file=os.path.basename(f)) for f in files]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for c in _BATCH_NUM[kind]:
            if c in df:
                df[c] = _num(df[c])
        for c in _BATCH_DATE[kind]:
            if c in df:
                df[c] = _date(df[c])
        out[kind] = df
    return out


# --------------------------------------------------------------------------- PDFs
_ESC = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
_STREAM_RE = re.compile(rb"<<(.*?)>>\s*stream\r?\n(.*?)endstream", re.S)
_STR_RE = r"\((?:\\.|[^\\)])*\)"
_OP_RE = re.compile(r"([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) Tm|"
                    r"([-\d.]+) ([-\d.]+) Td|(" + _STR_RE + r")\s*Tj|\[((?:" + _STR_RE + r"|[^\]])*)\]\s*TJ", re.S)


def _unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n in _ESC:
                out.append(_ESC[n]); i += 2; continue
            m = re.match(r"[0-7]{1,3}", s[i + 1:])
            if m:
                out.append(chr(int(m.group(), 8))); i += 1 + len(m.group()); continue
            out.append(n); i += 2; continue
        out.append(c); i += 1
    return "".join(out)


def _fix(s):
    try:
        return s.encode("latin-1").decode("cp1252", errors="replace")   # WinAnsiEncoding
    except UnicodeEncodeError:
        return s


def _content_streams(data):
    for m in _STREAM_RE.finditer(data):
        d, raw = m.group(1), m.group(2)
        if b"/Subtype /Image" in d:
            continue
        try:
            if b"ASCII85Decode" in d:
                raw = raw.strip()
                if raw.endswith(b"~>"):
                    raw = raw[:-2]
                raw = base64.a85decode(raw.replace(b"\n", b"").replace(b"\r", b""))
            if b"FlateDecode" in d:
                raw = zlib.decompress(raw)
        except (ValueError, zlib.error):
            continue
        s = raw.decode("latin-1")
        if "BT" in s:
            yield s


def pdf_spans(path):
    """[(x, y, text)] for every text show op (ReportLab PDFs: one Tm + Tj per cell)."""
    spans = []
    for s in _content_streams(open(path, "rb").read()):
        x = y = 0.0
        for m in _OP_RE.finditer(s):
            if m.group(1) is not None:
                x, y = float(m.group(5)), float(m.group(6))
            elif m.group(7) is not None:
                x += float(m.group(7)); y += float(m.group(8))
            elif m.group(9) is not None:
                spans.append((x, y, _fix(_unescape(m.group(9)[1:-1]))))
            elif m.group(10) is not None:
                t = "".join(_unescape(z[1:-1]) for z in re.findall(_STR_RE, m.group(10)))
                spans.append((x, y, _fix(t)))
    return spans


def pdf_rows(path, ytol=1.0):
    """Group spans into visual rows (top->bottom), each a list of (x, text) left->right.
    Use for remittance-advice tables: empty Deduction cells are simply absent, so map
    cells to columns by x position relative to the header row."""
    rows = []
    for x, y, t in sorted(pdf_spans(path), key=lambda r: (-r[1], r[0])):
        if rows and abs(rows[-1][0] - y) <= ytol:
            rows[-1][1].append((x, t))
        else:
            rows.append((y, [(x, t)]))
    return [sorted(r) for _, r in rows]


def extract_pdf_text(path):
    """Plain text, one line per text op, in stream order."""
    return "\n".join(t for _, _, t in pdf_spans(path))


# --------------------------------------------------------------------------- timing
if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "live"
    for name, fn in [("load_bank", load_bank), ("load_items", load_items), ("load_batches", load_batches)]:
        t = time.time(); r = fn(phase); dt = time.time() - t
        desc = {k: len(v) for k, v in r.items()} if isinstance(r, dict) else (len(r), r.shape[1])
        print(f"{name}({phase}): {dt:.1f}s {desc}")
    pdfs = sorted(glob.glob(_p(phase, "pdfs", "*", "*.pdf")))
    t = time.time(); n = sum(len(extract_pdf_text(f)) > 0 for f in pdfs); dt = time.time() - t
    print(f"extract_pdf_text: {len(pdfs)} pdfs, {n} non-empty, {dt:.1f}s")
