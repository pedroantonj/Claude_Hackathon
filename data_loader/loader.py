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

# default: a `data/` folder next to the repo (…/Claude Hackathon/data); override with BANKREC_DATA
DATA = os.environ.get("BANKREC_DATA", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")))
BASE_MASTERS = os.path.join(DATA, "masters")
MASTERS = os.environ.get("BANKREC_MASTERS", BASE_MASTERS)   # re-pointed per phase by set_phase_masters()
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_HERE, "outputs", "cache")
PREV_PHASES = {"live": ["train"], "final": ["train", "live"], "train": [], "sample": []}
# low-cardinality columns -> category (less RAM; itertuples then yields shared str objects)
_CAT_ITEM = ["kind", "entity", "currency", "payment_terms", "payment_method", "source", "side", "party_id"]


def _p(phase, *parts):
    return os.path.join(DATA, phase, *parts)


def _read(path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)


def _num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _date(s):
    return pd.to_datetime(s.replace("", np.nan), format="%Y-%m-%d", errors="coerce")


# --------------------------------------------------------------------------- masters (phase-aware)
def masters_dir(phase):
    """Directory with the masters to use for `phase`.

    final: final/masters_final/ when it exists (files missing there fall back to masters/).
    Every phase: bank_accounts.csv / entities.csv are augmented with accounts/entities that appear in the
    phase bank lines but not in the masters (e.g. ZA01-UMVC-ZAR shows up in LIVE but masters lack it).
    The merged copy lives in outputs/cache/masters_<phase>/ (rebuilt on every call: cheap)."""
    src = [BASE_MASTERS]
    mf = _p(phase, "masters_final")
    if phase == "final" and os.path.isdir(mf):
        src.insert(0, mf)
    bank_path = _p(phase, "bank_transactions.csv.gz")
    if not os.path.exists(bank_path):
        return src[0]
    out = os.path.join(CACHE, f"masters_{phase}")
    os.makedirs(out, exist_ok=True)
    names = sorted({f for d in src for f in os.listdir(d) if f.endswith(".csv")})
    pick = {f: next(os.path.join(d, f) for d in src if os.path.exists(os.path.join(d, f))) for f in names}
    bt = _read(bank_path, usecols=["account_id", "entity", "bank", "currency", "statement_format"]).drop_duplicates("account_id")
    for f, path in pick.items():
        dst = os.path.join(out, f)
        if f == "bank_accounts.csv":
            acc = _read(path)
            miss = bt[~bt["account_id"].isin(acc["account_id"])]
            if len(miss):
                add = pd.DataFrame({c: [""] * len(miss) for c in acc.columns})
                for c in ["account_id", "entity", "bank", "currency", "statement_format"]:
                    if c in add:
                        add[c] = miss[c].values
                if "bank_name" in add:
                    add["bank_name"] = miss["bank"].values
                acc = pd.concat([acc, add], ignore_index=True)
            acc.to_csv(dst + ".tmp", index=False)
        elif f == "entities.csv":
            ent = _read(path)
            miss = bt[~bt["entity"].isin(ent["entity"])].drop_duplicates("entity")
            if len(miss):
                add = pd.DataFrame({c: [""] * len(miss) for c in ent.columns})
                add["entity"] = miss["entity"].values
                if "country" in add:
                    add["country"] = miss["entity"].str[:2].values
                if "functional_currency" in add:
                    add["functional_currency"] = miss["currency"].values
                ent = pd.concat([ent, add], ignore_index=True)
            ent.to_csv(dst + ".tmp", index=False)
        else:
            import shutil
            shutil.copyfile(path, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    return out


def set_phase_masters(phase):
    """Point MASTERS (here and in every already-imported module that did `from ...loader import MASTERS`)
    to the masters of `phase`. Call before building State."""
    import sys
    global MASTERS
    new = os.environ.get("BANKREC_MASTERS") or masters_dir(phase)
    old = MASTERS
    MASTERS = new
    for m in list(sys.modules.values()):
        try:
            if getattr(m, "MASTERS", None) in (old, BASE_MASTERS) and m.__name__ != __name__:
                m.MASTERS = new
        except Exception:
            pass
    return new


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
    # engine rebuilds its own search string (T); keep the column for API compatibility but cheap
    df["search"] = ""
    return df


# --------------------------------------------------------------------------- items
_ITEM_COLS = ["item_id", "kind", "entity", "party_id", "party_name", "doc_number", "issue_date",
              "due_date", "currency", "total", "open_amount", "net_amount", "tax_amount",
              "payment_terms", "payment_method", "po", "related_document", "reason", "source"]
_ENRICH = ["party_name", "payment_terms", "payment_method", "po", "related_document", "reason",
           "net_amount", "tax_amount"]

_AR_MAP = {"customer_id": "party_id", "customer_name": "party_name", "document_number": "doc_number",
           "customer_po": "po", "total_amount": "total"}
_AP_MAP = {"supplier_id": "party_id", "supplier_name": "party_name", "supplier_invoice_number": "doc_number",
           "our_po": "po", "total_amount": "total"}
_CN_MAP = {"document_number": "doc_number", "total_amount": "total"}


def _party_names():
    c = _read(os.path.join(MASTERS, "customers.csv"), usecols=["customer_id", "name"])   # phase-aware (final)
    s = _read(os.path.join(MASTERS, "suppliers.csv"), usecols=["supplier_id", "name"])
    return pd.concat([c.set_index("customer_id")["name"], s.set_index("supplier_id")["name"]])


def _period_items(phase, keep=None, usecols=None):
    """Items issued in `phase` in unified columns. keep(df)->bool mask is applied per file (memory)."""
    frames = []
    for fname, mp, kind in [("ar_invoices", _AR_MAP, "ARI"), ("ap_invoices", _AP_MAP, "API"),
                            ("credit_notes", _CN_MAP, None)]:
        path = _p(phase, f"{fname}.csv.gz")
        if not os.path.exists(path):
            continue
        df = _read(path).rename(columns=mp)
        if kind:
            df["kind"] = kind
        else:
            df["kind"] = np.where(df["side"] == "AR", "ARC", "APC")
            df["due_date"] = df["issue_date"]
            df = df.drop(columns=["side"])
        if keep is not None:
            df = df[keep(df)]
        if usecols is not None:
            df = df[[c for c in usecols if c in df.columns]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["source"] = "period"
    if "total" in df:
        df["open_amount"] = df["total"]
    return df


def _enrich_open_items(oi, phase):
    """open_items_at_start lacks printed name / terms / PO / method / related doc: take them from the item
    files of the previous phases (train for live; train+live for final), by item_id."""
    ids = set(oi["item_id"])
    parts = []
    for prev in PREV_PHASES.get(phase, []):
        if not os.path.exists(_p(prev, "ar_invoices.csv.gz")):
            continue
        parts.append(_period_items(prev, keep=lambda d: d["item_id"].isin(ids), usecols=["item_id"] + _ENRICH))
    if not parts:
        return oi
    src = pd.concat(parts, ignore_index=True).drop_duplicates("item_id", keep="last").set_index("item_id")
    for c in _ENRICH:
        if c not in src:
            continue
        v = oi["item_id"].map(src[c])
        if c in oi:
            oi[c] = v.where(v.notna() & (v != ""), oi[c])
        else:
            oi[c] = v
    return oi


def _finish_items(df):
    for c in _ITEM_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[_ITEM_COLS].copy()
    df = df.fillna("")
    for c in ["total", "open_amount", "net_amount", "tax_amount"]:
        df[c] = _num(df[c].astype(str).replace("nan", ""))
    df["issue_date"] = _date(df["issue_date"])
    df["due_date"] = _date(df["due_date"])
    df["side"] = df["kind"].str[:2]          # AR / AP
    df["is_credit"] = df["kind"].str[2] == "C"
    df = df.drop(columns=["net_amount", "tax_amount", "reason"])   # unused downstream
    for c in _CAT_ITEM:
        df[c] = df[c].astype("category")
    return df


def load_items(phase, include_period=True, keep=None):
    """Unified item table. LIVE/FINAL = open_items_at_start (open_amount given; name/terms/PO/method/
    related_document enriched from previous-phase item files, fallback name = masters current name)
    + items issued in the phase (open_amount = total).
    TRAIN/sample = the phase item files (open_amount = total, i.e. no history applied).
    keep: optional per-file row filter for period items (used by train dev windows)."""
    parts = []
    oi_path = _p(phase, "open_items_at_start.csv.gz")
    if os.path.exists(oi_path):
        oi = _read(oi_path).rename(columns={"item_type": "kind", "document_number": "doc_number",
                                             "total_amount": "total"})
        oi = _enrich_open_items(oi, phase)
        names = oi["party_id"].map(_party_names())
        oi["party_name"] = oi.get("party_name", pd.Series("", index=oi.index)).fillna("")
        oi["party_name"] = oi["party_name"].where(oi["party_name"] != "", names.fillna(""))
        oi["source"] = "open_at_start"
        parts.append(oi)
    if include_period:
        parts.append(_period_items(phase, keep=keep))
    return _finish_items(pd.concat(parts, ignore_index=True))


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
