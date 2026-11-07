"""Remittance advices: index customer payment references -> advice; allocate the documents listed in the advice.

Layouts (TRAIN 2021-2025): csv (',' or ';': document,doc_date,gross,discount_or_deduction,reason,paid,your_order),
xlsx (inlineStr / sharedStrings cells: Document, Date, Gross, Deduction, Reason, Paid), txt e-mail
("FV25-0310834   24/04/25   1,648.97  -16.49 DISCOUNT"), pdf (one cell per line; empty cells omitted).
Every layout is turned into rows of cells; a row starts with a document number (ours, ours without prefix, or the
customer's PO) and carries gross / deduction / reason / paid.

Row -> allocation (POLICIES §2-3): paid == gross -> full; reason DISCOUNT -> SHORT_DISCOUNT (within terms, full open)
else PARTIAL (paid); reason RET (BR withholding) -> SHORT_WITHHOLDING (full); other reason -> SHORT_DISPUTE (paid);
no reason and paid < gross -> PARTIAL (paid). Credit notes: full amount, no exception. Currency mismatch -> FX_DIFFERENCE.

Bank line -> advice: payment reference token present in both (advice may arrive after the cash), else
(entity, Σ paid == cash) with the advice payer among the line's candidate parties, within [-20, +45] days.
"""
import datetime as dt
import os
import re
import zipfile
from collections import defaultdict

import pandas as pd

from data_loader.loader import _read, extract_pdf_text
from .text import cents, fmt, nz

TOK = r"[A-Z0-9][A-Z0-9\-/]{6,}[0-9]"
DATE_RX = re.compile(r"^(\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|\d{4}-\d{2}-\d{2})$")
NUM_RX = re.compile(r"^-?\(?[\d.,]*\d\)?$")
DISC_RX = re.compile(r"DISC|DTO|DESC|SKONTO|ESCOMPTE|SCONTO|PRONTO|RABAT|PP\b")
WHT_RX = re.compile(r"^RET\b|RETEN|WITHHOLD|ISR|IRRF|WHT")


def _num(s):
    s = s.strip().replace(" ", "")
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("-()")
    if not s or not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _xlsx_rows(path):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        x = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
        shared = [re.sub(r"<[^>]+>", "", si) for si in re.findall(r"<si>(.*?)</si>", x, re.S)]
    rows = []
    for n in sorted(z.namelist()):
        if not n.startswith("xl/worksheets/sheet"):
            continue
        x = z.read(n).decode("utf-8", "ignore")
        for row in re.findall(r"<row[^>]*>(.*?)</row>", x, re.S):
            cells = {}
            for m in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', row, re.S):
                col, attrs, body = m.group(1), m.group(2), m.group(3) or ""
                if 't="s"' in attrs:
                    v = re.search(r"<v>(.*?)</v>", body)
                    val = shared[int(v.group(1))] if v else ""
                elif "inlineStr" in attrs:
                    val = re.sub(r"<[^>]+>", "", body)
                else:
                    v = re.search(r"<v>(.*?)</v>", body)
                    val = v.group(1) if v else ""
                cells[col] = val.strip()
            if cells:
                idx = sorted(cells, key=lambda c: (len(c), c))
                width = max(ord(c[-1]) - 64 for c in idx) if all(len(c) == 1 for c in idx) else len(idx)
                rows.append([cells.get(chr(65 + i), "") for i in range(width)] if all(len(c) == 1 for c in idx)
                            else [cells[c] for c in idx])
    return rows


def read_rows(path, fmt_):
    """-> (list of rows (list of cell str, upper-cased), full text upper)"""
    if fmt_ == "xlsx":
        rows = [[c.upper() for c in r] for r in _xlsx_rows(path)]
        return rows, " ".join(" ".join(r) for r in rows)
    if fmt_ == "csv":
        s = open(path, encoding="utf-8", errors="ignore").read().upper()
        lines = s.splitlines()
        hdr = next((l for l in lines if "DOCUMENT" in l or "INVOICE" in l), lines[1] if len(lines) > 1 else "")
        delim = ";" if hdr.count(";") > hdr.count(",") else ","
        return [l.split(delim) for l in lines], s
    if fmt_ == "txt":
        s = open(path, encoding="utf-8", errors="ignore").read().upper()
        rows = []
        for l in s.splitlines():
            parts = [p for p in re.split(r"\s{2,}|\t", l.strip()) if p]
            out = []
            for p in parts:   # "-16.49 DISCOUNT" -> number + reason
                m = re.match(r"^(-?[\d.,]+\d)\s+(\D.*)$", p)
                out += [m.group(1), m.group(2)] if m else [p]
            rows.append(out)
        return rows, s
    s = extract_pdf_text(path).upper()
    return [[l.strip()] for l in s.splitlines() if l.strip()], s


class Remittance:
    def __init__(self, S):
        self.S = S
        self.ref = defaultdict(set)
        self.txt = {}
        self.adv = {}            # file -> dict(ent, rows, text, received, payer)
        self.by_total = defaultdict(list)
        self.used = set()
        rdir = os.path.join(S.data_dir, "remittance_advices")
        idx_p = os.path.join(rdir, "index.csv")
        if not os.path.exists(idx_p):
            return
        idx = _read(idx_p)
        if S.phase == "train" and S.start is not None:
            lo = (S.start - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
            hi = (S.end + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
            idx = idx[(idx["received_date"] >= lo) & (idx["received_date"] <= hi)]
        for r in idx.itertuples(index=False):
            p = os.path.join(rdir, r.file)
            try:
                rows, s = read_rows(p, r.format)
            except Exception:
                continue
            self.txt[r.file] = (r.entity, s, r.received_date)
            self.adv[r.file] = dict(ent=r.entity, rows=rows, fmt=r.format, rec=r.received_date, payer=r.payer_name,
                                    parsed=None)
            for tok in set(re.findall(TOK, s)):
                if sum(ch.isdigit() for ch in tok) >= 6:
                    self.ref[tok].add(r.file)
            parsed = self.parse(r.file)
            tot = sum(-p if k else p for _, p, _, _, k in parsed)
            if parsed:
                self.by_total[(r.entity, tot)].append(r.file)

    # ------------------------------------------------------------------ parsing
    def parties(self, f):
        a = self.adv[f]
        ps = set(self.S.CX.query(a["ent"], a["payer"].upper())) if a["payer"] else set()
        for p in list(ps):
            ps |= self.S.children.get(p, set())
        return ps

    def resolve(self, ent, tok, parties):
        S, I = self.S, self.S.I
        k = nz(tok)
        if len(k) < 4 or not re.search(r"\d", k):
            return None
        c = [i for i in S.full.get((ent, k), []) if I[i]["k"] in ("ARI", "ARC")]
        if c:
            c2 = [i for i in c if I[i]["open"] > 0] or c
            c3 = [i for i in c2 if I[i]["p"] in parties] or c2
            return c3[0]
        dd = re.sub(r"\D", "", tok)
        if len(dd) >= 5 and len(dd) == len(k):          # our number without prefix
            c = [i for i in S.suf.get((ent, dd[-6:].rjust(6, "0")), []) if I[i]["open"] > 0]
            c = [i for i in c if I[i]["p"] in parties] if parties else c
            if len(c) == 1:
                return c[0]
        c = [i for i in S.po.get((ent, k), []) if I[i]["open"] > 0]       # customer's PO / order number
        c = [i for i in c if I[i]["p"] in parties] or c
        if len(c) == 1:
            return c[0]
        return None

    def parse(self, f):
        """-> list of (item, paid_cents(abs), gross_cents(abs), reason, is_credit)"""
        a = self.adv[f]
        if a["parsed"] is not None:
            return a["parsed"]
        ent, rows = a["ent"], a["rows"]
        parties = self.parties(f)
        S, I = self.S, self.S.I
        out, seen = [], set()
        hdr = None
        if a["fmt"] in ("csv", "xlsx"):
            for r in rows:
                cols = [c.strip() for c in r]
                if any(c.startswith("DOC") or c.startswith("INVOICE") or c == "FACTURA" for c in cols) and \
                        any("PAID" in c or "PAGADO" in c or "NET" in c for c in cols):
                    hdr = cols
                    break
        if hdr:
            def col(*names):
                for j, c in enumerate(hdr):
                    if any(c.startswith(n) for n in names):
                        return j
                return None
            jd, jg, jx, jr, jp = col("DOC", "INVOICE", "FACTURA"), col("GROSS", "BRUTO", "AMOUNT"), \
                col("DISCOUNT", "DEDUC", "DTO"), col("REASON", "MOTIVO"), col("PAID", "PAGADO", "NET")
            jo = col("YOUR", "ORDER", "PO")
            started = False
            for r in rows:
                cols = [c.strip() for c in r]
                if cols == hdr:
                    started = True
                    continue
                if not started or jd is None or jd >= len(cols):
                    continue
                it = self.resolve(ent, cols[jd], parties)
                if it is None and jo is not None and jo < len(cols) and cols[jo]:
                    it = self.resolve(ent, cols[jo], parties)
                if it is None or it in seen:
                    continue
                g = _num(cols[jg]) if jg is not None and jg < len(cols) else None
                p = _num(cols[jp]) if jp is not None and jp < len(cols) else None
                x = _num(cols[jx]) if jx is not None and jx < len(cols) and cols[jx] else None
                reason = cols[jr] if jr is not None and jr < len(cols) else ""
                if g is None:
                    g = p
                if p is None:
                    p = g + (x if x and x < 0 else -(x or 0)) if g is not None else None
                seen.add(it)
                out.append(self._row(it, g, p, reason))
        else:
            cur, cells = None, []

            def flush():
                if cur is None:
                    return
                nums = [_num(c) for c in cells if NUM_RX.match(c) and not DATE_RX.match(c)]
                nums = [n for n in nums if n is not None]
                reason = " ".join(c for c in cells if re.search(r"[A-Z]{2,}", c) and not NUM_RX.match(c))
                g = nums[0] if nums else None
                if len(nums) >= 3:
                    p = nums[-1]
                elif len(nums) == 2:
                    p = g + nums[1] if (nums[1] < 0 < g) else nums[1]
                else:
                    p = g
                out.append(self._row(cur, g, p, reason))
            for r in rows:
                for c in r:
                    c = c.strip()
                    if not c:
                        continue
                    it = None
                    if re.search(r"\d{3}", c) and not DATE_RX.match(c) and not NUM_RX.match(c.replace(" ", "")) or \
                            re.fullmatch(r"\d{5,12}", c):
                        it = self.resolve(ent, c, parties)
                    if it is not None and it not in seen:
                        flush()
                        cur, cells = it, []
                        seen.add(it)
                    elif cur is not None:
                        if re.search(r"TOTAL|KIND REGARDS|SALUDOS|CORDIALEMENT|DOCUMENTO SINT|MIT FREUNDLICHEN", c):
                            flush()
                            cur, cells = None, []
                        else:
                            cells.append(c)
            flush()
        a["parsed"] = out
        return out

    def _row(self, it, g, p, reason):
        v = self.S.I[it]
        credit = v["k"] == "ARC"
        g_c = abs(cents(g)) if g is not None else v["open"]
        p_c = abs(cents(p)) if p is not None else g_c
        return (it, p_c, g_c, reason.strip(), credit)

    # ------------------------------------------------------------------ allocation
    def rows_for(self, t, f):
        S, I = self.S, self.S.I
        rows = []
        for it, paid, gross, reason, credit in self.parse(f):
            v = I[it]
            full = v["open"] if v["open"] > 0 else v["tot"]
            if credit:
                rows.append((it, fmt(full), "", ""))
                continue
            exc, amt = "", full
            short = paid < min(gross, full) - 1
            if short:
                if DISC_RX.search(reason):
                    if self._disc_ok(v, t.booking_date):
                        exc = "SHORT_DISCOUNT"
                    else:
                        exc, amt = "PARTIAL", paid
                elif WHT_RX.search(reason):
                    exc = "SHORT_WITHHOLDING"
                elif reason:
                    exc, amt = "SHORT_DISPUTE", paid
                elif self._dispute_ratio(v, paid, gross):
                    # agent 4: golden disputes deduct exactly 3/5/10/20/30 % of the invoice (reason often missing)
                    exc, amt = "SHORT_DISPUTE", paid
                else:
                    exc, amt = "PARTIAL", paid
            if v["cur"] != t.currency and exc in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING"):
                exc = "FX_DIFFERENCE"
            rows.append((it, fmt(amt), "", exc))
        return rows

    def _dispute_ratio(self, v, paid, gross):
        S = self.S
        base = v["tot"] or gross
        if not base:
            return False
        ded = base - paid
        code = v["terms"]
        pct = 0.0
        if code and code in S.terms.index:
            try:
                pct = float(S.terms.at[code, "early_payment_discount_pct"] or 0)
            except (TypeError, ValueError):
                pct = 0.0
        wht = S.cust_country.get(v["p"], "") in ("MX", "BR", "MA")
        for r in (0.03, 0.05, 0.10, 0.20, 0.30):
            if wht and r == 0.10:
                continue      # 10 % withholding (MX/BR/MA) looks like a 10 % dispute
            if abs(ded - r * base) <= 1.01 and abs(r * 100 - pct) > 1e-6:
                return True
        return False

    def _disc_ok(self, v, d):
        S = self.S
        code = v["terms"]
        if not code or code not in S.terms.index:
            return True
        days = S.terms.at[code, "discount_days"]
        if not days:
            return True
        lim = (dt.date.fromisoformat(v["iss"]) + dt.timedelta(days=int(float(days)) + 3)).isoformat()
        return d <= lim

    def _ok_date(self, f, d):
        rec = self.adv[f]["rec"]
        try:
            delta = (dt.date.fromisoformat(rec) - dt.date.fromisoformat(d)).days
        except ValueError:
            return True
        return -25 <= delta <= 45

    def match(self, t, T, S):
        cand = []
        for tok in re.findall(TOK, T):
            fs = self.ref.get(tok)
            if fs and len(fs) <= 3 and not re.fullmatch(r"(FV|FA|RE|INV|FT|NF|VTN|A01)[\d\-]+", tok):
                cand += [f for f in fs if f not in self.used and self.adv[f]["ent"] == t.entity and self._ok_date(f, t.booking_date)]
        A = abs(cents(t.amount))
        if not cand:
            fs = [f for f in self.by_total.get((t.entity, A), []) if f not in self.used and self._ok_date(f, t.booking_date)]
            if fs:
                tp = S.CX.query(t.entity, T)
                fs2 = [f for f in fs if self.parties(f) & tp]
                cand = fs2 or (fs if len(fs) == 1 and t.currency else [])
        if not cand:
            return False
        cand = sorted(set(cand))
        cand.sort(key=lambda f: abs((dt.date.fromisoformat(self.adv[f]["rec"]) - dt.date.fromisoformat(t.booking_date)).days))
        f = cand[0]
        rows = self.rows_for(t, f)
        if not rows:
            return False
        self.used.add(f)
        S.take(t, rows, "remittance_advice", 0.85, f)
        return True
