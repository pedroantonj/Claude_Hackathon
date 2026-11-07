"""Agent2 prototype matcher (Phase A). Deterministic keys first, then party+amount.

    python3 agent2_proto.py --data <data root> --phase sample --out /tmp/a2.csv.gz

Matchers (in order): AP payment runs, DD batches, card settlements, cheque deposits, remittance advices,
AR party(prefix name / IBAN) + refs + exact amount / subset-sum / FIFO / discount. Everything else -> crude category.
"""
import argparse, bisect, csv, glob, gzip, itertools, os, re, sys, unicodedata, zipfile
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent2_pdf import pdf_text


def fold(s):
    return unicodedata.normalize('NFKD', str(s).upper()).encode('ascii', 'ignore').decode()


def nz(s):
    return re.sub(r'[^A-Z0-9]', '', fold(s))


def cents(x):
    return int(round(float(x) * 100))


def rcsv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


class PrefixIndex:
    """longest-common-prefix lookup of (bank text from any word start) vs sorted normalised party names."""
    def __init__(self, rows):
        self.ent = {}
        tmp = defaultdict(lambda: defaultdict(set))
        for e, p, n in rows:
            k = nz(n)
            if len(k) >= 4:
                tmp[e][k].add(p)
        for e, d in tmp.items():
            self.ent[e] = (sorted(d), d)

    def query(self, e, text, minlcp=8):
        if e not in self.ent:
            return set()
        keys, d = self.ent[e]
        words = re.findall(r'[A-Z0-9]+', fold(text))
        best, bp = 0, None
        for i in range(len(words)):
            s = ''.join(words[i:i + 8])[:48]
            if len(s) < 4:
                continue
            j = bisect.bisect_left(keys, s)
            for k in (j - 1, j):
                if 0 <= k < len(keys):
                    a = keys[k]
                    l = 0
                    while l < min(len(a), len(s)) and a[l] == s[l]:
                        l += 1
                    if l > best:
                        best, bp = l, s[:l]
        if best < minlcp:
            return set()
        j = bisect.bisect_left(keys, bp)
        out = set()
        while j < len(keys) and keys[j].startswith(bp):
            if len(keys[j]) == best or best >= 12 or best >= 0.6 * len(keys[j]):
                out |= d[keys[j]]
            j += 1
        return out


def mitm(vals, target, cap=2, maxn=22):
    n = len(vals)
    if n == 0 or n > maxn:
        return 0, None
    h = n // 2
    A, B = vals[:h], vals[h:]
    left = defaultdict(list)
    for r in range(len(A) + 1):
        for comb in itertools.combinations(range(len(A)), r):
            left[sum(A[i] for i in comb)].append(comb)
    cnt, sol = 0, None
    for r in range(len(B) + 1):
        for comb in itertools.combinations(range(len(B)), r):
            for lc in left.get(target - sum(B[i] for i in comb), ()):
                cnt += 1
                if sol is None:
                    sol = list(lc) + [h + i for i in comb]
                if cnt >= cap:
                    return cnt, sol
    return cnt, sol


NUM = re.compile(r'-?\d{1,3}(?:\.\d{3})*,\d{2}|-?\d+\.\d{2}|-?\d{1,3}(?:,\d{3})*\.\d{2}')


def pnum(s):
    if ',' in s and (s.rfind(',') > s.rfind('.')):
        return float(s.replace('.', '').replace(',', '.'))
    return float(s.replace(',', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='.')
    ap.add_argument('--phase', default='sample')
    ap.add_argument('--out', default='/tmp/a2.csv.gz')
    a = ap.parse_args()
    P = os.path.join(a.data, a.phase)
    M = os.path.join(a.data, 'masters')
    b = rcsv(f'{P}/bank_transactions.csv.gz')
    ar = rcsv(f'{P}/ar_invoices.csv.gz')
    apv = rcsv(f'{P}/ap_invoices.csv.gz')
    cn = rcsv(f'{P}/credit_notes.csv.gz')
    cust = rcsv(f'{M}/customers.csv')
    sup = rcsv(f'{M}/suppliers.csv')
    terms = rcsv(f'{M}/payment_terms.csv').set_index('code')
    acct_ent = rcsv(f'{M}/bank_accounts.csv').set_index('account_id').entity.to_dict()

    # ---------------- items
    I = {}
    for r in ar.itertuples():
        I[r.item_id] = dict(k='ARI', e=r.entity, p=r.customer_id, dn=r.document_number, iss=r.issue_date, due=r.due_date,
                            tot=cents(r.total_amount), cur=r.currency, terms=r.payment_terms, po=r.customer_po)
    for r in apv.itertuples():
        I[r.item_id] = dict(k='API', e=r.entity, p=r.supplier_id, dn=r.supplier_invoice_number, iss=r.issue_date,
                            due=r.due_date, tot=cents(r.total_amount), cur=r.currency, terms=r.payment_terms, po='')
    for r in cn.itertuples():
        I[r.item_id] = dict(k=r.item_id[:3], e=r.entity, p=r.party_id, dn=r.document_number, iss=r.issue_date, due=r.issue_date,
                            tot=cents(r.total_amount), cur=r.currency, terms='', po=r.related_document)
    oi = f'{P}/open_items_at_start.csv.gz'
    if os.path.exists(oi):
        for r in rcsv(oi).itertuples():
            I.setdefault(r.item_id, dict(k=r.item_type[:3] if r.item_type[:3] in ('ARI', 'ARC', 'API', 'APC') else r.item_id[:3],
                                         e=r.entity, p=r.party_id, dn=r.document_number, iss=r.issue_date, due=r.due_date,
                                         tot=cents(r.total_amount), cur=r.currency, terms='', po=''))
            I[r.item_id]['open'] = cents(r.open_amount)
    for v in I.values():
        v.setdefault('open', v['tot'])
    full = defaultdict(list)            # (entity, nz docno) -> items
    suf = defaultdict(list)             # (entity, last6 digits) -> AR items
    byparty = defaultdict(list)
    for i, v in I.items():
        full[(v['e'], nz(v['dn']))].append(i)
        if v['k'] in ('ARI', 'ARC'):
            dd = re.sub(r'\D', '', v['dn'])
            suf[(v['e'], dd[-6:])].append(i)
        byparty[v['p']].append(i)
    for p in byparty:
        byparty[p].sort(key=lambda i: (I[i]['iss'], I[i]['dn']))

    def sgn(i):
        return -1 if I[i]['k'] in ('ARC', 'APC') else 1

    names = [(r.entity, r.customer_id, r.name) for r in cust.itertuples()] + \
            [(r.entity, r.customer_id, r.trade_name) for r in cust.itertuples() if r.trade_name] + \
            [(r.entity, r.customer_id, r.customer_name) for r in ar.drop_duplicates(['customer_id', 'customer_name']).itertuples()]
    CX = PrefixIndex(names)
    SX = PrefixIndex([(r.entity, r.supplier_id, r.name) for r in sup.itertuples()])
    iban_c = {r.iban: r.customer_id for r in cust.itertuples() if r.iban}
    children = defaultdict(set)
    for r in cust.itertuples():
        if r.parent_customer_id:
            children[r.parent_customer_id].add(r.customer_id)

    out = {}   # txn -> list of rows (item, amount, cat, exc)
    used = set()

    def take(t, rows):
        out[t] = rows
        for it, amt, _, _ in rows:
            if it:
                I[it]['open'] = max(0, I[it]['open'] - cents(amt))

    def resolve_ar(ent, ref, amt_c=None):
        c = [i for i in full.get((ent, nz(ref)), []) if I[i]['k'] in ('ARI', 'ARC')]
        if len(c) == 1:
            return c[0]
        dd = re.sub(r'\D', '', ref)
        if len(dd) >= 5:
            c = suf.get((ent, dd[-6:].rjust(6, '0')), [])
            c2 = [i for i in c if amt_c is None or I[i]['open'] == amt_c] or c
            c2 = [i for i in c2 if I[i]['open'] > 0] or c2
            if c2:
                return c2[-1]
        return None

    # ---------------- AP payment runs
    pr = pd.concat([rcsv(f) for f in glob.glob(f'{P}/payment_runs/*.csv')]) if glob.glob(f'{P}/payment_runs/*.csv') else pd.DataFrame()
    e2e_rows = defaultdict(list); run_rows = defaultdict(list)
    for r in pr.itertuples():
        ent = acct_ent.get(r.debtor_account_id, '')
        cands = [i for e in {ent, 'ES01', 'ES02', 'PT01', 'ES00'} for i in full.get((e, nz(r.document_number)), []) if I[i]['k'] in ('API', 'APC')]
        da = cents(r.document_amount)
        c2 = [i for i in cands if I[i]['tot'] == abs(da)] or cands
        if len(c2) > 1:
            c2 = [i for i in c2 if I[i]['e'] == ent] or c2
        it = c2[0] if c2 else None
        e2e_rows[r.end_to_end_id].append((it, abs(da), cents(r.transfer_amount)))
        run_rows[r.run_id].append((r.end_to_end_id, it, abs(da), cents(r.transfer_amount)))

    def pr_rows(rows):
        return [(it, f'{amt / 100:.2f}', '', '') for it, amt, *_ in rows if it]

    # ---------------- DD / card / cheque
    def load_dir(d):
        fs = glob.glob(f'{P}/{d}/*.csv')
        return pd.concat([rcsv(f) for f in fs]) if fs else pd.DataFrame()
    dd = load_dir('direct_debit_batches'); dd_b = defaultdict(list)
    for r in dd.itertuples():
        dd_b[r.batch_id].append((acct_ent.get(r.account_id, ''), r.remittance_info, cents(r.amount)))
    cs = load_dir('card_settlements'); card = defaultdict(list); card_net = defaultdict(int)
    for r in cs.itertuples():
        card[(r.merchant_id, r.settlement_id)].append((acct_ent.get(r.account_id, ''), r.order_number, cents(r.gross)))
        card_net[(r.merchant_id, r.settlement_id)] += cents(r.net)
    card_key = defaultdict(list)
    for (mid, sid), net in card_net.items():
        card_key[(mid, net)].append(sid)
    ch = load_dir('cheque_deposits'); chq = defaultdict(list)
    for r in ch.itertuples():
        chq[r.deposit_id].append((acct_ent.get(r.account_id, ''), r.stub_reference, cents(r.amount), r.drawer))

    # ---------------- remittance advices
    TOK = r'[A-Z0-9][A-Z0-9\-/]{6,}[0-9]'
    ra_ref = defaultdict(set); ra_txt = {}
    rdir = f'{P}/remittance_advices'
    if os.path.exists(f'{rdir}/index.csv'):
        for r in rcsv(f'{rdir}/index.csv').itertuples():
            p = f'{rdir}/{r.file}'
            try:
                if r.format in ('csv', 'txt'):
                    s = open(p, encoding='utf-8', errors='ignore').read()
                elif r.format == 'xlsx':
                    z = zipfile.ZipFile(p)
                    s = ' '.join(re.sub(r'<[^>]+>', ' ', z.read(n).decode('utf-8', 'ignore')) for n in z.namelist() if 'sharedStrings' in n or 'worksheets/sheet' in n)
                else:
                    s = pdf_text(p)
            except Exception:
                continue
            s = s.upper(); ra_txt[r.file] = (r.entity, s)
            for tok in set(re.findall(TOK, s)):
                if sum(ch_.isdigit() for ch_ in tok) >= 6:
                    ra_ref[tok].add(r.file)

    def ra_rows(ent, s, amt_c):
        """docnos in advice order, each followed by numbers; paid = last number before next doc."""
        toks = re.findall(r'[A-Z0-9][A-Z0-9\-/\.,]*', s)
        res = []; cur = None; nums = []
        for tk in toks:
            it = None
            if re.search(r'[A-Z]', tk) and re.search(r'\d{4}', tk) or re.fullmatch(r'\d{5,10}', tk):
                c = full.get((ent, nz(tk)), [])
                c = [i for i in c if I[i]['k'] in ('ARI', 'ARC')]
                it = c[0] if c else None
            if it:
                if cur:
                    res.append((cur, nums))
                cur, nums = it, []
            elif cur and NUM.fullmatch(tk):
                nums.append(pnum(tk))
        if cur:
            res.append((cur, nums))
        rows = []
        for it, nums in res:
            paid = abs(nums[-1]) if nums else I[it]['open'] / 100
            gross = I[it]['open'] / 100
            exc = ''
            if I[it]['k'] == 'ARC':
                paid = gross
            elif paid + 0.01 < gross:
                exc = 'SHORT_DISPUTE'
            else:
                paid = gross
            rows.append((it, f'{paid:.2f}', '', exc))
        return rows

    # ---------------- main loop
    b = b.sort_values(['booking_date', 'bank_txn_id'])
    stats = defaultdict(int)
    for t in b.itertuples():
        T = ' '.join([t.text, t.ref1, t.ref2, t.remittance_info, t.end_to_end_id, t.counterparty_name]).upper()
        amt = cents(t.amount); A = abs(amt); ent = t.entity; d = t.booking_date
        if amt < 0:
            hit = None
            for tok in set(re.findall(r'(?:PR|CF)\d{12}-\d{4}', T)) | ({t.end_to_end_id} if t.end_to_end_id else set()):
                if tok in e2e_rows:
                    hit = e2e_rows[tok]; break
            if hit is None:
                for run in set(re.findall(r'(?:PR|CF)\d{12}', T)):
                    rr = run_rows.get(run)
                    if not rr:
                        continue
                    tot = sum({e: ta for e, _, _, ta in rr}.values())
                    if tot == A:
                        hit = [(it, da) for _, it, da, _ in rr]; break
                    byE = defaultdict(list)
                    for e, it, da, ta in rr:
                        byE[e].append((it, da, ta))
                    ok = [v for v in byE.values() if v[0][2] == A]
                    if len(ok) == 1:
                        hit = ok[0]; break
            if hit:
                take(t.bank_txn_id, pr_rows(hit)); stats['ap_run'] += 1; continue
            sp = SX.query(ent, T)
            c = [i for p in sp for i in byparty[p] if I[i]['k'] == 'API' and I[i]['open'] == A and I[i]['iss'] <= d]
            if c:
                take(t.bank_txn_id, [(c[0], f'{A / 100:.2f}', '', '')]); stats['ap_name_amt'] += 1; continue
            out[t.bank_txn_id] = [('', '', 'BANK_FEES', '')]; continue
        # ---- credits: batches
        done = False
        for tok in re.findall(r'[A-Z]*\d{8,}', T):
            if tok in dd_b:
                rows = []
                for e, ref, a_ in dd_b[tok]:
                    it = resolve_ar(e, ref, a_)
                    if it:
                        rows.append((it, f'{a_ / 100:.2f}', '', ''))
                take(t.bank_txn_id, rows); stats['dd'] += 1; done = True; break
            if tok in chq:
                rows = []
                for e, ref, a_, drawer in chq[tok]:
                    it = resolve_ar(e, ref, a_) if ref else None
                    if not it:
                        ps = CX.query(e, drawer)
                        c = [i for p in ps for i in byparty[p] if I[i]['open'] == a_]
                        it = c[0] if c else None
                    if it:
                        rows.append((it, f'{a_ / 100:.2f}', '', ''))
                take(t.bank_txn_id, rows); stats['cheque'] += 1; done = True; break
        if not done:
            for tok in set(re.findall(r'\d{9}', T)):
                sids = card_key.get((tok, A))
                if sids:
                    rows = []
                    for e, order, g_ in card[(tok, sids[0])]:
                        c = full.get((e, nz(order)), [])
                        if c:
                            rows.append((c[0], f'{g_ / 100:.2f}', '', ''))
                    take(t.bank_txn_id, rows); stats['card'] += 1; done = True; break
        if not done:
            for tok in re.findall(TOK, T):
                fs = ra_ref.get(tok)
                if fs and len(fs) <= 2 and not re.fullmatch(r'(FV|FA|RE|INV|FT)[\d\-]+', tok):
                    f = sorted(fs)[0]
                    rows = ra_rows(ra_txt[f][0], ra_txt[f][1], A)
                    if rows:
                        take(t.bank_txn_id, rows); stats['ra'] += 1; done = True
                    break
        if done:
            continue
        # ---- AR party + amount
        parties = CX.query(ent, T)
        if t.counterparty_account in iban_c:
            parties.add(iban_c[t.counterparty_account])
        for p in list(parties):
            parties |= children.get(p, set())
        refs = []
        for tok in set(re.findall(r'[A-Z0-9\-/]{5,}', T)):
            it = None
            c = [i for i in full.get((ent, nz(tok)), []) if I[i]['k'] in ('ARI', 'ARC')]
            if c:
                it = c[0]
            elif re.fullmatch(r'\d{5,7}', tok):
                c = [i for i in suf.get((ent, tok.rjust(6, '0')[-6:]), []) if I[i]['open'] > 0 and (not parties or I[i]['p'] in parties)]
                it = c[0] if len(c) == 1 else None
            if it and I[it]['open'] > 0:
                refs.append(it)
        if not parties and refs:
            parties = {I[refs[0]]['p']}
        pool = [i for p in parties for i in byparty[p] if I[i]['open'] > 0 and I[i]['iss'] <= d and I[i]['k'] in ('ARI', 'ARC') and I[i]['cur'] == t.currency]
        pool.sort(key=lambda i: (I[i]['iss'], I[i]['dn']))
        rows = None
        refs = list(dict.fromkeys(refs))
        if refs and sum(sgn(i) * I[i]['open'] for i in refs) == A:
            rows = [(i, f'{I[i]["open"] / 100:.2f}', '', '') for i in refs]; stats['refs_sum'] += 1
        if rows is None:
            c = [i for i in pool if I[i]['k'] == 'ARI' and I[i]['open'] == A]
            if c:
                pref = [i for i in c if i in refs]
                i = (pref or c)[0]
                rows = [(i, f'{A / 100:.2f}', '', '')]; stats['single_amt'] += 1
        if rows is None and pool:
            vals = [sgn(i) * I[i]['open'] for i in pool]
            cs_ = list(itertools.accumulate(vals))
            if A in cs_:
                k = cs_.index(A)
                rows = [(i, f'{I[i]["open"] / 100:.2f}', '', '') for i in pool[:k + 1]]; stats['fifo_prefix'] += 1
        if rows is None and 1 < len(pool) <= 22:
            vals = [sgn(i) * I[i]['open'] for i in pool]
            cnt, sol = mitm(vals, A)
            if cnt >= 1:
                rows = [(pool[k], f'{I[pool[k]]["open"] / 100:.2f}', '', '') for k in sol]; stats['subset%d' % min(cnt, 2)] += 1
        if rows is None:
            for i in pool:
                v = I[i]
                if v['k'] != 'ARI' or v['terms'] not in terms.index:
                    continue
                pct = terms.at[v['terms'], 'early_payment_discount_pct']
                if not pct:
                    continue
                if round(v['open'] * (1 - float(pct) / 100)) == A or abs(round(v['tot'] * (1 - float(pct) / 100)) - A) <= 1:
                    lim = (pd.Timestamp(v['iss']) + pd.Timedelta(days=int(terms.at[v['terms'], 'discount_days']) + 3)).strftime('%Y-%m-%d')
                    if d <= lim:
                        rows = [(i, f'{v["open"] / 100:.2f}', '', 'SHORT_DISCOUNT')]
                    else:
                        rows = [(i, f'{A / 100:.2f}', '', 'PARTIAL')]
                    stats['discount'] += 1; break
        if rows is None and len(refs) == 1:
            i = refs[0]; o = I[i]['open']
            if A < o:
                rows = [(i, f'{A / 100:.2f}', '', 'PARTIAL')]
            else:
                rows = [(i, f'{o / 100:.2f}', '', 'OVERPAYMENT')]
            stats['ref_partial'] += 1
        if rows:
            take(t.bank_txn_id, rows); continue
        out[t.bank_txn_id] = [('', '', 'UNIDENTIFIED', 'UNIDENTIFIED')]; stats['unident'] += 1
    print(dict(stats), file=sys.stderr)
    op = gzip.open if a.out.endswith('.gz') else open
    with op(a.out, 'wt', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['bank_txn_id', 'item_id', 'amount', 'category', 'exception'])
        for t in b.bank_txn_id:
            rows = out.get(t) or [('', '', 'UNIDENTIFIED', 'UNIDENTIFIED')]
            seen = set()
            for it, amt, cat, exc in rows:
                if it in seen:
                    continue
                seen.add(it)
                w.writerow([t, it, amt, cat, exc])


if __name__ == '__main__':
    main()
