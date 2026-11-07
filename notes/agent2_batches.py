import glob, re, pandas as pd, numpy as np
from agent2_load import nrm
D="/Users/pedroantonperez/Desktop/Claude Hackathon/data/train"
def rd(d,pat=r'_2024(1[0-2])|_2025(0[1-7])'):
    fs=[f for f in glob.glob(f"{D}/{d}/*.csv") if re.search(pat,f)]
    return pd.concat([pd.read_csv(f,dtype=str,keep_default_na=False) for f in fs])
m=pd.read_pickle('/tmp/a2_m_train.pkl'); b=pd.read_pickle('/tmp/a2_b_train.pkl'); it=pd.read_pickle('/tmp/a2_ittrain.pkl')
gold=m.groupby('bank_txn_id').item_id.apply(frozenset)
segof=m.groupby('bank_txn_id').segment.first()
def evaluate(name, pred):  # pred: dict txn->set(items)
    rows=[]
    for t,s in pred.items():
        gs=gold.get(t,frozenset()); rows.append((segof.get(t,'nonitem'),len(s&gs),len(s),len(gs),s==gs))
    df=pd.DataFrame(rows,columns=['seg','tp','np','ng','exact'])
    r=df.groupby('seg').agg(lines=('tp','size'),prec=('tp','sum'),np=('np','sum'),ng=('ng','sum'),exact=('exact','mean'))
    r['prec']=r.prec/r.np; print('==',name); print(r.round(3).sort_values('lines',ascending=False).head(12).to_string())
# doc-number -> item index per entity (AR docno, AP docno per supplier)
it['dn']=it.docno.map(nrm)
# ---- payment runs
pr=rd('payment_runs'); print('payment rows',len(pr), pr.run_type.value_counts().to_dict())
ap=it[it.kind.isin(['API','APC'])]
sup=pd.read_csv("/Users/pedroantonperez/Desktop/Claude Hackathon/data/masters/suppliers.csv",dtype=str,keep_default_na=False)
apk=ap.assign(k=ap.party_name.str.upper()+'|'+ap.dn)
# map by (beneficiary name, docno)  -- docno alone ambiguous across suppliers
key2item=apk.drop_duplicates('k').set_index('k').item_id
pr['k']=pr.beneficiary.str.upper()+'|'+pr.document_number.map(nrm)
pr['item']=pr.k.map(key2item)
print('payment_run doc->item resolved by name+docno', pr.item.notna().mean())

# better doc->item: by normalized docno, disambiguate by name, amount, entity
apidx=ap.groupby('dn')
cand={k:v[['item_id','party_name','total_amount','entity']].values.tolist() for k,v in apidx}
from rapidfuzz import fuzz
def resolve(dn,ben,amt):
    c=cand.get(dn,[])
    if len(c)==1: return c[0][0]
    if not c: return None
    c2=[x for x in c if abs(abs(x[2])-abs(amt))<0.011] or c
    if len(c2)==1: return c2[0][0]
    return max(c2,key=lambda x:fuzz.token_set_ratio(x[1].upper(),ben.upper()))[0]
pr['item']=[resolve(nrm(d),bn,float(a)) for d,bn,a in zip(pr.document_number,pr.beneficiary,pr.document_amount)]
print('resolved',pr.item.notna().mean())
pr['ta']=pr.transfer_amount.astype(float)
e2e=pr.groupby('end_to_end_id').agg(items=('item',lambda s:set(s.dropna())),amt=('ta','first'),run=('run_id','first'))
runtot=e2e.groupby('run').amt.sum()
runs=pr.groupby('run_id').item.apply(lambda s:set(s.dropna()))
e2e_by_run=e2e.reset_index().groupby('run')
bd=b[b.amount<0]
pred={}; how={}
for t,T,ee,amt in zip(bd.bank_txn_id,bd['T'],bd.end_to_end_id,bd.amount):
    amt=-amt; hit=None
    toks=set(re.findall(r'(?:PR|CF)\d{12}-\d{4}',T))|({ee} if ee else set())
    for tok in toks:
        if tok in e2e.index: hit=e2e.at[tok,'items']; how[t]='e2e'; break
    if hit is None:
        for run in set(re.findall(r'(?:PR|CF)\d{12}',T)):
            if run not in runtot.index: continue
            if abs(runtot[run]-amt)<0.011: hit=runs[run]; how[t]='run_total'; break
            grp=e2e_by_run.get_group(run)
            # truncated e2e prefix (ref1 16 chars) + amount
            ok=grp[(grp.amt-amt).abs()<0.011]
            if len(ok)==1: hit=ok['items'].iloc[0]; how[t]='run+amt'; break
            # confirming: group by beneficiary?  try sum of all e2e of same amount... fallback none
    if hit is not None: pred[t]=hit
import collections; print(collections.Counter(how.values()))
print('AP debit lines with run hit', len(pred), 'of', len(bd))
evaluate('payment_runs', pred)
miss=[t for t in gold.index if segof[t].startswith('ap_') and t not in pred]
mm=m[m.bank_txn_id.isin(miss)].drop_duplicates('bank_txn_id')
print(mm.segment.value_counts().head(10))
for r in mm.groupby('segment').head(2).itertuples(): print(r.segment,r.entity,r.bamt,r.T[:140])
