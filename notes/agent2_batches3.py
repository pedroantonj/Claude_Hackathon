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
pr=rd('payment_runs').head(1); print('payment rows',len(pr), pr.run_type.value_counts().to_dict())
ap=it[it.kind.isin(['API','APC'])]
sup=pd.read_csv("/Users/pedroantonperez/Desktop/Claude Hackathon/data/masters/suppliers.csv",dtype=str,keep_default_na=False)
apk=ap.assign(k=ap.party_name.str.upper()+'|'+ap.dn)
# map by (beneficiary name, docno)  -- docno alone ambiguous across suppliers
key2item=apk.drop_duplicates('k').set_index('k').item_id
pr['k']=pr.beneficiary.str.upper()+'|'+pr.document_number.map(nrm)
pr['item']=pr.k.map(key2item)
print('payment_run doc->item resolved by name+docno', pr.item.notna().mean())


ar=it[it.kind=='ARI']; arc=ar.groupby('dn').item_id.apply(list).to_dict()
def res_ar(ref,amt):
    c=arc.get(nrm(ref),[])
    return c[0] if len(c)==1 else (None if not c else c[0])

# suffix index per entity: (entity, last6 digits) -> [(item, amount, party)]
from collections import defaultdict
suf=defaultdict(list)
for i,e,d,a,p in zip(ar.item_id,ar.entity,ar.docno,ar.total_amount,ar.party_id):
    dd=re.sub(r'\D','',d)
    suf[(e,dd[-6:])].append((i,a,p))
acct=pd.read_csv("/Users/pedroantonperez/Desktop/Claude Hackathon/data/masters/bank_accounts.csv",dtype=str).set_index('account_id').entity
cust=pd.read_csv("/Users/pedroantonperez/Desktop/Claude Hackathon/data/masters/customers.csv",dtype=str,keep_default_na=False)
def res2(ent,ref,amt):
    c=arc.get(nrm(ref),[])
    if len(c)==1: return c[0],'full'
    dd=re.sub(r'\D','',str(ref))
    if len(dd)>=5:
        c=suf.get((ent,dd[-6:].rjust(6,'0')),[])
        c2=[x for x in c if abs(x[1]-amt)<0.011] or c
        if len(c2)==1: return c2[0][0],'suffix'
        if c2: return c2[-1][0],'suffix_amb'
    return None,'none'
cr=b[b.amount>0]
for name,d,idcol,refcol,amtcol in [('dd','direct_debit_batches','batch_id','remittance_info','amount'),('cheque','cheque_deposits','deposit_id','stub_reference','amount')]:
    x=rd(d); x['ent']=x.account_id.map(acct)
    r=[res2(e,rf,float(a)) for e,rf,a in zip(x.ent,x[refcol],x[amtcol])]
    x['item']=[a for a,_ in r]; x['how']=[h for _,h in r]
    print(name, x.how.value_counts(normalize=True).round(3).to_dict())
    grp=x.groupby(idcol).item.apply(lambda s:set(s.dropna()))
    ids=set(grp.index); pred={}
    for t,T in zip(cr.bank_txn_id,cr['T']):
        for tok in re.findall(r'[A-Z]*\d{8,}',T):
            if tok in ids: pred[t]=grp[tok]; break
    evaluate(name,pred)
c=rd('card_settlements'); c['net']=c.net.astype(float)
c['item']=[res_ar(r,0) for r in c.order_number]
cs=c.groupby('settlement_id').agg(mid=('merchant_id','first'),d=('settlement_date','first'),net=('net','sum'),items=('item',lambda s:set(s.dropna())))
cs['netr']=cs.net.round(2)
key={(mm,round(n,2)):i for i,mm,n in zip(cs.index,cs.mid,cs.net)}
pred={}
for t,T,a in zip(cr.bank_txn_id,cr['T'],cr.amount):
    for tok in re.findall(r'\d{9}',T):
        k=(tok,round(a,2))
        if k in key: pred[t]=cs.at[key[k],'items']; break
print('card hits',len(pred)); evaluate('card',pred)
