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
cr=b[b.amount>0]
for name,d,idcol,refcol,amtcol in [('dd','direct_debit_batches','batch_id','remittance_info','amount'),('card','card_settlements','settlement_id','order_number','gross'),('cheque','cheque_deposits','deposit_id','stub_reference','amount')]:
    x=rd(d); x['item']=[res_ar(r,a) for r,a in zip(x[refcol],x[amtcol])]
    print(name,len(x),'ref->item',x.item.notna().mean(), x[[refcol]].head(3).values.ravel())
    grp=x.groupby(idcol).item.apply(lambda s:set(s.dropna()))
    ids=set(grp.index); pred={}
    for t,T in zip(cr.bank_txn_id,cr['T']):
        for tok in re.findall(r'[A-Z]*\d{8,}',T):
            if tok in ids: pred[t]=grp[tok]; break
            # card: settlement_id may be embedded
        else:
            pass
    print(name,'lines hit',len(pred)); 
    if pred: evaluate(name,pred)
    miss=m[m.segment.str.contains({'dd':'dd_batch','card':'card_settlement','cheque':'cheque_deposit'}[name])&~m.bank_txn_id.isin(pred)].drop_duplicates('bank_txn_id')
    print('missed lines',len(miss)); 
    for r in miss.head(4).itertuples(): print('   ',r.entity,r.bamt,r.T[:120])
x=rd('direct_debit_batches'); x['item']=[res_ar(r,a) for r,a in zip(x.remittance_info,x.amount)]
print(x[x.item.isna()].remittance_info.str.replace(r'\d','9',regex=True).value_counts().head(8))
print(x[x.item.isna()].head(5).to_string())
y=rd('cheque_deposits'); y['item']=[res_ar(r,a) for r,a in zip(y.stub_reference,y.amount)]
print(y[y.item.isna()].stub_reference.str.replace(r'\d','9',regex=True).value_counts().head(8))
print(y[y.item.isna()].head(5).to_string())
c=rd('card_settlements'); print(c.groupby('settlement_id').agg(m=('merchant_id','first'),d=('settlement_date','first'),n=('gross','size'),net=('net',lambda s:s.astype(float).sum()),gross=('gross',lambda s:s.astype(float).sum())).head(5))
