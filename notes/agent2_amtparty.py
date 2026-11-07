"""Realistic single-item matcher: exact-amount open items in entity x name candidates."""
import pandas as pd, numpy as np, re
from collections import defaultdict
from agent2_party import PartyIndex, build_names
pool=pd.read_pickle('/tmp/a2_pool.pkl'); m=pd.read_pickle('/tmp/a2_m_train.pkl'); it=pd.read_pickle('/tmp/a2_ittrain.pkl')
R=pd.read_pickle('/tmp/a2_ar_rules.pkl')
idx=PartyIndex(build_names(it[it.kind=='ARI']))
arp=pool[pool.kind=='ARI']
arp=arp.assign(c=(arp.total_amount*100).round().astype('int64'))
byamt=defaultdict(list)
for i,e,c,p,iss,st,dn in zip(arp.item_id,arp.entity,arp.c,arp.party_id,arp.issue_date,arp.settle,arp.docno):
    byamt[(e,c)].append((i,p,iss,st,dn))
L=m.groupby('bank_txn_id').agg(seg=('segment','first'),party=('party_id',lambda s:s.mode()[0]),items=('item_id',frozenset),ent=('ientity','first'),d=('booking_date','first'),amt=('bamt','first'),T=('T','first'),cp=('counterparty_name','first'))
L=L.loc[R.index.intersection(L.index)].join(R[['rule']])
# also include nonitem/unidentified lines positive to measure false positives
b=pd.read_pickle('/tmp/a2_b_train.pkl'); g=pd.read_pickle('/tmp/a2_gtrain.pkl'); g=g[g.yr=='2025']
ni=g[(g.item_id=='')].drop_duplicates('bank_txn_id').set_index('bank_txn_id')
U=b[b.bank_txn_id.isin(ni.index)&(b.amount>0)].set_index('bank_txn_id')
U=U.assign(seg=ni.segment.reindex(U.index),party='',items=[frozenset()]*len(U),ent=U.entity,d=U.booking_date,amt=U.amount,cp=U.counterparty_name,rule='nonitem')[L.columns]
U=U.sample(min(len(U),5000),random_state=0)
A=pd.concat([L,U])
out=[]
for t,r in A.iterrows():
    tgt=int(round(abs(r.amt)*100))
    c=[x for x in byamt.get((r.ent,tgt),[]) if x[2]<=r.d and x[3]>=r.d]
    q=idx.query(r.ent,(r.cp or '')+' '+r['T'],k=30)
    top=q[0][1] if q else 0
    names={p for p,s in q if s>=0.6*top and s>0}
    cn=[x for x in c if x[1] in names]
    # pick: name-filtered unique ; else FIFO by issue among name-filtered
    pick=None; how='none'
    if len(cn)>=1:
        cn.sort(key=lambda x:(x[2],x[4])); pick=cn[0][0]; how='name+amt' if len(cn)==1 else 'name+amt_fifo'
    elif len(c)==1: pick=c[0][0]; how='amt_only_unique'
    out.append((t,len(c),len(cn),how,pick is not None and frozenset([pick])==r['items']))
O=pd.DataFrame(out,columns=['t','n_amt','n_amt_name','how','ok']).set_index('t').join(A[['seg','rule']])
print(O.groupby('how').ok.agg(['size','mean']))
s=O.groupby('seg').agg(n=('ok','size'),ok=('ok','mean'),amt_any=('n_amt',lambda s:(s>0).mean()),amt_med=('n_amt','median'),name_amt=('n_amt_name',lambda s:(s>0).mean()))
print(s.sort_values('n',ascending=False).round(3).head(30).to_string())
print('nonitem lines that would get a forced match:', (O[O.rule=='nonitem'].how!='none').groupby(O.seg).mean().round(3).to_dict())
