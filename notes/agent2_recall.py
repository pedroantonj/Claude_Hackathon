import pandas as pd, numpy as np
from agent2_party import PartyIndex, build_names
L=pd.read_pickle('/tmp/a2_party.pkl'); it=pd.read_pickle('/tmp/a2_ittrain.pkl')
idx=PartyIndex(build_names(it[it.kind=='ARI']))
res=[idx.query(e,(cp or '')+' '+T,k=50) for e,cp,T in zip(L.ent,L.cp,L['T'])]
for frac in [0.99,0.8,0.6]:
    cand=[{p for p,s in q if q and s>=frac*q[0][1]} for q in res]
    L['rec']=[p in c for p,c in zip(L.party,cand)]; L['sz']=[len(c) for c in cand]
    print(frac,'recall',round(L.rec.mean(),3),'median size',L.sz.median(),'p90',L.sz.quantile(.9))
print(L.assign(rec=L.rec).groupby('fmt').rec.mean())
print(L[~L.rec].groupby('seg').size().sort_values().tail(5))
x=L[~L.rec].sample(12,random_state=1)
import re
for r in x.itertuples():
    cust=it[(it.party_id==r.party)].party_name.unique()[:3]
    print(r.seg,'|',r.cp,'|',r.T[:110],'| TRUE:',list(cust))
