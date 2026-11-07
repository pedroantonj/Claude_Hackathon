import sys, re, pandas as pd, numpy as np
from agent2_load import load, nrm
phase=sys.argv[1] if len(sys.argv)>1 else 'sample'; since=sys.argv[2] if len(sys.argv)>2 else None
b,g,it=load(phase,since)
b['T']=(b.text+' '+b.ref1+' '+b.ref2+' '+b.remittance_info+' '+b.end_to_end_id).str.upper()
b['TN']=b['T'].map(nrm)
b['DIG']=b['T'].map(lambda s:set(re.findall(r'\d{4,}',s)))
m=g[g.item_id!=''].merge(b[['bank_txn_id','T','TN','DIG','booking_date','amount','currency','statement_format','counterparty_account','counterparty_name','entity']],on='bank_txn_id').merge(it,on='item_id',how='left',suffixes=('','_it'))
m['dn']=m.docno.map(nrm)
m['full']=[d in t if d else False for d,t in zip(m.dn,m.TN)]
def suf(d,digs):
    dd=re.sub(r'\D','',d)
    for k in digs:
        if len(k)>=5 and dd.endswith(k): return True
        if len(k)>=5 and dd.lstrip('0').endswith(k.lstrip('0')) and len(k.lstrip('0'))>=4: return True
    return False
m['suffix']=[suf(d,s) for d,s in zip(m.docno,m.DIG)]
po=m.customer_po.map(nrm)
m['po']=[bool(p) and len(p)>=5 and p in t for p,t in zip(po,m.TN)]
m['anyref']=m.full|m.suffix|m.po
r=m.groupby('segment').agg(rows=('item_id','size'),full=('full','mean'),suffix=('suffix','mean'),po=('po','mean'),anyref=('anyref','mean')).sort_values('rows',ascending=False)
print(r.round(3).to_string())
print('ALL',m[['full','suffix','po','anyref']].mean().round(3).to_dict())
m.to_pickle(f'/tmp/a2_m_{phase}.pkl'); b.to_pickle(f'/tmp/a2_b_{phase}.pkl')
