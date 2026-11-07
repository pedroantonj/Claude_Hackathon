"""AR amount-logic evaluation on train 2025H1 given the TRUE party (upper bound for party-ID+amount matcher)."""
import pandas as pd, numpy as np, itertools, sys
from collections import defaultdict, Counter
pool=pd.read_pickle('/tmp/a2_pool.pkl'); m=pd.read_pickle('/tmp/a2_m_train.pkl')
arp=pool[pool.kind.isin(['ARI','ARC'])].copy()
arp['c']=(arp.total_amount*100).round().astype('int64')*np.where(arp.kind=='ARC',-1,1)
byp={p:df.sort_values(['due_date','issue_date','docno']) for p,df in arp.groupby('party_id')}
ar=m[m.kind.isin(['ARI','ARC'])]
L=ar.groupby('bank_txn_id').agg(seg=('segment','first'),party=('party_id',lambda s:s.mode()[0]),nparty=('party_id','nunique'),d=('booking_date','first'),amt=('bamt','first'),items=('item_id',frozenset),bc=('currency','first'),ic=('currency_it','first'))
L=L[~L.seg.isin(['ar_dd_batch','ar_card_settlement','ar_cheque_deposit','ar_dd_returned','ar_card_chargeback','ar_cheque_returned'])]
L=L.sample(min(len(L),30000),random_state=0)
def mitm(vals, target, cap=2):
    """count subsets summing to target (meet in the middle), return (count, one solution idx)"""
    n=len(vals); h=n//2; A=vals[:h]; B=vals[h:]
    left=defaultdict(list)
    for r in range(len(A)+1):
        for comb in itertools.combinations(range(len(A)),r):
            left[sum(A[i] for i in comb)].append(comb)
    cnt=0; sol=None
    for r in range(len(B)+1):
        for comb in itertools.combinations(range(len(B)),r):
            s=sum(B[i] for i in comb)
            for lc in left.get(target-s,()):
                cnt+=1
                if sol is None: sol=list(lc)+[h+i for i in comb]
                if cnt>=cap: return cnt,sol
    return cnt,sol
out=[]
for t,r in L.iterrows():
    df=byp.get(r.party)
    if df is None: out.append((t,0,'noparty',False)); continue
    o=df[(df.issue_date<=r.d)&(df.settle>=r.d)]
    n=len(o); tgt=int(round(abs(r.amt)*100)); ids=o.item_id.tolist(); c=o.c.tolist()
    if r.bc!=r.ic: out.append((t,n,'fx',False)); continue
    sub=frozenset(ids)>=r['items']
    single=[i for i,v in zip(ids,c) if v==tgt]
    if len(single)==1: out.append((t,n,'single_unique',frozenset(single)==r['items'])); continue
    if len(single)>1:
        fifo=o[o.c==tgt].sort_values(['issue_date','docno']).item_id.iloc[0]  # oldest by issue date
        out.append((t,n,'single_multi_fifo',frozenset([fifo])==r['items'])); continue
    if sum(c)==tgt: out.append((t,n,'all_open',frozenset(ids)==r['items'])); continue
    cs=np.cumsum(c); k=np.where(cs==tgt)[0]
    if len(k): out.append((t,n,'fifo_prefix_due',frozenset(ids[:k[0]+1])==r['items'])); continue
    o2=o.sort_values(['issue_date','docno']); cs=np.cumsum(o2.c.values); k=np.where(cs==tgt)[0]
    if len(k): out.append((t,n,'fifo_prefix_issue',frozenset(o2.item_id.values[:k[0]+1])==r['items'])); continue
    if n<=26:
        cnt,sol=mitm(c,tgt)
        if cnt==1: out.append((t,n,'subset_unique',frozenset(ids[i] for i in sol)==r['items'])); continue
        if cnt>1: out.append((t,n,'subset_ambig',frozenset(ids[i] for i in sol)==r['items'])); continue
        out.append((t,n,'no_exact',False)); continue
    out.append((t,n,'too_many',False))
R=pd.DataFrame(out,columns=['t','n_open','rule','ok']).set_index('t').join(L[['seg']])
R.to_pickle('/tmp/a2_ar_rules.pkl')
print(R.n_open.describe(percentiles=[.5,.9,.99]).round(1).to_dict())
print(pd.crosstab(R.rule,R.ok,margins=True))
x=R.groupby('seg').agg(n=('ok','size'),exact=('ok','mean'),nopen_med=('n_open','median'))
top=R.groupby(['seg','rule']).size().unstack(fill_value=0)
print(x.join((top.T/top.sum(1)).T.round(2)).sort_values('n',ascending=False).to_string())
