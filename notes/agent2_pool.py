import pandas as pd, numpy as np, time
t0=time.time()
D="/Users/pedroantonperez/Desktop/Claude Hackathon/data/train"
bd=pd.read_csv(f"{D}/bank_transactions.csv.gz",usecols=['bank_txn_id','booking_date'],dtype=str)
g=pd.read_pickle('/tmp/a2_gtrain.pkl'); it=pd.read_pickle('/tmp/a2_ittrain.pkl')
gi=g[g.item_id!=''].merge(bd,on='bank_txn_id')
# final settle date = last allocation date that is not RETURNED
s=gi[gi.exception!='RETURNED'].groupby('item_id').booking_date.agg(['min','max'])
it=it.join(s,on='item_id'); it=it.rename(columns={'min':'first_alloc','max':'settle'})
it['settle']=it.settle.fillna('9999-12-31'); it['first_alloc']=it.first_alloc.fillna('9999-12-31')
it[['item_id','kind','entity','party_id','docno','issue_date','due_date','total_amount','currency','payment_terms','payment_method','customer_po','first_alloc','settle']].to_pickle('/tmp/a2_pool.pkl')
gi[['bank_txn_id','item_id','amount','exception','segment','booking_date']].to_pickle('/tmp/a2_gi.pkl')
print('never allocated share by kind', it[it.settle=='9999-12-31'].kind.value_counts().to_dict(), time.time()-t0)
