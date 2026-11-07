"""Build /tmp/a2data/sample = one TRAIN month with realistic open pool (items open at month start or issued in month)."""
import os, glob, re, pandas as pd, sys
M=sys.argv[1] if len(sys.argv)>1 else '2025-05'
D="/Users/pedroantonperez/Desktop/Claude Hackathon/data"; O="/tmp/a2data"; P=f"{O}/sample"
os.makedirs(P,exist_ok=True)
if not os.path.exists(f"{O}/masters"): os.symlink(f"{D}/masters",f"{O}/masters")
start=M+'-01'; end=M+'-31'
pool=pd.read_pickle('/tmp/a2_pool.pkl').set_index('item_id')
keep=set(pool.index[(pool.issue_date<=end)&(pool.settle>=start)])
print('pool items',len(keep))
for f in ['ar_invoices','ap_invoices','credit_notes']:
    x=pd.read_csv(f"{D}/train/{f}.csv.gz",dtype=str,keep_default_na=False); x[x.item_id.isin(keep)].to_csv(f"{P}/{f}.csv.gz",index=False)
b=pd.read_csv(f"{D}/train/bank_transactions.csv.gz",dtype=str,keep_default_na=False); b=b[b.booking_date.str[:7]==M]; b.to_csv(f"{P}/bank_transactions.csv.gz",index=False)
g=pd.read_pickle('/tmp/a2_gtrain.pkl'); g[g.bank_txn_id.isin(set(b.bank_txn_id))].drop(columns=['yr']).to_csv(f"{P}/golden_allocations.csv.gz",index=False)
y,mo=map(int,M.split('-')); months={f"{y}{mo:02d}", f"{y if mo>1 else y-1}{(mo-2)%12+1:02d}", f"{y if mo<12 else y+1}{mo%12+1:02d}"}
for d in ['payment_runs','direct_debit_batches','card_settlements','cheque_deposits']:
    os.makedirs(f"{P}/{d}",exist_ok=True)
    for f in glob.glob(f"{D}/train/{d}/*.csv"):
        if any(mm in os.path.basename(f) for mm in months):
            dst=f"{P}/{d}/{os.path.basename(f)}"
            if not os.path.exists(dst): os.symlink(f,dst)
R=f"{P}/remittance_advices"; os.makedirs(R,exist_ok=True)
idx=pd.read_csv(f"{D}/train/remittance_advices/index.csv",dtype=str)
idx[idx.received_date.str[:7].isin({f"{M}", f"{y}-{mo%12+1:02d}"})|((idx.received_date>=f"{y}-{(mo-2)%12+1:02d}-20")&(idx.received_date<start))].to_csv(f"{R}/index.csv",index=False)
for e in os.listdir(f"{D}/train/remittance_advices"):
    if os.path.isdir(f"{D}/train/remittance_advices/{e}") and not os.path.exists(f"{R}/{e}"): os.symlink(f"{D}/train/remittance_advices/{e}",f"{R}/{e}")
print('bank',len(b),'golden',len(g[g.bank_txn_id.isin(set(b.bank_txn_id))]))
