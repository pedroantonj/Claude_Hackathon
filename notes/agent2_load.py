"""Shared loader for agent2 analysis. usage: from agent2_load import load; d=load('sample')"""
import pandas as pd, numpy as np, re, os
D="/Users/pedroantonperez/Desktop/Claude Hackathon/data"
def nrm(s): return re.sub(r'[^A-Z0-9]','',str(s).upper()) if isinstance(s,str) else ''
def load(phase='sample', since=None):
    p=f"{D}/{phase}"
    b=pd.read_csv(f"{p}/bank_transactions.csv.gz",dtype=str,keep_default_na=False)
    b['amount']=b.amount.astype(float)
    if since: b=b[b.booking_date>=since]
    g=pd.read_csv(f"{p}/golden_allocations.csv.gz",dtype=str,keep_default_na=False); g["amount"]=pd.to_numeric(g.amount)
    g=g[g.bank_txn_id.isin(set(b.bank_txn_id))]
    ar=pd.read_csv(f"{p}/ar_invoices.csv.gz",dtype=str,keep_default_na=False)
    ap=pd.read_csv(f"{p}/ap_invoices.csv.gz",dtype=str,keep_default_na=False)
    cn=pd.read_csv(f"{p}/credit_notes.csv.gz",dtype=str,keep_default_na=False)
    it=pd.concat([
      ar.assign(kind='ARI',party_id=ar.customer_id,party_name=ar.customer_name,docno=ar.document_number)[['item_id','kind','entity','party_id','party_name','docno','issue_date','due_date','currency','total_amount','payment_terms','customer_po','payment_method']],
      ap.assign(kind='API',party_id=ap.supplier_id,party_name=ap.supplier_name,docno=ap.supplier_invoice_number,customer_po=ap.our_po)[['item_id','kind','entity','party_id','party_name','docno','issue_date','due_date','currency','total_amount','payment_terms','customer_po','payment_method']],
      cn.assign(kind=cn.item_id.str[:3],docno=cn.document_number,due_date='',payment_terms='',customer_po=cn.related_document,payment_method='')[['item_id','kind','entity','party_id','party_name','docno','issue_date','due_date','currency','total_amount','payment_terms','customer_po','payment_method']]])
    it['total_amount']=it.total_amount.astype(float)
    return b,g,it
