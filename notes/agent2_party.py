"""Party identification: IDF token matcher over names (master + invoice-name history) + IBAN maps."""
import re, unicodedata, math, pandas as pd, numpy as np
from collections import defaultdict, Counter
M="/Users/pedroantonperez/Desktop/Claude Hackathon/data/masters"
def fold(s): return unicodedata.normalize('NFKD',str(s).upper()).encode('ascii','ignore').decode()
STOP=set('SL SA SAS SARL EURL GMBH CO KG EK BV LTD LLC INC CORP PTY SPA SRL SNC SCOOP SLU DE DEL LA LOS LAS Y E ET DU DES THE AND LIMITED CV RL SAU LDA LTDA ME OF'.split())
def toks(s): return [t for t in re.findall(r'[A-Z0-9]{2,}',fold(s)) if t not in STOP]
class PartyIndex:
    def __init__(self, names):  # names: DataFrame entity, party_id, name
        self.post=defaultdict(lambda: defaultdict(set)); self.pref=defaultdict(lambda: defaultdict(set)); self.ptoks={}
        for e,p,n in names[['entity','party_id','name']].itertuples(index=False):
            ts=set(toks(n))
            if not ts: continue
            self.ptoks.setdefault((e,p),set()).update(ts)
            for t in ts:
                self.post[e][t].add(p)
                for k in range(4,len(t)): self.pref[e][t[:k]].add(p)
        self.df={e:{t:len(v) for t,v in d.items()} for e,d in self.post.items()}
        self.np={e:len({p for (ee,p) in self.ptoks if ee==e}) for e in self.post}
    def query(self, e, text, k=3):
        if e not in self.post: return []
        N=self.np[e]; sc=Counter()
        tt=toks(text)
        for i,t in enumerate(tt):
            ps=self.post[e].get(t)
            if ps and len(ps)<=500:
                w=math.log(N/len(ps))
                for p in ps: sc[p]+=w
            elif not ps and len(t)>=4:   # truncated token -> prefix
                ps=self.pref[e].get(t)
                if ps and len(ps)<=200:
                    w=0.7*math.log(N/len(ps))
                    for p in ps: sc[p]+=w
        return sc.most_common(k)
def build_names(it):
    cust=pd.read_csv(f"{M}/customers.csv",dtype=str,keep_default_na=False)
    sup=pd.read_csv(f"{M}/suppliers.csv",dtype=str,keep_default_na=False)
    n=pd.concat([cust[['entity','customer_id','name']].rename(columns={'customer_id':'party_id'}),
                 cust[cust.trade_name!=''][['entity','customer_id','trade_name']].rename(columns={'customer_id':'party_id','trade_name':'name'}),
                 sup[['entity','supplier_id','name']].rename(columns={'supplier_id':'party_id'}),
                 it[['entity','party_id','party_name']].rename(columns={'party_name':'name'}).drop_duplicates()])
    return n.drop_duplicates()
if __name__=='__main__':
    import time; t0=time.time()
    m=pd.read_pickle('/tmp/a2_m_train.pkl'); b=pd.read_pickle('/tmp/a2_b_train.pkl'); it=pd.read_pickle('/tmp/a2_ittrain.pkl')
    g=pd.read_pickle('/tmp/a2_gtrain.pkl')
    # names: master + invoice names of items issued before 2025 only (avoid leakage of future renames? invoice names are inputs anyway)
    idx=PartyIndex(build_names(it[it.kind=='ARI'])); print('index',time.time()-t0)
    ar=m[m.kind=='ARI']
    L=ar.groupby('bank_txn_id').agg(seg=('segment','first'),party=('party_id',lambda s:s.mode()[0]),ent=('ientity','first'),T=('T','first'),cp=('counterparty_name','first'),iban=('counterparty_account','first'),fmt=('statement_format','first'),bent=('entity','first'))
    L=L[~L.seg.isin(['ar_dd_batch','ar_card_settlement','ar_cheque_deposit','ar_dd_returned'])]
    L=L.sample(min(len(L),40000),random_state=0)
    res=[idx.query(e,cp+' '+T) for e,cp,T in zip(L.ent,L.cp,L['T'])]
    L['top1']=[r[0][0] if r else None for r in res]
    L['margin']=[(r[0][1]-(r[1][1] if len(r)>1 else 0)) if r else 0 for r in res]
    L['ok']=L.top1==L.party
    L['top3']=[p in [x[0] for x in r] for p,r in zip(L.party,res)]
    # IBAN: master
    cust=pd.read_csv(f"{M}/customers.csv",dtype=str,keep_default_na=False)
    ib=cust[cust.iban!=''].set_index('iban').customer_id
    L['iban_master']=L.iban.map(ib)
    # learned IBAN map from history (golden lines before 2025)
    print('ok overall',L.ok.mean(),'top3',L.top3.mean(),'entity mismatch (bank vs item)',(L.ent!=L.bent).mean())
    r=L.groupby('seg').agg(n=('ok','size'),top1=('ok','mean'),top3=('top3','mean'),has_iban=('iban',lambda s:(s!='').mean()))
    r2=L[L.iban!=''].assign(ibok=lambda d:d.iban_master==d.party).groupby('seg').ibok.mean()
    print(r.join(r2).sort_values('n',ascending=False).round(3).to_string())
    print(L.groupby('fmt').ok.mean())
    for th in [0,2,4,6,8]:
        s=L[L.margin>=th]; print('margin>=',th,'coverage',round(len(s)/len(L),3),'acc',round(s.ok.mean(),4))
    print(L[~L.ok].head(15)[['seg','party','top1','cp','T']].to_string())
    L.to_pickle('/tmp/a2_party.pkl')
    print(time.time()-t0)
