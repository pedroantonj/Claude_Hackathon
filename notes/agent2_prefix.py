"""Truncation-aware name matcher: longest common prefix of (text from word start) vs sorted normalized names."""
import bisect, re, pandas as pd
from collections import defaultdict
from agent2_party import fold, build_names
def nz(s): return re.sub(r'[^A-Z0-9]','',fold(s))
class PrefixIndex:
    def __init__(self,names):
        self.ent={}
        for e,df in names.groupby('entity'):
            d=defaultdict(set)
            for p,n in zip(df.party_id,df.name):
                k=nz(n)
                if len(k)>=4: d[k].add(p)
            keys=sorted(d); self.ent[e]=(keys,d)
    def query(self,e,text,minlcp=8):
        if e not in self.ent: return set(),0
        keys,d=self.ent[e]
        words=re.findall(r'[A-Z0-9]+',fold(text))
        best=0; bestpref=None
        for i in range(len(words)):
            s=''.join(words[i:i+8])[:48]
            if len(s)<4: continue
            j=bisect.bisect_left(keys,s)
            for k in (j-1,j):
                if 0<=k<len(keys):
                    a=keys[k]; l=0
                    while l<min(len(a),len(s)) and a[l]==s[l]: l+=1
                    if l>best: best=l; bestpref=s[:l]
        if best<minlcp: return set(),best
        j=bisect.bisect_left(keys,bestpref); out=set()
        while j<len(keys) and keys[j].startswith(bestpref):
            # require: name fully matched OR text truncated here (prefix covers >=60% of name or >=12 chars)
            if len(keys[j])==best or best>=12 or best>=0.6*len(keys[j]): out|=d[keys[j]]
            j+=1
        return out,best
if __name__=='__main__':
    L=pd.read_pickle('/tmp/a2_party.pkl'); it=pd.read_pickle('/tmp/a2_ittrain.pkl')
    px=PrefixIndex(build_names(it[it.kind=='ARI']))
    import time; t0=time.time()
    r=[px.query(e,(cp or '')+' '+T) for e,cp,T in zip(L.ent,L.cp,L['T'])]
    print('time per 1k',(time.time()-t0)/len(L)*1000)
    L['rec']=[p in c for p,(c,_) in zip(L.party,r)]; L['sz']=[len(c) for c,_ in r]
    print('recall',L.rec.mean().round(3),'size median',L.sz.median(),'p90',L.sz.quantile(.9),'unique&correct',((L.sz==1)&L.rec).mean().round(3))
    print(L.groupby('fmt').agg(rec=('rec','mean'),sz=('sz','median')))
    print(L.groupby('seg').agg(n=('rec','size'),rec=('rec','mean')).sort_values('n',ascending=False).head(12).round(3).T.to_string())
    for r_ in L[~L.rec].sample(10,random_state=2).itertuples():
        print(r_.seg,'|',r_.cp,'|',r_.T[:100],'| TRUE:',it[it.party_id==r_.party].party_name.unique()[:2])
