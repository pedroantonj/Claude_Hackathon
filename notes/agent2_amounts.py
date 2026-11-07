import sys,pandas as pd,numpy as np
phase=sys.argv[1] if len(sys.argv)>1 else 'sample'
m=pd.read_pickle(f'/tmp/a2_m_{phase}.pkl')
m['sgn']=np.where(m.kind.isin(['ARC','APC']),-1,1)
m['net']=m.amount_x*m.sgn
m['fulltot']=(abs(m.amount_x-m.total_amount)<=0.01)
L=m.groupby('bank_txn_id').agg(seg=('segment','first'),bank=('amount_y','first'),net=('net','sum'),gross=('amount_x','sum'),n=('item_id','size'),
   samecur=('currency_it', lambda s: True), parties=('party_id','nunique'),fulltot=('fulltot','mean'),
   ccy=('currency','first'),iccy=('currency_it','first'))
L['abs']=L.bank.abs()
L['exact_net']=(L['abs']-L.net).abs()<=0.011
L['exact_gross']=(L['abs']-L.gross).abs()<=0.011
L['ccy_diff']=L.ccy!=L.iccy
r=L.groupby('seg').agg(lines=('n','size'),n_mean=('n','mean'),n_max=('n','max'),exact_net=('exact_net','mean'),exact_gross=('exact_gross','mean'),multi_party=('parties',lambda s:(s>1).mean()),item_full=('fulltot','mean'),ccydiff=('ccy_diff','mean')).sort_values('lines',ascending=False)
print(r.round(3).to_string())
L.to_pickle(f'/tmp/a2_L_{phase}.pkl')
