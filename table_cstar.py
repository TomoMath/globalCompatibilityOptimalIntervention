import numpy as np, json
from tri import *
beta=-0.3; a=1.0
res={}
for d in [0,0.05,0.2,0.5]:
  bbar=np.array([1+d,1.0,1-d])
  for eps in [0.05,0.1,0.2,0.4,0.8]:
    ts,sig,mono,zs=zero_analysis(a,eps,beta,bbar,n=6000)
    # dedupe zeros mod 2pi
    u=[]
    for z in zs:
        if not any(min(abs(z['t']-w['t']),2*np.pi-abs(z['t']-w['t']))<1e-6 for w in u): u.append(z)
    Rc2=[z['Rc']**2 for z in u]
    res[f"{d}_{eps}"]=dict(nz=len(u),Cobs=min(Rc2),Cstar=max(Rc2),b2=float(bbar@bbar),
        rel=float(np.sqrt(min(Rc2))/np.linalg.norm(bbar)), tz=[z['t'] for z in u])
    print(d,eps,len(u),round(min(Rc2),4),round(max(Rc2),4),round(np.sqrt(min(Rc2))/np.linalg.norm(bbar),3))
json.dump(res,open('cstar_table.json','w'),indent=1)
