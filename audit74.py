import numpy as np
from tri import *
rng=np.random.default_rng(0)
a=1.0
def limit(beta,bbar,tz):
    Pb=np.linalg.norm(bbar-bbar.mean()); g=np.sqrt(4-2*np.sin(2*tz))
    return 3*(1-a*abs(beta))*Pb/(4*abs(beta)*g)
print("A. random heterogeneous status quos, two betas")
for beta in [-0.3,-0.45]:
  for k in range(4):
    bbar=rng.normal(1,0.3,3)
    for eps in [0.04,0.01,0.0025]:
        ts,sig,mono,zs=zero_analysis(a,eps,beta,bbar,n=4000)
        r=[eps*z['Rc']/limit(beta,bbar,z['t']) for z in zs]
        print(beta,k,eps,'zeros',len(zs),'ratio',np.round(r,4))
print("B. large common component, small heterogeneity: bbar=(10.1,10,9.9)")
bbar=np.array([10.1,10,9.9])
for eps in [0.2,0.05,0.01,0.002]:
    ts,sig,mono,zs=zero_analysis(a,eps,-0.3,bbar,n=6000)
    print(eps,'zeros',len(zs),[round(eps*z['Rc']/limit(-0.3,bbar,z['t']),4) for z in zs])
print("C. near stability boundary beta=-0.49")
bbar=np.array([1.2,1,0.8])
for eps in [0.01,0.002]:
    ts,sig,mono,zs=zero_analysis(a,eps,-0.49,bbar,n=4000)
    print(eps,len(zs),[round(eps*z['Rc']/limit(-0.49,bbar,z['t']),4) for z in zs])
print("D. t-bar from limiting eigenvector vs numerical zero")
F=np.array([[-2,1],[0,-np.sqrt(3)]]); e1=np.array([1,-1,0])/2**.5; e2=np.array([1,1,-2])/6**.5
bbar=np.array([1.2,1,0.8]); Pb=bbar-bbar.mean(); c=np.array([Pb@e1,Pb@e2])
tt=np.linspace(0,2*np.pi,200001); th=np.unwrap(np.arctan2(*(F@np.vstack([np.cos(tt),np.sin(tt)]))[::-1]))
phi=th/2+np.pi/2; s=np.cos(phi)*0+ (c[0]*np.cos(phi)+c[1]*np.sin(phi))
z=np.where(np.diff(np.sign(s))!=0)[0]; print('limit zeros',tt[z])
ts,sig,mono,zs=zero_analysis(a,1e-3,-0.3,bbar,n=6000); print('numerical zero eps=1e-3',[z['t'] for z in zs])
print('theta monotone:',np.all(np.diff(th)>0),'total',round(th[-1]-th[0],6))
