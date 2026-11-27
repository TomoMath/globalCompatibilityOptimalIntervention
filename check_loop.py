import numpy as np
from tri import *
beta=-0.3; a=1.0
def maxjump(eps,bbar,C,n=40000):
    tt=np.linspace(0,2*np.pi,n+1); X=[]
    for t in tt:
        mu,U,lam=welfare_eig(G(*loop(t,a,eps)),beta)
        z,V=solve(mu,U,bbar,np.sqrt(C)); X.append(z[0])
    X=np.array(X); d=np.linalg.norm(np.diff(X,axis=0),axis=1)
    return d.max(), tt[d.argmax()]
bbar=np.array([1.05,1.0,0.95])
ts,sig,mono,zs=zero_analysis(a,0.4,beta,bbar,n=6000)
print([(round(z['t'],4),z['Rc']**2) for z in zs])
for C in [0.04,0.052,0.053,0.056,0.058,0.06,0.2,0.46]:
    print(C, maxjump(0.4,bbar,C))
