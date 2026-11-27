import numpy as np
from tri import *
beta=-0.3;a=1.0
# 1) asymptotic threshold formula
for d in [0.2,0.5]:
    bbar=np.array([1+d,1,1-d]); Pb=np.linalg.norm(bbar-bbar.mean())
    for eps in [0.01,0.02,0.05,0.1]:
        ts,sig,mono,zs=zero_analysis(a,eps,beta,bbar,n=4000)
        t0=zs[0]['t']; g=np.sqrt(4-2*np.sin(2*t0))
        pred=3*(1-abs(beta)*a)*Pb/(4*abs(beta)*eps*g)
        print('asym',d,eps,len(zs),round(zs[0]['Rc'],4),round(pred,4),round(zs[0]['Rc']/pred,4))
# 2) directed perturbation: monodromy of top eigvec of M=(I-bG)^-T (I-bG)^-1
A=np.array([[0,1,0],[0,0,1],[1,0,0.]])  # cyclic asymmetry
for delta in [0.0,0.05,0.1,0.2]:
  for eps in [0.4]:
    prev=None;first=None;mingap=1e9
    for t in np.linspace(0,2*np.pi,20001):
        Gm=G(*loop(t,a,eps))+delta*A
        B=np.linalg.inv(np.eye(3)-beta*Gm); M=B.T@B
        w,U=np.linalg.eigh(M); u=U[:,-1]; mingap=min(mingap,w[-1]-w[-2])
        if prev is not None and u@prev<0:u=-u
        if first is None:first=u
        prev=u
    print('directed',delta,eps,round(first@prev,6),round(mingap,5))
