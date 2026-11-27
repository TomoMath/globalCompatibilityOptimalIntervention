import numpy as np, json
from tri import *
beta=-0.3; a=1.0; eps=0.4
bbar=np.array([1.2,1.0,0.8])
ts,sig,mono,zs=zero_analysis(a,eps,beta,bbar,n=6000)
t0=zs[0]['t']; print('t0',t0,'Cobs',zs[0]['Rc']**2,'mono',mono)
out={}
for C in [1.5,2.0,4.0,8.0]:
    R=np.sqrt(C)
    tt=np.linspace(0,2*np.pi,2001)
    Vs=[];Ds=[];bnd=[]
    for t in tt:
        mu,U,lam=welfare_eig(G(*loop(t,a,eps)),beta)
        zsol,V=solve(mu,U,bbar,R)
        Cv=constrained_value(mu,U,bbar,R)
        hc=U.T@bbar; hperp=np.linalg.norm(hc[1:])
        Vs.append(V);Ds.append(V-Cv);bnd.append(mu[0]*C-mu[1]*(R+hperp)**2)
    Vs=np.array(Vs);Ds=np.array(Ds);bnd=np.array(bnd)
    assert np.all(Ds>=bnd-1e-9)
    # Delta at switch point
    mu,U,lam=welfare_eig(G(*loop(t0,a,eps)),beta)
    zsol,V0=solve(mu,U,bbar,R)
    D0=V0-constrained_value(mu,U,bbar,R)
    # path check: great-circle arcs z+ -> w* -> z- ; compute min W along
    hc=U.T@bbar
    # constrained optimum w*
    m2,h2=mu[1:],hc[1:]
    from scipy.optimize import minimize
    best=None
    for th in np.linspace(0,2*np.pi,721):
        w=R*np.array([0,np.cos(th),np.sin(th)])
        v=W(mu,np.eye(3),hc,w)
        if best is None or v>best[0]: best=(v,w)
    wstar=best[1]
    def zc(z): return U.T@z
    zp,zm=[zc(z) for z in zsol] if len(zsol)==2 else (None,None)
    def arc(x,y,n=400):
        x=x/np.linalg.norm(x); y=y/np.linalg.norm(y)
        om=np.arccos(np.clip(x@y,-1,1))
        return [R*(np.sin((1-s)*om)*x+np.sin(s*om)*y)/np.sin(om) for s in np.linspace(0,1,n)]
    pathmin=None
    if zp is not None:
        pts=arc(zp,wstar)+arc(wstar,zm)
        pathmin=min(W(mu,np.eye(3),hc,p) for p in pts)
    out[C]=dict(Vmin=float(Vs.min()),Vmax=float(Vs.max()),minDelta=float(Ds.min()),argminDelta=float(tt[Ds.argmin()]),
                Delta_t0=float(D0),V_t0=float(V0),path_loss=None if pathmin is None else float(V0-pathmin),
                rel_lower=float(Ds.min()/Vs.max()), rel_upper=float(D0/V0), gapC=float(min(Ds)/C))
    print(C,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[C].items()})
json.dump(dict(t0=t0,res=out),open('cost.json','w'),indent=1)
