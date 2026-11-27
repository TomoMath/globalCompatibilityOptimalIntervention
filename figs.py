import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker
from tri import *
plt.rcParams.update({'font.size':9,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','lines.linewidth':1.6})
COL=['#2a78d6','#eb6834','#1baf7a','#eda100']
beta=-0.3; a=1.0; eps=0.4; bbar=np.array([1.2,1.0,0.8])
ts,sig,mono,zs=zero_analysis(a,eps,beta,bbar,n=6000); t0=zs[0]['t']
# ---------- Figure 1: optimal intervention along loop
C=4.0; R=2.0
tt=np.linspace(0,2*np.pi,1201)
X=[];S=[]
fr=tracked_frames(tt,a,eps,beta)
for t,(mu,U,lam) in zip(tt,fr):
    z,V=solve(mu,U,bbar,R)
    X.append(z[0]); S.append(z[0]@U[:,0])
X=np.array(X)
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1,1.6]})
th=np.linspace(0,2*np.pi,400)
ax[0].plot(a+eps*np.cos(th),a+eps*np.sin(th),color=COL[0])
ax[0].plot([a],[a],'o',color='#0b0b0b',ms=5)
ax[0].annotate('symmetric\nnetwork $p=q=r$',(a,a),xytext=(a-0.33,a+0.06),fontsize=8,color='#52514e')
ax[0].plot([a+eps*np.cos(t0)],[a+eps*np.sin(t0)],'D',color=COL[1],ms=6)
ax[0].annotate('switch',(a+eps*np.cos(t0),a+eps*np.sin(t0)),xytext=(a-0.36,a-0.43),fontsize=8,color='#52514e')
ax[0].annotate('',xy=(a+eps*np.cos(0.5),a+eps*np.sin(0.5)),xytext=(a+eps*np.cos(0.3),a+eps*np.sin(0.3)),arrowprops=dict(arrowstyle='->',color=COL[0]))
ax[0].set_xlabel('link weight $p$ (agents 1–2)'); ax[0].set_ylabel('link weight $q$ (agents 1–3)')
ax[0].set_aspect('equal'); ax[0].set_title('(a) Path of networks ($r=1$)',fontsize=9,loc='left')
for i in range(3):
    xi=X[:,i].copy(); jump=np.where(np.abs(np.diff(xi))>0.5)[0]
    xi_plot=xi.copy(); xi_plot[jump+1]=np.nan
    ax[1].plot(tt,xi_plot,color=COL[i],label=f'agent {i+1}')
ax[1].axvline(t0,color='#52514e',ls='--',lw=0.8)
ax[1].axhline(0,color='#c3c2b7',lw=0.6)
ax[1].set_xlabel('position on the path, $t$'); ax[1].set_ylabel('optimal intervention $x_i^*$')
ax[1].set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]); ax[1].set_xticklabels(['0','$\\pi/2$','$\\pi$','$3\\pi/2$','$2\\pi$'])
ax[1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3)
ax[1].set_title('(b) Welfare-maximising intervention, $C=4$',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig('../fig_path.pdf'); fig.savefig('../fig_path.png',dpi=150)
jumpinfo=dict(t0=t0,left=X[np.searchsorted(tt,t0)-1].tolist(),right=X[np.searchsorted(tt,t0)].tolist(),
              start=X[0].tolist(),end=X[-1].tolist())
print(jumpinfo)
# ---------- Figure 2
tab=json.load(open('cstar_table.json'))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
epss=[0.05,0.1,0.2,0.4,0.8]
for k,d in enumerate([0,0.05,0.2,0.5]):
    ys=[tab[f"{d}_{e}"]['rel'] for e in epss]
    ax[0].plot(epss,ys,marker='o',ms=4,color=COL[k],label=f'$d={d}$')
ax[0].set_xscale('log');ax[0].set_yscale('log')
ax[0].set_xlabel('loop radius $\\varepsilon$'); ax[0].set_ylabel('$\\sqrt{\\underline{C}}\\,/\\,\\|\\bar b\\|$')
ax[0].set_xticks(epss); ax[0].set_xticklabels([str(e) for e in epss]); ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(frameon=False,fontsize=7.5,title='heterogeneity',title_fontsize=7.5)
ax[0].set_title('(a) Budget at which continuity fails',fontsize=9,loc='left')
# cost vs C
Cs=np.linspace(1.02,10,40); share=[]; lossrel=[]
mu,U,lam=welfare_eig(G(*loop(t0,a,eps)),beta)
W0=W(mu,U,bbar,np.zeros(3))
for C in Cs:
    z,V=solve(mu,U,bbar,np.sqrt(C)); D=V-constrained_value(mu,U,bbar,np.sqrt(C))
    share.append(D/(V-W0)); lossrel.append(D/V)
ax[1].plot(Cs/(bbar@bbar),np.array(share)*100,color=COL[0],label='share of gain from intervening')
ax[1].plot(Cs/(bbar@bbar),np.array(lossrel)*100,color=COL[1],ls='--',label='share of total welfare')
ax[1].set_xlabel('budget relative to status quo, $C/\\|\\bar b\\|^2$'); ax[1].set_ylabel('minimum worst-case loss (%)')
ax[1].legend(frameon=False,fontsize=7.5)
ax[1].set_title('(b) Welfare cost of a continuous rule',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig('../fig_threshold_cost.pdf'); fig.savefig('../fig_threshold_cost.png',dpi=150)
json.dump(dict(Cs=Cs.tolist(),share=share,lossrel=lossrel,W0=W0,jump=jumpinfo),open('fig_data.json','w'),indent=1)
for C in [2,4,8]:
    z,V=solve(mu,U,bbar,np.sqrt(C)); D=V-constrained_value(mu,U,bbar,np.sqrt(C)); print(C, D, D/(V-W0), D/V)
