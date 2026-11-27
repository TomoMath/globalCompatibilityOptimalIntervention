"""Numerical analysis of the three-agent example (revision of JME-D-26-00615).

Network G(p,q,r), welfare matrix M=(I-beta G)^{-2}, budget ||x||^2<=C, status quo bbar.
Loop gamma_eps(t)=(a+eps cos t, a+eps sin t, a).
"""
import numpy as np
from scipy.optimize import brentq

def G(p, q, r):
    return np.array([[0, p, q], [p, 0, r], [q, r, 0.]])

def loop(t, a=1.0, eps=0.5):
    return a + eps*np.cos(t), a + eps*np.sin(t), a

def welfare_eig(Gm, beta):
    """Eigen-decomposition of M=(I-beta G)^{-2}, sorted mu1>mu2>mu3."""
    lam, U = np.linalg.eigh(Gm)
    mu = (1 - beta*lam)**-2
    idx = np.argsort(-mu)
    return mu[idx], U[:, idx], lam[idx]

def tracked_frames(ts, a, eps, beta):
    """Eigenbases along the loop with u1 (dominant) continued continuously."""
    out = []
    prev = None
    for t in ts:
        mu, U, lam = welfare_eig(G(*loop(t, a, eps)), beta)
        if prev is not None:
            for k in range(3):
                if U[:, k] @ prev[:, k] < 0:
                    U[:, k] = -U[:, k]
        out.append((mu, U.copy(), lam))
        prev = U
    return out

def solve(mu, U, h, R, tol=1e-12):
    """max_{||z||<=R} (h+z)' M (h+z), M=U diag(mu) U'. Returns list of optimal z, value."""
    hc = U.T @ h
    if abs(hc[0]) > tol*max(1, np.linalg.norm(h)):
        phi = lambda L: np.sum((mu*hc/(L-mu))**2) - R**2
        lo = mu[0] + 1e-15*mu[0]
        # bracket
        hi = mu[0] + 1.0
        while phi(hi) > 0:
            hi = mu[0] + 2*(hi-mu[0])
        lo_ = mu[0] + (hi-mu[0])*1e-14
        while phi(lo_) < 0:
            lo_ = mu[0] + (lo_-mu[0])*1e-3
        L = brentq(phi, lo_, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
        zc = mu*hc/(L-mu)
        zs = [U @ zc]
    else:
        z0c = np.zeros(3)
        z0c[1:] = mu[1:]*hc[1:]/(mu[0]-mu[1:])
        Rc = np.linalg.norm(z0c)
        if R > Rc:
            aa = np.sqrt(R**2 - Rc**2)
            zs = [U @ (z0c + np.array([aa, 0, 0])), U @ (z0c - np.array([aa, 0, 0]))]
        else:
            phi = lambda L: np.sum((mu[1:]*hc[1:]/(L-mu[1:]))**2) - R**2
            hi = mu[0] + 1.0
            while phi(hi) > 0:
                hi = mu[0] + 2*(hi-mu[0])
            L = brentq(phi, mu[0], hi) if R < Rc else mu[0]
            zc = np.zeros(3); zc[1:] = mu[1:]*hc[1:]/(L-mu[1:])
            zs = [U @ zc]
    M = U @ np.diag(mu) @ U.T
    val = max((h+z) @ M @ (h+z) for z in zs)
    return zs, val

def W(mu, U, h, z):
    M = U @ np.diag(mu) @ U.T
    return (h+z) @ M @ (h+z)

def constrained_value(mu, U, h, R):
    """max over ||z||<=R with z orthogonal to u1 (zero dominant component)."""
    hc = U.T @ h
    m2, h2 = mu[1:], hc[1:]
    # 2-dim trust region problem max (h2+w)' diag(m2) (h2+w), ||w||<=R
    U2 = np.eye(2)
    if abs(h2[0]) > 1e-14:
        phi = lambda L: np.sum((m2*h2/(L-m2))**2) - R**2
        hi = m2[0] + 1.0
        while phi(hi) > 0:
            hi = m2[0] + 2*(hi-m2[0])
        lo = m2[0] + (hi-m2[0])*1e-14
        while phi(lo) < 0:
            lo = m2[0] + (lo-m2[0])*1e-3
        L = brentq(phi, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
        w = m2*h2/(L-m2)
    else:
        w0 = np.array([0, m2[1]*h2[1]/(m2[0]-m2[1])])
        r0 = np.linalg.norm(w0)
        w = w0 + np.array([np.sqrt(max(R**2-r0**2, 0)), 0]) if R > r0 else w0*R/max(r0, 1e-300)
    return mu[0]*hc[0]**2 + np.sum(m2*(h2+w)**2)

def zero_analysis(a, eps, beta, bbar, n=20000):
    """Zeros of sigma(t)=<bbar,u1(t)> on the loop, crossing type and critical radii."""
    ts = np.linspace(0, 2*np.pi, n+1)
    fr = tracked_frames(ts, a, eps, beta)
    sig = np.array([U[:, 0] @ bbar for (mu, U, lam) in fr])
    monodromy = np.sign(fr[0][1][:, 0] @ fr[-1][1][:, 0])
    zeros = []
    for k in range(n):
        if sig[k] == 0 or sig[k]*sig[k+1] < 0:
            f = lambda t: _sig_at(t, ts[k], fr[k][1][:, 0], a, eps, beta, bbar)
            tz = brentq(f, ts[k], ts[k+1], xtol=1e-14)
            mu, U, lam = welfare_eig(G(*loop(tz, a, eps)), beta)
            hc = U.T @ bbar
            z0c = mu[1:]*hc[1:]/(mu[0]-mu[1:])
            zeros.append(dict(t=tz, Rc=np.linalg.norm(z0c), crossing=True,
                              gap=mu[0]-mu[1]))
    return ts, sig, monodromy, zeros

def _sig_at(t, tref, uref, a, eps, beta, bbar):
    mu, U, lam = welfare_eig(G(*loop(t, a, eps)), beta)
    u = U[:, 0] if U[:, 0] @ uref >= 0 else -U[:, 0]
    return u @ bbar
