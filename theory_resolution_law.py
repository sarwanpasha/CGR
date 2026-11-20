import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
# Theorem 1: d2(w) = p(w)^2 / (lam*w), p(w)=2Phi(w/2s)-1. With u=w/(2s): g(u)=(2Phi(u)-1)^2/u
g=lambda u:(2*norm.cdf(u)-1)**2/u
f=lambda u:4*u*norm.pdf(u)-(2*norm.cdf(u)-1)
u=brentq(f,0.1,5); print('u* = %.4f  ->  w* = %.4f sigma ; g(u*) = %.4f'%(u,2*u,g(u)))
# exact (finite-signal) version: d2 = p^2/(lam w + p(1-p)) ; show w* depends weakly on lam*sigma
for ls in (0.001,0.01,0.1,1.0):
    h=lambda w: -( (2*norm.cdf(w/2)-1)**2/(ls*w+(2*norm.cdf(w/2)-1)*(2-2*norm.cdf(w/2))) )
    r=minimize_scalar(h,bounds=(0.01,50),method='bounded'); print('  lam*sigma=%-6g  w*/sigma=%.3f'%(ls,r.x))
# Theorem 2: geometric pyramid with ratio rho: worst-case efficiency over sigma
for rho in (2,4,8,16):
    t=np.linspace(-0.5,0.5,2001)*np.log(rho)
    eta=min(g(u*np.exp(x))/g(u) for x in t)
    print('rho=%2d  worst-case SNR efficiency of best pyramid level = %.3f'%(rho,eta))
