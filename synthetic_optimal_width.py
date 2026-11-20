# Validate Theorems 1-2 on simulated DNA (real k-mer counting, overlaps included).
import numpy as np, json, time
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
rs=np.random.RandomState(0); L=1000; MOT=np.array([2,0,3,0]); K=len(MOT); MU=L//2; N=6000
lam=(1/4**K)   # background occurrence rate of a fixed 4-mer per position
def make(n,sig,seed):
    r=np.random.RandomState(seed); A=r.randint(0,4,(n,L)); y=np.arange(n)%2
    for i in np.where(y==1)[0]:
        p=int(round(MU+r.randn()*sig)); p=min(max(p,0),L-K); A[i,p:p+K]=MOT
    return A,y
def occ(A):   # occurrence start positions -> boolean matrix (n, L-K+1)
    n=L-K+1; m=np.ones((len(A),n),bool)
    for j in range(K): m&=A[:,j:j+n]==MOT[j]
    return m
def pred_d2(w,sig):
    p=2*norm.cdf(w/(2*sig))-1 if sig>0 else 1.0
    return p**2/(lam*w+p*(1-p)/2+1e-12)
W=np.unique(np.round(np.logspace(0,np.log10(L),40)).astype(int))
res={}
for sig in (2,5,10,20,40,80,160):
    A,y=make(N,sig,sig); O=occ(A); cs=np.concatenate([np.zeros((len(O),1)),np.cumsum(O,1)],1)
    emp=[];auc=[];prd=[]
    for w in W:
        a=max(0,MU-w//2); b=min(O.shape[1],MU-w//2+w); c=cs[:,b]-cs[:,a]
        d=(c[y==1].mean()-c[y==0].mean())**2/(0.5*(c[y==1].var()+c[y==0].var())+1e-12)
        emp.append(d); auc.append(roc_auc_score(y,c)); prd.append(pred_d2(w,sig))
    emp=np.array(emp); prd=np.array(prd)
    we=W[np.argmax(emp)]; wp=W[np.argmax(prd)]
    rho=np.corrcoef(np.log(emp+1e-9),np.log(prd))[0,1]
    # pyramids on a fixed (unaligned) grid starting at 0: ratio 4 vs ratio 2, logistic regression on all bin counts
    out={}
    for name,Rs in (('ratio4',[1,4,16,64,256,L-K+1]),('ratio2',[1,2,4,8,16,32,64,128,256,512,L-K+1])):
        F=[]
        for R in Rs:
            edges=np.linspace(0,O.shape[1],R+1).astype(int); F.append(cs[:,edges[1:]]-cs[:,edges[:-1]])
        F=np.log1p(np.concatenate(F,1)); tr=np.arange(N)<N*2//3
        mu=F[tr].mean(0); sd=F[tr].std(0)+1e-6; F=(F-mu)/sd
        m=LogisticRegression(C=0.1,max_iter=500).fit(F[tr],y[tr]); out[name]=roc_auc_score(y[~tr],m.decision_function(F[~tr]))
    best_auc=max(auc)
    res[sig]={'w_emp':int(we),'w_pred':int(wp),'w_pred_asym':2.8*sig,'logcorr':float(rho),'oracle_auc':float(best_auc),**{k:float(v) for k,v in out.items()}}
    print('sigma %3d | best w: measured %4d  predicted %4d  (2.8s=%5.0f) | log-corr pred vs meas %.3f | AUC oracle bin %.3f  pyr-ratio4 %.3f  pyr-ratio2 %.3f'%(sig,we,wp,2.8*sig,rho,best_auc,out['ratio4'],out['ratio2']),flush=True)
json.dump(res,open('synth2.json','w'))
