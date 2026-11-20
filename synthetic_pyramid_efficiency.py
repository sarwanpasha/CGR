import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
exec(open('synthetic_optimal_width.py').read().split('W=np.unique')[0])
def dsq(c,y): return (c[y==1].mean()-c[y==0].mean())**2/(0.5*(c[y==1].var()+c[y==0].var())+1e-12)
for sig in (5,20,40,80,160):
    A,y=make(N,sig,sig); O=occ(A); n=O.shape[1]; cs=np.concatenate([np.zeros((len(O),1)),np.cumsum(O,1)],1)
    opt=max(dsq(cs[:,min(n,MU-w//2+w)]-cs[:,max(0,MU-w//2)],y) for w in range(1,n,1))
    row=[]
    for name,Rs,shift in (('ratio4',[1,4,16,64,256,n],False),('ratio4+half-shift',[1,4,16,64,256,n],True),('ratio2',[1,2,4,8,16,32,64,128,256,512,n],False)):
        best=0; tr=np.arange(N)<N*2//3; lv=[]
        for R in Rs:
            for off in ((0,0.5) if shift and R>1 else (0,)):
                e=np.clip((np.arange(R+1)+off)*n/R,0,n).astype(int); e=np.unique(np.r_[0,e,n])
                C=cs[:,e[1:]]-cs[:,e[:-1]]
                for i in range(C.shape[1]):
                    best=max(best,dsq(C[:,i],y))
                    if i+1<C.shape[1]: best=max(best,dsq(C[:,i]+C[:,i+1],y))
                F=np.log1p(C); mu=F[tr].mean(0); sd=F[tr].std(0)+1e-6; F=(F-mu)/sd
                m=LogisticRegression(C=0.1,max_iter=500).fit(F[tr],y[tr]); lv.append(roc_auc_score(y[~tr],m.decision_function(F[~tr])))
        row.append('%s eff %.2f bestlevelAUC %.3f'%(name,best/opt,max(lv)))
    print('sigma %3d | '%sig+' | '.join(row),flush=True)
