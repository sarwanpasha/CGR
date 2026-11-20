# Proposition 3: canonical (strand-merged) k-mer features. Motif with random orientation -> canonical
# is as good with half the features; fixed orientation -> canonical halves the SNR.
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
L=300; K=4; N=6000; MU=150; rs=np.random.RandomState(1)
MOT=np.array([2,0,3,3]); RC=3-MOT[::-1]          # GATT / AATC
def make(fixed,sig,seed):
    r=np.random.RandomState(seed); A=r.randint(0,4,(N,L)); y=np.arange(N)%2
    for i in np.where(y==1)[0]:
        p=int(round(MU+r.randn()*sig)); p=min(max(p,0),L-K)
        A[i,p:p+K]=MOT if (fixed or r.rand()<.5) else RC
    return A,y
def kfeat(A,canon,R=8):
    n=L-K+1; km=np.zeros((len(A),n),np.int64)
    for j in range(K): km=km*4+A[:,j:j+n]
    if canon:
        idx=np.arange(4**K); rc=np.zeros(4**K,np.int64)
        for j in range(K): rc=rc*4+(3-((idx>>(2*j))&3))
        can=np.minimum(idx,rc); u,inv=np.unique(can,return_inverse=True); km=inv[km]; M=len(u)
    else: M=4**K
    b=np.arange(n)*R//n; col=b*M+km; F=np.zeros((len(A),R*M),np.float32)
    for i in range(len(A)): F[i]=np.bincount(col[i],minlength=R*M)
    return np.log1p(F)
for fixed in (True,False):
    for sig in (5,20):
        A,y=make(fixed,sig,sig); tr=np.arange(N)<N*2//3; out=[]
        for canon in (False,True):
            F=kfeat(A,canon); mu=F[tr].mean(0); sd=F[tr].std(0)+1e-6; F=(F-mu)/sd
            m=LogisticRegression(C=0.05,max_iter=500).fit(F[tr],y[tr]); out.append((roc_auc_score(y[~tr],m.decision_function(F[~tr])),F.shape[1]))
        print('%-18s sigma %2d | strand-specific AUC %.3f (dim %d) | canonical AUC %.3f (dim %d) | change %+.3f'%('fixed orientation' if fixed else 'random orientation',sig,out[0][0],out[0][1],out[1][0],out[1][1],out[1][0]-out[0][0]))
