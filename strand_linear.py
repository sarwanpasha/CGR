import sys,json,time,numpy as np
from cgr_features import read,feat,norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
T=sys.argv[1]; S,Y=read(T,'train'); St,Yt=read(T,'test'); L0=len(S[0]); t0=time.time()
def rcmap(k):
    n=4**k; idx=np.arange(n); rc=np.zeros(n,np.int64)
    for j in range(k): rc=rc*4+(3-((idx>>(2*j))&3))
    can=np.minimum(idx,rc); u=np.unique(can); rm={int(v):i for i,v in enumerate(u)}
    return np.array([rm[int(v)] for v in can]),len(u)
W={k:rcmap(k) for k in (5,4,3,2)}
out={'task':T}
for name,w in (('plain',None),('canonical',W)):
    X=np.stack([feat(q,L0,None,w) for q in S]); Xt=np.stack([feat(q,L0,None,w) for q in St]); X,Xt=norm(X,Xt)
    m=LogisticRegression(C=0.01,max_iter=300).fit(X,Y)
    v=float(matthews_corrcoef(Yt,m.predict(Xt))); out[name]={'mcc':v,'dim':int(X.shape[1])}
    print('TEST-rc %s %s seed 0 acc 0 mcc %.4f dim %d (%ds)'%(T,name,v,X.shape[1],time.time()-t0),flush=True)
json.dump(out,open('results/rc_%s.json'%T,'w')); print('RC_DONE %s'%T)
