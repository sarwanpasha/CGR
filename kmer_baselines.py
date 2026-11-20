import sys,json,time,numpy as np
from itertools import combinations
from cgr_features import read,LUT,kmers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
T,WH=sys.argv[1],sys.argv[2]; S,Y=read(T,'train'); St,Yt=read(T,'test'); t0=time.time()
MK=[m for m in combinations(range(6),4)]
def gkm(q):
    a=LUT[np.frombuffer(q.encode(),np.uint8)]; n=len(a)-6+1; v=np.zeros(len(MK)*256,np.float32)
    for mi,m in enumerate(MK):
        km=np.zeros(n,np.int64); bd=np.zeros(n,bool)
        for j in m: km=km*4+np.minimum(a[j:j+n],3); bd|=a[j:j+n]>3
        v[mi*256:(mi+1)*256]=np.bincount(km[~bd],minlength=256)
    return np.log1p(v)
def kc(q,k,w=None):
    a=LUT[np.frombuffer(q.encode(),np.uint8)]; km,bad,n=kmers(a,k)
    v=np.bincount(km[~bad],minlength=4**k).astype(np.float32)
    return np.log1p(v*w if w is not None else v)
NW=np.array([1.0,1.17,1.17,1.0])
if WH=='gkm': f=gkm
elif WH=='fcgr': f=lambda q: kc(q,6)
else:
    idx=np.arange(4096); w=np.ones(4096)
    for j in range(6): w=w*NW[(idx>>(2*j))&3]
    w/=w.mean(); f=lambda q: kc(q,6,w)
X=np.stack([f(q) for q in S]); Xt=np.stack([f(q) for q in St])
mu=X.mean(0); sd=X.std(0)+1e-6; X=(X-mu)/sd; Xt=(Xt-mu)/sd
m=LogisticRegression(C=0.01,max_iter=400,class_weight='balanced').fit(X,Y); p=m.predict(Xt)
r={'task':T,'method':WH,'dim':int(X.shape[1]),'acc':float(accuracy_score(Yt,p)),'f1':float(f1_score(Yt,p,average='macro'))}
if len(set(Y))==2: r['mcc']=float(matthews_corrcoef(Yt,p))
json.dump(r,open('results/b2_%s_%s.json'%(T,WH),'w'))
print('TEST-%s %s base seed 0 acc %.4f mcc %.4f dim %d (%ds)'%(WH,T,r['acc'],r.get('mcc',0),r['dim'],time.time()-t0))
print('B2_DONE %s %s'%(T,WH))
