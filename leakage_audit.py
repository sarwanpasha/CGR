import sys,json,time,numpy as np
from cgr_features import read,feat,norm,LUT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
T=sys.argv[1]; THR=float(sys.argv[2]) if len(sys.argv)>2 else 0.5; K=8; NH=64
S,Y=read(T,'train'); St,Yt=read(T,'test'); L0=len(S[0]); t0=time.time()
rs=np.random.RandomState(0); A=rs.randint(1,2**31-1,NH); B=rs.randint(0,2**31-1,NH); P=(1<<31)-1
def sig(q):
    a=LUT[np.frombuffer(q.encode(),np.uint8)]; n=len(a)-K+1
    km=np.zeros(n,np.int64); bad=np.zeros(n,bool)
    for j in range(K): km=km*4+np.minimum(a[j:j+n],3); bad|=a[j:j+n]>3
    u=np.unique(km[~bad])
    return np.zeros(NH,np.int64) if len(u)==0 else ((A[:,None]*u[None,:]+B[:,None])%P).min(1)
G=np.stack([sig(q) for q in S]); Gt=np.stack([sig(q) for q in St]); print('sigs %ds'%(time.time()-t0),flush=True)
best=np.zeros(len(St))
for i in range(0,len(Gt),200): best[i:i+200]=(Gt[i:i+200,None,:]==G[None,:,:]).mean(2).max(1)
R={'task':T,'n_train':len(S),'n_test':len(St),'median_nn_jaccard':float(np.median(best)),'max_nn_jaccard':float(best.max())}
for t in (0.9,0.8,0.7,0.6,0.5): R['frac_ge_%.1f'%t]=float((best>=t).mean())
print('%s median NN jaccard %.3f  frac>=0.5 %.3f  frac>=0.8 %.3f  max %.3f'%(T,R['median_nn_jaccard'],R['frac_ge_0.5'],R['frac_ge_0.8'],R['max_nn_jaccard']),flush=True)
X=np.stack([feat(q,L0) for q in S]); Xt=np.stack([feat(q,L0) for q in St]); X,Xt=norm(X,Xt)
m=LogisticRegression(C=0.01,max_iter=300).fit(X,Y); p=m.predict(Xt)
R['mcc_all']=float(matthews_corrcoef(Yt,p)); k=best<THR; R['n_kept']=int(k.sum())
print('mcc all %.4f'%R['mcc_all'],flush=True)
if k.sum()>50:
    R['mcc_filtered']=float(matthews_corrcoef(Yt[k],p[k])); print('mcc on %d non-homologous test seqs %.4f'%(k.sum(),R['mcc_filtered']),flush=True)
json.dump(R,open('results/leak_%s.json'%T,'w')); print('LEAK_DONE %s %ds'%(T,time.time()-t0))
