# Pyramid variants. usage: train_pyramid.py <task> <variant> [seed]
# variants: base | r2 (ratio-2 levels, Theorem 2) | fuse_<fm> (base + frozen FM embedding branch) | r2fuse_<fm>
import sys,json,time,numpy as np,cgr_features as PP,torch,torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
T,VAR=sys.argv[1],sys.argv[2]; SEED=int(sys.argv[3]) if len(sys.argv)>3 else 0
dev='cuda' if torch.cuda.is_available() else 'cpu'; t0=time.time()
LEVS={'sp':[(5,1),(4,4),(3,16),(3,64),(2,0),(3,0)],'k6':[(6,1),(5,1),(4,4),(3,16),(3,64),(2,0)],'base':PP.LEV,'r2':[(5,1),(4,2),(4,4),(3,8),(3,16),(3,32),(3,64),(2,0)]}
lev=LEVS[next((k for k in ('k6','r2','sp') if VAR.startswith(k)),'base')]; H=128 if 'w' in VAR.split('_')[0][2:] else 64; LR=3e-4 if 'lr3' in VAR else 1e-3
FM=VAR.split('_',1)[1] if 'fuse' in VAR else None
def rcmap(k):
    n=4**k; idx=np.arange(n); r=np.zeros(n,np.int64)
    for j in range(k): r=r*4+(3-(idx//4**j)%4)
    u,inv=np.unique(np.minimum(idx,r),return_inverse=True); return inv,len(u)
CAN=not (T.startswith('splice') or T=='promoter_tata')
W={k:rcmap(k) for k in (2,3,4,5,6)} if CAN else None
S,Y=PP.read(T,'train'); St,Yt=PP.read(T,'test'); L=len(S[0])
X=np.stack([PP.feat(q,L,lev,W) for q in S]); Xt=np.stack([PP.feat(q,L,lev,W) for q in St]); X,Xt=PP.norm(X,Xt)
DL=PP.dims(L,lev,W)
if FM:
    z=np.load('emb/%s_%s.npz'%(FM,T)); assert (z['Y']==Y).all() and (z['Yt']==Yt).all()
    E=np.concatenate([z['E'],z['Em']],1).astype(np.float32); Et=np.concatenate([z['Et'],z['Etm']],1).astype(np.float32)
    E,Et=PP.norm(E,Et); X=np.concatenate([X,E],1); Xt=np.concatenate([Xt,Et],1); DL=DL+[E.shape[1]]
X=X.astype(np.float32); Xt=Xt.astype(np.float32)
NC=int(Y.max())+1
class Pyr(nn.Module):
    def __init__(s,DL,h=64,p=0.3):
        super().__init__(); s.DL=DL
        s.br=nn.ModuleList([nn.Sequential(nn.Dropout(0.1),nn.Linear(d,h),nn.ReLU(),nn.Dropout(p)) for d in DL])
        s.out=nn.Linear(h*len(DL),NC)
    def forward(s,x):
        o=[];i=0
        for d,b in zip(s.DL,s.br): o.append(b(x[:,i:i+d])); i+=d
        return s.out(torch.cat(o,1))
def prob(m,A,bs=1024):
    m.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(A),bs): o.append(torch.softmax(m(torch.tensor(A[i:i+bs]).to(dev)),1).cpu().numpy())
    return np.concatenate(o)
torch.manual_seed(SEED); rs=np.random.RandomState(SEED)
tr,va=train_test_split(np.arange(len(Y)),test_size=0.1,stratify=Y,random_state=SEED)
m=Pyr(DL,h=H).to(dev); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=1e-5)
best=(-2,None,0); pat=0
for ep in range(40):
    m.train()
    for b in np.array_split(rs.permutation(tr),max(1,len(tr)//128)):
        opt.zero_grad(); l=nn.functional.cross_entropy(m(torch.tensor(X[b]).to(dev)),torch.tensor(Y[b]).to(dev)); l.backward(); opt.step()
    v=matthews_corrcoef(Y[va],prob(m,X[va]).argmax(1))
    if v>best[0]: best=(v,{k:t.clone() for k,t in m.state_dict().items()},ep); pat=0
    else:
        pat+=1
        if pat>=8: break
m.load_state_dict(best[1]); p=prob(m,Xt); mcc=matthews_corrcoef(Yt,p.argmax(1)); acc=float((p.argmax(1)==Yt).mean())
r=dict(task=T,variant=VAR,seed=SEED,val_mcc=float(best[0]),best_ep=best[2],test_mcc=float(mcc),test_acc=acc,canonical=CAN,dims=int(sum(DL)),sec=time.time()-t0)
json.dump(r,open('results/pyr2_%s_%s_s%d.json'%(VAR,T,SEED),'w')); np.save('results/pyr2_%s_%s_s%d.npy'%(VAR,T,SEED),p.astype(np.float32))
print('TEST-pyr2-%s %s seed %d acc %.4f mcc %.4f val %.4f ep %d dims %d (%.0fs)'%(VAR,T,SEED,acc,mcc,best[0],best[2],sum(DL),time.time()-t0),flush=True)
