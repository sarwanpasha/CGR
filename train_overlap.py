# Pyramid with half-shifted (overlapping) bins. usage: train_overlap.py <task> [seed] [base|k6]
# Every binned level gets a second set of bins shifted by half a bin width, so that a motif is never
# split across a bin edge at the matching resolution. Results are written as variant ov / k6ov.
import sys,json,time,numpy as np,torch,torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
try: import cgr_features as PP
except ImportError: import P as PP
T=sys.argv[1]; SEED=int(sys.argv[2]) if len(sys.argv)>2 else 0; BV=sys.argv[3] if len(sys.argv)>3 else 'base'; VAR='ov' if BV=='base' else BV+'ov'
dev='cuda' if torch.cuda.is_available() else 'cpu'; t0=time.time()
lev={'base':[(5,1),(4,4),(3,16),(3,64),(2,0)],'k6':[(6,1),(5,1),(4,4),(3,16),(3,64),(2,0)]}[BV]; H=64; LR=1e-3
def rcmap(k):
    n=4**k; idx=np.arange(n); r=np.zeros(n,np.int64)
    for j in range(k): r=r*4+(3-(idx//4**j)%4)
    u,inv=np.unique(np.minimum(idx,r),return_inverse=True); return inv,len(u)
CAN=not (T.startswith('splice') or T=='promoter_tata')
W={k:rcmap(k) for k in (2,3,4,5,6)} if CAN else None
def feat_ov(q,L0):
    a=PP.LUT[np.frombuffer(q.encode(),np.uint8)]; o=[]
    for k,r in lev:
        km,bad,n=PP.kmers(a,k); M=W[k][1] if W else 4**k; pos=np.arange(n)[~bad]; kk=km[~bad]
        if W: kk=W[k][0][kk]
        R=r if r else L0; b=(pos*r//n) if r else np.minimum(pos,R-1)
        o.append(np.bincount(b*M+kk,minlength=R*M).astype(np.float32))
        if r>1: o.append(np.bincount(((2*pos*r+n)//(2*n))*M+kk,minlength=(r+1)*M).astype(np.float32))
    return np.log1p(np.concatenate(o))
def dims_ov(L0):
    d=[]
    for k,r in lev:
        M=W[k][1] if W else 4**k; d.append((r if r else L0)*M)
        if r>1: d.append((r+1)*M)
    return d
S,Y=PP.read(T,'train'); St,Yt=PP.read(T,'test'); L=len(S[0])
X=np.stack([feat_ov(q,L) for q in S]); Xt=np.stack([feat_ov(q,L) for q in St]); X,Xt=PP.norm(X,Xt)
DL=dims_ov(L)
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
