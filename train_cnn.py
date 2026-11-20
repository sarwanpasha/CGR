# Position-aware one-hot CNN branch for the hybrid. usage: train_cnn.py <task> [seed]
import sys,json,time,numpy as np,cgr_features as PP,torch,torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
T=sys.argv[1]; SEED=int(sys.argv[2]) if len(sys.argv)>2 else 0
dev='cuda' if torch.cuda.is_available() else 'cpu'; t0=time.time()
def onehot(S):
    X=np.zeros((len(S),4,len(S[0])),np.float32)
    for i,q in enumerate(S):
        a=PP.LUT[np.frombuffer(q.encode(),np.uint8)]; m=a<4; X[i,a[m],np.where(m)[0]]=1
    return X
S,Y=PP.read(T,'train'); St,Yt=PP.read(T,'test'); X=onehot(S); Xt=onehot(St); NC=int(Y.max())+1
class Res(nn.Module):
    def __init__(s,c,d):
        super().__init__(); s.f=nn.Sequential(nn.Conv1d(c,c,9,padding=4*d,dilation=d),nn.BatchNorm1d(c),nn.ReLU(),nn.Conv1d(c,c,1),nn.BatchNorm1d(c))
    def forward(s,x): return torch.relu(x+s.f(x))
class Net(nn.Module):
    def __init__(s,c=96,nb=32):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(4,c,11,padding=5),nn.BatchNorm1d(c),nn.ReLU())
        s.res=nn.Sequential(*[Res(c,d) for d in (1,2,4,8)])
        s.pool=nn.AdaptiveMaxPool1d(nb)
        s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(c*nb,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,NC))
    def forward(s,x): return s.head(s.pool(s.res(s.stem(x))).flatten(1))
def prob(m,A,bs=512):
    m.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(A),bs): o.append(torch.softmax(m(torch.tensor(A[i:i+bs]).to(dev)),1).cpu().numpy())
    return np.concatenate(o)
torch.manual_seed(SEED); rs=np.random.RandomState(SEED)
tr,va=train_test_split(np.arange(len(Y)),test_size=0.1,stratify=Y,random_state=SEED)
EP=15; nb=max(1,len(tr)//128); m=Net().to(dev)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=EP*nb)
best=(-2,None,0)
for ep in range(EP):
    m.train()
    for b in np.array_split(rs.permutation(tr),nb):
        opt.zero_grad(); l=nn.functional.cross_entropy(m(torch.tensor(X[b]).to(dev)),torch.tensor(Y[b]).to(dev)); l.backward(); opt.step(); sch.step()
    v=matthews_corrcoef(Y[va],prob(m,X[va]).argmax(1))
    print('ep',ep,'val mcc %.4f (%.0fs)'%(v,time.time()-t0),flush=True)
    if v>best[0]: best=(v,{k:t.clone() for k,t in m.state_dict().items()},ep)
m.load_state_dict(best[1]); p=prob(m,Xt); mcc=matthews_corrcoef(Yt,p.argmax(1)); acc=float((p.argmax(1)==Yt).mean())
r=dict(task=T,variant='cnn',seed=SEED,val_mcc=float(best[0]),best_ep=best[2],test_mcc=float(mcc),test_acc=acc,sec=time.time()-t0)
json.dump(r,open('results/pyr2_cnn_%s_s%d.json'%(T,SEED),'w')); np.save('results/pyr2_cnn_%s_s%d.npy'%(T,SEED),p.astype(np.float32))
print('TEST-pyr2-cnn %s seed %d acc %.4f mcc %.4f val %.4f ep %d (%.0fs)'%(T,SEED,acc,mcc,best[0],best[2],time.time()-t0),flush=True)
