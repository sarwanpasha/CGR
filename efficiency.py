import sys,json,time,numpy as np,torch,torch.nn as nn
from cgr_features import read,feat,dims
T=sys.argv[1]; St,_=read(T,'test'); St=St[:2000]; L0=len(St[0]); DL=dims(L0)
t0=time.time(); X=np.stack([feat(q,L0) for q in St]); ft=time.time()-t0
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Pyr(nn.Module):
    def __init__(s):
        super().__init__(); s.dl=DL
        s.g=nn.ModuleList([nn.Sequential(nn.Linear(d,64),nn.ReLU()) for d in DL]); s.o=nn.Linear(64*len(DL),2)
    def forward(s,p): return s.o(torch.cat([g(q) for g,q in zip(s.g,torch.split(p,s.dl,1))],1))
class Cnn(nn.Module):
    def __init__(s):
        super().__init__(); c=[]; ic=4
        for oc,d in ((64,1),(64,2),(128,4),(128,8)): c+=[nn.Conv1d(ic,oc,7,padding=3*d,dilation=d),nn.GroupNorm(8,oc),nn.ReLU()]; ic=oc
        s.f=nn.Sequential(*c); s.p=nn.AdaptiveMaxPool1d(24)
        s.h=nn.Sequential(nn.Flatten(),nn.Linear(128*24,256),nn.ReLU(),nn.Linear(256,2))
    def forward(s,x): return s.h(s.p(s.f(x)))
R={'task':T,'n':len(St),'L':L0,'dim':int(X.shape[1]),'feat_s_per_1k':ft/len(St)*1000}
for nm,net,inp in (('pyramid',Pyr(),torch.tensor(X)),('onehot_cnn',Cnn(),torch.zeros(len(St),4,L0))):
    net=net.to(dev).eval(); n=sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        for i in range(0,512,256): net(inp[i:i+256].to(dev))
        if dev.type=='cuda': torch.cuda.synchronize()
        t1=time.time()
        for i in range(0,len(inp),256): net(inp[i:i+256].to(dev))
        if dev.type=='cuda': torch.cuda.synchronize()
        dt=time.time()-t1
    R[nm]={'params':int(n),'MB':n*4/1e6,'seq_per_s':len(inp)/dt}
    print('%-11s params %9d (%6.1f MB)  %8.0f seq/s'%(nm,n,n*4/1e6,len(inp)/dt),flush=True)
print('features dim %d  %.2f s per 1000 seq'%(R['dim'],R['feat_s_per_1k']))
json.dump(R,open('results/eff_%s.json'%T,'w')); print('EFF_DONE %s'%T)
