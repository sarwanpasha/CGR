# Frozen foundation-model embeddings + linear probe. usage: fm_embed_probe.py <model> <task> [limit]
# model: dnabert2 | nt500 | ntv2   Saves mean-pooled embeddings (last + middle layer) to emb/<model>_<task>.npz
import sys,os,time,json,numpy as np

import torch
from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score
from cgr_features import read
M,T=sys.argv[1],sys.argv[2]; LIM=int(sys.argv[3]) if len(sys.argv)>3 else 0
NAME={'dnabert2':'zhihan1996/DNABERT-2-117M','nt500':'InstaDeepAI/nucleotide-transformer-500m-human-ref','ntv2':'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'}[M]
t0=time.time(); dev=torch.device('cuda')
tok=AutoTokenizer.from_pretrained(NAME,trust_remote_code=True)
model=(AutoModel if M=='dnabert2' else AutoModelForMaskedLM).from_pretrained(NAME,trust_remote_code=True).to(dev).eval()
npar=sum(p.numel() for p in model.parameters()); print('loaded',NAME,'params %.1fM'%(npar/1e6),'%ds'%(time.time()-t0),flush=True)
S,Y=read(T,'train'); St,Yt=read(T,'test')
if LIM: a=max(1,len(S)//LIM); b=max(1,len(St)//LIM); S,Y,St,Yt=S[::a][:LIM],Y[::a][:LIM],St[::b][:LIM],Yt[::b][:LIM]
BS=64 if M=='dnabert2' else 32
def emb(Q):
    last=[];mid=[]
    with torch.no_grad():
        for i in range(0,len(Q),BS):
            t=tok(Q[i:i+BS],return_tensors='pt',padding=True,truncation=True,max_length=1024)
            ids=t['input_ids'].to(dev); am=t['attention_mask'].to(dev); m=am.unsqueeze(-1).float()
            if M=='dnabert2':
                h=model(ids,attention_mask=am)[0]; hm=h
            else:
                o=model(ids,attention_mask=am,encoder_attention_mask=am,output_hidden_states=True)['hidden_states']
                h=o[-1]; hm=o[len(o)//2]
            last.append(((h*m).sum(1)/m.sum(1)).cpu().numpy()); mid.append(((hm*m).sum(1)/m.sum(1)).cpu().numpy())
    return np.concatenate(last),np.concatenate(mid)
E,Em=emb(S); print('train emb',E.shape,'%ds'%(time.time()-t0),flush=True)
Et,Etm=emb(St); te=time.time()-t0
os.makedirs('emb',exist_ok=True)
if not LIM: np.savez('emb/%s_%s.npz'%(M,T),E=E,Em=Em,Et=Et,Etm=Etm,Y=Y,Yt=Yt)
rs=np.random.RandomState(0); va=np.zeros(len(Y),bool)
for c in np.unique(Y):
    ix=np.where(Y==c)[0]; va[rs.choice(ix,max(1,len(ix)//10),replace=False)]=True
best=None
for lay,(A,B) in (('last',(E,Et)),('mid',(Em,Etm))):
    mu=A[~va].mean(0); sd=A[~va].std(0)+1e-6; A=(A-mu)/sd; B=(B-mu)/sd
    for C in (0.001,0.01,0.1,1.0):
        m=LogisticRegression(C=C,max_iter=2000).fit(A[~va],Y[~va]); v=matthews_corrcoef(Y[va],m.predict(A[va]))
        if best is None or v>best[0]: best=(v,lay,C,A,B)
v,lay,C,A,B=best; m=LogisticRegression(C=C,max_iter=2000).fit(A,Y); p=m.predict(B)
r={'model':M,'task':T,'params_M':npar/1e6,'layer':lay,'C':C,'val_mcc':float(v),'acc':float(accuracy_score(Yt,p)),'f1':float(f1_score(Yt,p,average='macro')),'mcc':float(matthews_corrcoef(Yt,p)),'embed_seconds':te}
os.makedirs('results',exist_ok=True)
if not LIM: json.dump(r,open('results/fmemb_%s_%s.json'%(M,T),'w'))
print('TEST-fmemb-%s %s probe seed 0 acc %.4f mcc %.4f layer %s C %g (%ds)'%(M,T,r['acc'],r['mcc'],lay,C,time.time()-t0),flush=True)
