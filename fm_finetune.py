# Full fine-tuning of a foundation model on one task. usage: fm_finetune.py <model> <task> [seed] [epochs] [limit]
# model: dnabert2 | ntv2_100   lr 3e-5, AdamW, linear decay, best epoch by validation MCC (same 10% split as our models)
import sys,os,time,json,math,numpy as np

import torch,torch.nn as nn
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score
from cgr_features import read
M,T=sys.argv[1],sys.argv[2]; SEED=int(sys.argv[3]) if len(sys.argv)>3 else 0; EP=int(sys.argv[4]) if len(sys.argv)>4 else 3; LIM=int(sys.argv[5]) if len(sys.argv)>5 else 0
NAME={'dnabert2':'zhihan1996/DNABERT-2-117M','ntv2_100':'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'}[M]
torch.manual_seed(SEED); np.random.seed(SEED); dev=torch.device('cuda'); t0=time.time()
S,Y=read(T,'train'); St,Yt=read(T,'test')
if LIM: a=max(1,len(S)//LIM); b=max(1,len(St)//LIM); S,Y,St,Yt=S[::a][:LIM],Y[::a][:LIM],St[::b][:LIM],Yt[::b][:LIM]
C=int(Y.max())+1; L0=len(S[0])
rs=np.random.RandomState(0); va=np.zeros(len(Y),bool)
for c in np.unique(Y):
    ix=np.where(Y==c)[0]; va[rs.choice(ix,max(1,len(ix)//10),replace=False)]=True
tri=np.where(~va)[0]; vai=np.where(va)[0]
tok=AutoTokenizer.from_pretrained(NAME,trust_remote_code=True)
model=AutoModelForSequenceClassification.from_pretrained(NAME,num_labels=C,trust_remote_code=True).to(dev)
npar=sum(p.numel() for p in model.parameters()); print('loaded %s %.1fM params, L=%d, train %d'%(NAME,npar/1e6,L0,len(tri)),flush=True)
BS=32 if L0<=400 else 16
def batch(Q,ix):
    t=tok([Q[i] for i in ix],return_tensors='pt',padding=True,truncation=True,max_length=1024)
    return t['input_ids'].to(dev),t['attention_mask'].to(dev)
def predict(Q,ixs):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(ixs),2*BS):
            ids,am=batch(Q,ixs[i:i+2*BS]); out.append(model(input_ids=ids,attention_mask=am).logits.softmax(-1).cpu().numpy())
    return np.concatenate(out)
opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.01)
steps=EP*math.ceil(len(tri)/BS); warm=max(1,int(0.06*steps))
sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: min((s+1)/warm,max(0.0,(steps-s)/max(1,steps-warm))))
lf=nn.CrossEntropyLoss(); best=-2; bst=None
for ep in range(EP):
    model.train(); pm=np.random.permutation(tri); tot=0.
    for i in range(0,len(pm),BS):
        ix=pm[i:i+BS]; ids,am=batch(S,ix); y=torch.tensor(Y[ix]).to(dev)
        opt.zero_grad(); l=lf(model(input_ids=ids,attention_mask=am).logits,y); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); tot+=l.item()*len(ix)
    pv=predict(S,vai).argmax(1); v=matthews_corrcoef(Y[vai],pv)
    print('ep %d loss %.4f val mcc %.4f %ds'%(ep,tot/len(tri),v,time.time()-t0),flush=True)
    if v>best: best=v; bst={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
model.load_state_dict(bst); pp=predict(St,np.arange(len(St))); p=pp.argmax(1)
r={'model':M,'task':T,'seed':SEED,'params_M':npar/1e6,'val_mcc':float(best),'acc':float(accuracy_score(Yt,p)),'f1':float(f1_score(Yt,p,average='macro')),'mcc':float(matthews_corrcoef(Yt,p)),'seconds':round(time.time()-t0)}
os.makedirs('results',exist_ok=True)
if not LIM: json.dump(r,open('results/fmft_%s_%s_s%d.json'%(M,T,SEED),'w')); np.save('results/fmft_%s_%s_s%d.npy'%(M,T,SEED),pp.astype(np.float32))
print('TEST-fmft-%s %s ft seed %d acc %.4f mcc %.4f (%ds)'%(M,T,SEED,r['acc'],r['mcc'],time.time()-t0),flush=True)
