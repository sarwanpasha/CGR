# ClinVar splice-variant effect study. usage: clinvar_splice_variants.py [maxdist=50] [seeds=3]
# Trains pyramid net and CNN on NT donor/acceptor tasks, scores ref->alt loss of splice-site logit
# for ClinVar pathogenic vs benign SNVs near MANE Select junctions.
import sys,gzip,json,re,collections,bisect,time,numpy as np,cgr_features as PP
import torch,torch.nn as nn
from sklearn.metrics import roc_auc_score
MAXD=int(sys.argv[1]) if len(sys.argv)>1 else 50
NS=int(sys.argv[2]) if len(sys.argv)>2 else 3
CV='data/cv/'; dev='cuda' if torch.cuda.is_available() else 'cpu'
COMP=str.maketrans('ACGTN','TGCAN')
def rc(s): return s.translate(COMP)[::-1]
t0=time.time()
def log(*a): print('[%5.0fs]'%(time.time()-t0),*a,flush=True)

# ---------- models ----------
class Pyr(nn.Module):
    def __init__(s,DL,h=64,p=0.3):
        super().__init__(); s.DL=DL
        s.br=nn.ModuleList([nn.Sequential(nn.Dropout(0.1),nn.Linear(d,h),nn.ReLU(),nn.Dropout(p)) for d in DL])
        s.out=nn.Linear(h*len(DL),1)
    def forward(s,x):
        o=[];i=0
        for d,b in zip(s.DL,s.br): o.append(b(x[:,i:i+d])); i+=d
        return s.out(torch.cat(o,1)).squeeze(1)
class CNN(nn.Module):
    def __init__(s,c=128):
        super().__init__()
        L=[nn.Conv1d(4,c,9,padding=4),nn.ReLU()]
        for d in (2,4,8,16): L+=[nn.Conv1d(c,c,9,padding=4*d,dilation=d),nn.BatchNorm1d(c),nn.ReLU()]
        s.f=nn.Sequential(*L); s.out=nn.Sequential(nn.Dropout(0.3),nn.Linear(2*c,1))
    def forward(s,x):
        h=s.f(x); return s.out(torch.cat([h.max(2)[0],h.mean(2)],1)).squeeze(1)
def onehot(S):
    X=np.zeros((len(S),4,len(S[0])),np.float32)
    for i,q in enumerate(S):
        a=PP.LUT[np.frombuffer(q.encode(),np.uint8)]; m=a<4; X[i,a[m],np.where(m)[0]]=1
    return X
def pred(m,X,bs=512):
    m.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),bs): o.append(m(torch.tensor(X[i:i+bs]).to(dev)).cpu().numpy())
    return np.concatenate(o)
def train(mk,X,Y,seed,ep=20,bs=128,lr=1e-3):
    torch.manual_seed(seed); rs=np.random.RandomState(seed)
    idx=rs.permutation(len(Y)); nv=len(Y)//10; va,tr=idx[:nv],idx[nv:]
    m=mk().to(dev); opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-5)
    Yf=Y.astype(np.float32); best=(-1,None)
    for e in range(ep):
        m.train()
        for b in np.array_split(rs.permutation(tr),max(1,len(tr)//bs)):
            opt.zero_grad()
            l=nn.functional.binary_cross_entropy_with_logits(m(torch.tensor(X[b]).to(dev)),torch.tensor(Yf[b]).to(dev))
            l.backward(); opt.step()
        a=roc_auc_score(Y[va],pred(m,X[va]))
        if a>best[0]: best=(a,{k:v.clone() for k,v in m.state_dict().items()})
    m.load_state_dict(best[1]); return m,best[0]

# ---------- site models ----------
SITE={}
for ty,T,dn in (('D','splice_sites_donors','GT'),('A','splice_sites_acceptors','AG')):
    S,Y=PP.read(T,'train'); St,Yt=PP.read(T,'test'); L=len(S[0])
    fr={c:np.mean([s[L//2-20:L//2+20].count(dn)>0 for s,y in zip(S,Y) if y==c]) for c in set(Y.tolist())}
    pc=max(fr,key=fr.get); Y=(Y==pc).astype(int); Yt=(Yt==pc).astype(int)
    pos=[s for s,y in zip(S,Y) if y==1]
    off=max(range(L//2-20,L//2+20),key=lambda i:sum(s[i:i+2]==dn for s in pos))
    frac=np.mean([s[off:off+2]==dn for s in pos])
    anc=off if ty=='D' else off+2          # window index of first intron base (D) / first exon base (A)
    log(ty,T,'L',L,'n',len(S),'pos-label',pc,'dinuc idx',off,'frac',round(frac,3),'anchor',anc)
    X=np.stack([PP.feat(q,L) for q in S]); Xt=np.stack([PP.feat(q,L) for q in St])
    mu=X.mean(0); sd=X.std(0)+1e-6; X=(X-mu)/sd; Xt=(Xt-mu)/sd
    O=onehot(S); Ot=onehot(St); DL=PP.dims(L)
    M={'pyr':[],'cnn':[]}; te={}
    for sd_ in range(NS):
        mp,ap=train(lambda:Pyr(DL),X,Y,sd_); M['pyr'].append(mp)
        mc,ac=train(lambda:CNN(),O,Y,sd_,lr=5e-4); M['cnn'].append(mc)
        log(ty,'seed',sd_,'val auc pyr',round(ap,4),'cnn',round(ac,4))
    for k in M:
        p=np.mean([pred(m,Xt if k=='pyr' else Ot) for m in M[k]],0)
        te[k]=roc_auc_score(Yt,p); log(ty,'TEST auc',k,round(te[k],4))
    SITE[ty]=dict(L=L,anc=anc,dn=dn,off=off,mu=mu,sd=sd,M=M,test=te)
    del X,Xt,O,Ot

# ---------- ClinVar ----------
PS={'Pathogenic','Likely_pathogenic','Pathogenic/Likely_pathogenic'}
BS={'Benign','Likely_benign','Benign/Likely_benign'}
V=[]
for l in gzip.open(CV+'clinvar.vcf.gz','rt'):
    if l[0]=='#': continue
    f=l.rstrip('\n').split('\t')
    if len(f[3])!=1 or len(f[4])!=1 or f[3] not in 'ACGT' or f[4] not in 'ACGT': continue
    inf=f[7]; m=re.search(r'CLNSIG=([^;]+)',inf)
    if not m: continue
    lab=1 if m.group(1) in PS else 0 if m.group(1) in BS else None
    if lab is None: continue
    rv=re.search(r'CLNREVSTAT=([^;]+)',inf); rv=rv.group(1) if rv else ''
    if 'no_assertion' in rv or 'no_interpretation' in rv or 'no_classification' in rv: continue
    mc=re.search(r'MC=([^;]+)',inf); g=re.search(r'GENEINFO=([^:;]+)',inf)
    V.append(('chr'+f[0],int(f[1]),f[3],f[4],lab,mc.group(1) if mc else '',g.group(1) if g else '',f[2],m.group(1)))
log('clinvar SNVs P/LP or B/LB with criteria',len(V),collections.Counter(v[4] for v in V))

ex=collections.defaultdict(list); tinfo={}
for l in gzip.open(CV+'gencode.v44.annotation.gtf.gz','rt'):
    if l[0]=='#': continue
    f=l.split('\t')
    if f[2]!='exon' or 'MANE_Select' not in f[8]: continue
    t=re.search(r'transcript_id "([^"]+)"',f[8]).group(1)
    ex[t].append((int(f[3]),int(f[4]))); tinfo[t]=(f[0],f[6])
SI=collections.defaultdict(list)   # chrom -> (anchor genomic coord, strand, type)
for t,E in ex.items():
    c,st=tinfo[t]; E.sort()
    for i,(s,e) in enumerate(E):
        if st=='+':
            if i<len(E)-1: SI[c].append((e+1,'+','D'))
            if i>0: SI[c].append((s,'+','A'))
        else:
            if i>0: SI[c].append((s-1,'-','D'))
            if i<len(E)-1: SI[c].append((e,'-','A'))
for c in SI: SI[c]=sorted(set(SI[c]))
log('MANE transcripts',len(ex),'junction sites',sum(len(v) for v in SI.values()))

G={}; cur=None; buf=[]
want=set('chr%s'%x for x in list(range(1,23))+['X','Y'])
for l in gzip.open(CV+'GRCh38.primary_assembly.genome.fa.gz','rt'):
    if l[0]=='>':
        if cur in want: G[cur]=''.join(buf).upper()
        cur=l[1:].split()[0]; buf=[]
    elif cur in want: buf.append(l.rstrip())
if cur in want: G[cur]=''.join(buf).upper()
log('genome chromosomes',len(G))

R=[]; stat=collections.Counter()
for c,v,ref,alt,lab,mc,gene,vid,sig in V:
    if c not in G or c not in SI: stat['nochrom']+=1; continue
    if G[c][v-1]!=ref: stat['refmismatch']+=1; continue
    A=SI[c]
    j=bisect.bisect_left(A,(v,)); best=None
    for k in (j-2,j-1,j,j+1):
        if 0<=k<len(A) and (best is None or abs(A[k][0]-v)<abs(best[0]-v)): best=A[k]
    if best is None or abs(best[0]-v)>MAXD: stat['far']+=1; continue
    an,st,ty=best; S_=SITE[ty]; L,ai=S_['L'],S_['anc']
    if st=='+':
        s0=an-ai; w=G[c][s0-1:s0-1+L]; idx=v-s0; wa=w[:idx]+alt+w[idx+1:]
    else:
        e0=an+ai; s0=e0-L+1; w=rc(G[c][s0-1:e0]); idx=e0-v; wa=w[:idx]+alt.translate(COMP)+w[idx+1:]
    if len(w)!=L: stat['edge']+=1; continue
    if w[S_['off']:S_['off']+2]!=S_['dn']: stat['noncanon_site']+=1; continue
    rel=idx-ai
    R.append(dict(chrom=c,pos=v,ref=ref,alt=alt,label=lab,type=ty,strand=st,rel=rel,gene=gene,id=vid,sig=sig,mc=mc,w=w,wa=wa))
    stat['kept']+=1
log('mapping',dict(stat))
log('kept by type/label',collections.Counter((r['type'],r['label']) for r in R))

# ---------- scoring ----------
for ty in ('D','A'):
    S_=SITE[ty]; I=[i for i,r in enumerate(R) if r['type']==ty]
    if not I: continue
    W=[R[i]['w'] for i in I]; WA=[R[i]['wa'] for i in I]; L=S_['L']
    F=(np.stack([PP.feat(q,L) for q in W])-S_['mu'])/S_['sd']; FA=(np.stack([PP.feat(q,L) for q in WA])-S_['mu'])/S_['sd']
    O=onehot(W); OA=onehot(WA)
    for k,(a,b) in (('pyr',(F,FA)),('cnn',(O,OA))):
        d=np.mean([pred(m,a)-pred(m,b) for m in S_['M'][k]],0)
        for i,x in zip(I,d): R[i]['d_'+k]=float(x)

def auc(y,s): return float(roc_auc_score(y,s)) if 0<y.sum()<len(y) else float('nan')
def posauc(y,s,rel):
    num=den=0.
    for r in np.unique(rel):
        m=rel==r; n1=y[m].sum(); n0=m.sum()-n1
        if n1 and n0: num+=auc(y[m],s[m])*n1*n0; den+=n1*n0
    return num/den if den else float('nan')
def boot(y,a,b,rel=None,B=1000):
    rr=rel if rel is not None else np.zeros(len(y),int)
    f=(lambda yy,ss,r_: posauc(yy,ss,r_)) if rel is not None else (lambda yy,ss,r_: auc(yy,ss))
    rs=np.random.RandomState(0); o=[]
    for _ in range(B):
        i=rs.randint(0,len(y),len(y))
        if 0<y[i].sum()<len(i): o.append(f(y[i],a[i],rr[i])-f(y[i],b[i],rr[i]))
    o=np.array(o); o=o[~np.isnan(o)]; return [float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
y=np.array([r['label'] for r in R]); rel=np.array([r['rel'] for r in R]); ty=np.array([r['type'] for r in R])
can=np.array([(r['type']=='D' and r['rel'] in (0,1)) or (r['type']=='A' and r['rel'] in (-2,-1)) for r in R])
exo=np.array([(r['type']=='D' and r['rel']<0) or (r['type']=='A' and r['rel']>=0) for r in R])
sc={'pyr':np.array([r['d_pyr'] for r in R]),'cnn':np.array([r['d_cnn'] for r in R]),'dist':-np.abs(rel).astype(float)}
strata={'all':np.ones(len(R),bool),'canonical':can,'noncanonical':~can,'noncanon_intronic':(~can)&(~exo),'exonic':exo,'donor':ty=='D','acceptor':ty=='A','donor_noncanon':(ty=='D')&~can,'acceptor_noncanon':(ty=='A')&~can}
out={'maxdist':MAXD,'seeds':NS,'test_auc':{t:SITE[t]['test'] for t in SITE},'mapping':dict(stat),'strata':{}}
for nm,m in strata.items():
    if m.sum()==0: continue
    e={'n':int(m.sum()),'n_path':int(y[m].sum())}
    for k,s in sc.items(): e['auc_'+k]=auc(y[m],s[m]); e['posauc_'+k]=posauc(y[m],s[m],rel[m])
    if 0<y[m].sum()<m.sum():
        e['diff_auc_ci']=boot(y[m],sc['pyr'][m],sc['cnn'][m]); e['diff_posauc_ci']=boot(y[m],sc['pyr'][m],sc['cnn'][m],rel[m],B=300)
    out['strata'][nm]=e
    log('CV',nm,{k:(round(v,4) if isinstance(v,float) else v) for k,v in e.items()})
json.dump(out,open('results/cv_splice.json','w'),indent=1)
with open('results/cv_variants.tsv','w') as fo:
    ks=['chrom','pos','ref','alt','label','sig','type','strand','rel','gene','id','d_pyr','d_cnn']
    fo.write('\t'.join(ks)+'\n')
    for r in R: fo.write('\t'.join(str(r[k]) for k in ks)+'\n')
np.savez('data/cv/windows.npz',w=np.array([r['w'] for r in R]),wa=np.array([r['wa'] for r in R]),y=y,rel=rel,ty=ty)
log('CV_DONE')
