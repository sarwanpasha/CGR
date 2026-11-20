# SpliceAI baseline on the ClinVar non-canonical splice-relevant stratum. usage: clinvar_spliceai_baseline.py [n_benign=10000]
import sys,os,gzip,re,json,glob,time,importlib.util,numpy as np
from sklearn.metrics import roc_auc_score as AU
NB=int(sys.argv[1]) if len(sys.argv)>1 else 10000
t0=time.time()
def log(*a): print('[%5.0fs]'%(time.time()-t0),*a,flush=True)
D=[l.rstrip('\n').split('\t') for l in open('results/cv_variants.tsv')];h=D[0];D=D[1:];ix={k:i for i,k in enumerate(h)}
ids={d[ix['id']] for d in D};MC={}
for l in gzip.open('data/cv/clinvar.vcf.gz','rt'):
    if l[0]=='#': continue
    f=l.split('\t',8)
    if f[2] in ids: m=re.search(r'MC=([^;\t\n]+)',f[7]); MC[f[2]]=m.group(1) if m else ''
y=np.array([int(d[ix['label']]) for d in D]); rel=np.array([int(d[ix['rel']]) for d in D]); ty=np.array([d[ix['type']] for d in D])
mc=[MC.get(d[ix['id']],'') for d in D]
cod=np.array([any(k in m for k in ('missense','nonsense','frameshift','stop_','start_lost','inframe')) for m in mc])
syn=np.array(['synonymous_variant' in m for m in mc])&~cod; intr=np.array(['intron_variant' in m or 'splice' in m for m in mc])&~cod
can=((ty=='D')&np.isin(rel,[0,1]))|((ty=='A')&np.isin(rel,[-2,-1]))
M=(syn|intr)&~can; rs=np.random.RandomState(0)
P=np.where(M&(y==1))[0]; B=np.where(M&(y==0))[0]; B=rs.choice(B,min(NB,len(B)),replace=False); I=np.sort(np.concatenate([P,B]))
log('stratum',int(M.sum()),'pathogenic',len(P),'benign sampled',len(B))
G={}; cur=None; buf=[]; want={d[ix['chrom']] for d in D}
for l in gzip.open('data/cv/GRCh38.primary_assembly.genome.fa.gz','rt'):
    if l[0]=='>':
        if cur in want: G[cur]=''.join(buf).upper()
        cur=l[1:].split()[0]; buf=[]
    elif cur in want: buf.append(l.rstrip())
if cur in want: G[cur]=''.join(buf).upper()
log('genome',len(G))
COMP=str.maketrans('ACGTN','TGCAN'); LUT=np.full(256,4,np.int64)
for i,c in enumerate('ACGT'): LUT[ord(c)]=i
EYE=np.vstack([np.eye(4,dtype=np.float32),np.zeros((1,4),np.float32)])
F=5050
def seqs(d):
    c=d[ix['chrom']]; p=int(d[ix['pos']]); ref=d[ix['ref']]; alt=d[ix['alt']]; g=G[c]
    s=g[p-1-F:p+F]
    if len(s)!=2*F+1 or s[F]!=ref: return None
    a=s[:F]+alt+s[F+1:]
    if d[ix['strand']]=='-': s=s.translate(COMP)[::-1]; a=a.translate(COMP)[::-1]
    return s,a
def oh(s): return EYE[LUT[np.frombuffer(s.encode(),np.uint8)]]
import tensorflow as tf
from tensorflow.keras.models import load_model
mods=[load_model(f,compile=False) for f in sorted(glob.glob(os.path.join(os.path.dirname(importlib.util.find_spec('spliceai').origin),'models','spliceai*.h5')))]
log('models',len(mods),'gpus',len(tf.config.list_physical_devices('GPU')))
def score(X): return np.mean([m.predict(X,batch_size=16,verbose=0) for m in mods],0)
out={}; BS=64; keep=[]; SS=[]
for i in I:
    r=seqs(D[i])
    if r is not None: keep.append(i); SS.append(r)
log('scoring',len(keep))
for b in range(0,len(keep),BS):
    R=score(np.stack([oh(s) for s,a in SS[b:b+BS]])); A=score(np.stack([oh(a) for s,a in SS[b:b+BS]]))
    d=np.abs(A[:,:,1:]-R[:,:,1:]).max((1,2))
    for i,v in zip(keep[b:b+BS],d): out[int(i)]=float(v)
    if (b//BS)%20==0: log('done',b+BS,'of',len(keep)); json.dump(out,open('results/cv_spliceai_scores.json','w'))
json.dump(out,open('results/cv_spliceai_scores.json','w'))
K=np.array(keep); yy=y[K]; rr=rel[K]
S={'spliceai':np.array([out[int(i)] for i in K]),'pyr':np.array([float(D[i][ix['d_pyr']]) for i in K]),'cnn':np.array([float(D[i][ix['d_cnn']]) for i in K]),'dist':-np.abs(rr).astype(float)}
z=lambda v:(v-v.mean())/(v.std()+1e-9); S['pyr+spliceai']=z(S['pyr'])+z(S['spliceai'])
def posauc(yy_,s,r):
    n=d=0.
    for v in np.unique(r):
        m=r==v;a=yy_[m].sum();b=m.sum()-a
        if a and b: n+=AU(yy_[m],s[m])*a*b; d+=a*b
    return n/d if d else float('nan')
res={'n':len(K),'n_path':int(yy.sum())}
for nm,m in (('all',np.ones(len(K),bool)),('synonymous',syn[K]),('intronic',intr[K]),('far_gt10bp',np.abs(rr)>10)):
    res[nm]={'n':int(m.sum()),'n_path':int(yy[m].sum())}
    for k,s in S.items(): res[nm]['auc_'+k]=round(float(AU(yy[m],s[m])),4); res[nm]['posauc_'+k]=round(float(posauc(yy[m],s[m],rr[m])),4)
    log('SAI',nm,res[nm])
json.dump(res,open('results/cv_spliceai.json','w'),indent=1); log('SAI_DONE')
