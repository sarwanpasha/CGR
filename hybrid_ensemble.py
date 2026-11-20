# Build the reported predictors from saved test probabilities (results/pyr2_<variant>_<task>_s<seed>.npy).
# pyr: 4 pyramid variants x seeds | pyrcnn: pyr + CNN on fixed-position tasks | pyrfm: pyr + frozen-FM fusion | hyb: pyrfm + CNN
import glob,json,numpy as np,cgr_features as P
from sklearn.metrics import matthews_corrcoef as M
T=sorted({f.split('pyr2_base_')[1].rsplit('_s',1)[0] for f in glob.glob('results/pyr2_base_*_s0.npy')})
FX=[t for t in T if t.startswith('splice') or t=='promoter_tata']   # rule fixed in advance
def pr(vs,t): return np.mean([np.load(f) for v in vs for f in glob.glob('results/pyr2_%s_%s_s[0-9].npy'%(v,t))],0)
N=['base','r2','k6','k6w_lr3']; F=N+['fuse_dnabert2','k6fuse_dnabert2']
o={}
for nm,vs,c in (('pyr',N,0),('pyrcnn',N,1),('pyrfm',F,0),('hyb',F,1)):
    r={}
    for t in T:
        y=P.read(t,'test')[1]; p=pr(vs,t)
        if c and t in FX: p=(p+pr(['cnn'],t))/2
        r[t]=float(M(y,p.argmax(1)))
    o[nm]=r; print(nm,round(float(np.mean(list(r.values()))),4),flush=True)
json.dump(o,open('results/hybrid_ensemble.json','w'),indent=1)
