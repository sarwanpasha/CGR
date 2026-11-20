# Aggregate pyramid variants, ensembles, and foundation-model runs vs published fine-tuned FMs.
import glob,json,collections,numpy as np,cgr_features as PP
from sklearn.metrics import matthews_corrcoef
PUB={'DNABERT2':[.490,.491,.599,.637,.490,.558,.646,.564,.443,.655,.517,.476,.754,.762,.784,.837,.835,.861],
     'NT500':[.465,.457,.589,.594,.468,.527,.622,.524,.433,.634,.515,.477,.734,.738,.831,.941,.939,.952],
     'NTv2_100m':[.492,.487,.595,.617,.485,.551,.633,.538,.445,.648,.507,.465,.753,.766,.826,.947,.960,.947],
     'NucEL':[.508,.470,.611,.584,.480,.546,.621,.550,.474,.640,.578,.536,.762,.769,.922,.960,.962,.970]}
TASKS=['H2AFZ','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H4K20me1','enhancers','enhancers_types','promoter_all','promoter_no_tata','promoter_tata','splice_sites_acceptors','splice_sites_all','splice_sites_donors']
PUB={m:dict(zip(TASKS,v)) for m,v in PUB.items()}
R=collections.defaultdict(dict)   # R[col][task]=mcc
for f in glob.glob('results/pyr2_*_s*.json'):
    r=json.load(open(f)); R.setdefault('mean:'+r['variant'],{}).setdefault(r['task'],[]).append(r['test_mcc'])
for c in [c for c in R if c.startswith('mean:')]:
    R[c]={t:float(np.mean(v)) for t,v in R[c].items()}
for f in glob.glob('results/fmemb_*.json'):
    r=json.load(open(f)); R['probe:'+r['model']][r['task']]=r['mcc']
for f in glob.glob('results/fmft_*.json'):
    r=json.load(open(f)); R.setdefault('ft:'+r['model'],{}).setdefault(r['task'],[]).append(r['mcc'])
for c in [c for c in R if c.startswith('ft:')]:
    R[c]={t:float(np.mean(v)) for t,v in R[c].items()}
# ensembles: average saved test probabilities over all seeds of the listed variants
YT={}
def ens(vars_,T):
    G_=[glob.glob('results/pyr2_%s_%s_s*.npy'%(v,T)) for v in vars_]
    if not all(G_): return None
    P_=[np.load(f) for g in G_ for f in g]
    if T not in YT: YT[T]=PP.read(T,'test')[1]
    return float(matthews_corrcoef(YT[T],np.mean(P_,0).argmax(1)))
vs=sorted({c[5:] for c in R if c.startswith('mean:')})
combos=[[v] for v in vs]+[['base','r2']]+[['base','r2','fuse_'+m] for m in ('dnabert2','ntv2','nt500')]+[['base','r2','fuse_dnabert2','fuse_ntv2','fuse_nt500']]
for cb in combos:
    nm='ens:'+'+'.join(cb)
    for T in TASKS:
        e=ens(cb,T)
        if e is not None: R[nm][T]=e
for m in PUB: R['pub:'+m]=PUB[m]
cols=[c for c in R if R[c]]
for i,c in enumerate(cols): print('  c%02d = %s'%(i,c))
print('%-17s'%'task'+''.join('%7s'%('c%02d'%i) for i in range(len(cols))))
for T in TASKS: print('%-17s'%T[:17]+''.join('%7s'%('%.3f'%R[c][T] if T in R[c] else '-') for c in cols))
print('\nmeans over tasks each column has, with published FM means over the same tasks:')
for c in cols:
    ts=[t for t in TASKS if t in R[c]]
    if not ts or c.startswith('pub:'): continue
    m=np.mean([R[c][t] for t in ts]); pub={k:np.mean([PUB[k][t] for t in ts]) for k in PUB}
    wins={k:sum(R[c][t]>PUB[k][t] for t in ts) for k in PUB}
    print('%-40s n=%2d mean %.4f | '%(c,len(ts),m)+'  '.join('%s %.4f (wins %d)'%(k,pub[k],wins[k]) for k in PUB))
json.dump(R,open('results/agg.json','w'),indent=1)
