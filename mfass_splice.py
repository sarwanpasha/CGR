# Experimentally measured splicing variants (MFASS, Cheung et al. 2019). usage: mfass_splice.py [seeds=3]
# Trains the pyramid network and the CNN on the benchmark donor and acceptor tasks, scores the loss of the splice-site
# logit for every assayed variant at the donor and the acceptor of its exon, and compares with MaxEntScan (maxentpy)
# and with distance to the junction. Label: strong loss of exon inclusion (column strong_lof).
import sys,gzip,json,re,collections,bisect,time,numpy as np,cgr_features as PP
import torch,torch.nn as nn
from sklearn.metrics import roc_auc_score
NS=int(sys.argv[1]) if len(sys.argv)>1 else 3
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

# ---------- MFASS ----------
import csv
from sklearn.metrics import average_precision_score as AP
from maxentpy import maxent
try: m5, m3 = maxent.load_matrix5(), maxent.load_matrix3(); f5 = lambda q: maxent.score5(q, matrix=m5); f3 = lambda q: maxent.score3(q, matrix=m3)
except Exception: f5, f3 = maxent.score5, maxent.score3
R_ = list(csv.reader(open('data/mfass/snv_data_clean.txt'), delimiter='\t')); H = [h.strip('"') for h in R_[0]]; R_ = [[x.strip('"') for x in r] for r in R_[1:]]
def col(f): return [i for i, h in enumerate(H) if f(h)][0]
cS = col(lambda h: 'start_hg38' in h and 'snp' not in h); cP1 = [i for i, h in enumerate(H) if 'snp_position_hg38' in h and '1based' in h]; cP0 = [i for i, h in enumerate(H) if 'snp_position_hg38' in h and '0based' in h]
ix = {h: i for i, h in enumerate(H)}; log('mfass rows', len(R_), 'cols', len(H), 'start col', H[cS], 'snp col', H[cP1[0]] if cP1 else H[cP0[0]])
G={}; cur=None; buf=[]; want=set(r[ix['chr']] for r in R_)
for l in gzip.open(CV+'GRCh38.primary_assembly.genome.fa.gz','rt'):
    if l[0]=='>':
        if cur in want: G[cur]=''.join(buf).upper()
        cur=l[1:].split()[0]; buf=[]
    elif cur in want: buf.append(l.rstrip())
if cur in want: G[cur]=''.join(buf).upper()
log('genome chromosomes',len(G))
def num(x):
    try: return int(float(x))
    except Exception: return None
def build(up_first):
    out = []; st = collections.Counter()
    for r in R_:
        c = r[ix['chr']]; sd_ = r[ix['strand']]; ref = r[ix['ref_allele']].upper(); alt = r[ix['alt_allele']].upper(); lab = r[ix['strong_lof']].upper()
        s0 = num(r[cS]); v = num(r[cP1[0]]) if cP1 else (num(r[cP0[0]]) + 1 if num(r[cP0[0]]) is not None else None); i1, ex, i2 = num(r[ix['intron1_len']]), num(r[ix['exon_len']]), num(r[ix['intron2_len']])
        if c not in G or s0 is None or v is None or None in (i1, ex, i2) or lab not in ('TRUE', 'FALSE') or len(ref) != 1 or len(alt) != 1 or sd_ not in '+-': st['skip'] += 1; continue
        g = G[c][v - 1]
        if g == ref: ga = alt
        elif g == ref.translate(COMP): ga = alt.translate(COMP); st['allele_on_transcript_strand'] += 1
        else: st['refmismatch'] += 1; continue
        left = (i1 if sd_ == '+' else i2) if up_first else (i2 if sd_ == '+' else i1); s1 = s0 + left + 1; e1 = s1 + ex - 1; e = dict(label=int(lab == 'TRUE'), chrom=c, pos=v, strand=sd_, id=r[0], dpsi=r[ix['v2_dpsi']] if 'v2_dpsi' in ix else '', sites={})
        for ty, an in (('D', e1 + 1 if sd_ == '+' else s1 - 1), ('A', s1 if sd_ == '+' else e1)):
            S_ = SITE[ty]; L, ai = S_['L'], S_['anc']
            if sd_ == '+': a0 = an - ai; w = G[c][a0 - 1:a0 - 1 + L]; idx = v - a0; wa = w[:idx] + ga + w[idx + 1:] if 0 <= idx < L else None
            else: e0 = an + ai; a0 = e0 - L + 1; w = rc(G[c][a0 - 1:e0]); idx = e0 - v; wa = w[:idx] + ga.translate(COMP) + w[idx + 1:] if 0 <= idx < L else None
            if len(w) != L or wa is None: continue
            st['site_' + ty] += 1
            if w[S_['off']:S_['off'] + 2] != S_['dn']: continue
            st['canon_' + ty] += 1; e['sites'][ty] = (w, wa, idx - ai)
        if e['sites']: out.append(e)
    return out, st
A1, s1_ = build(True); A2, s2_ = build(False); V, stt = (A1, s1_) if s1_['canon_D'] + s1_['canon_A'] >= s2_['canon_D'] + s2_['canon_A'] else (A2, s2_)
log('mapping intron1-upstream', dict(s1_)); log('mapping intron1-downstream', dict(s2_)); log('kept', len(V), 'strong_lof', sum(e['label'] for e in V))
for e in V: e['pyr'] = -1e9; e['cnn'] = -1e9; e['mes'] = 0.0; e['motif'] = False; e['canon'] = False; e['dist'] = 999; e['exon'] = False
for ty in ('D', 'A'):
    S_ = SITE[ty]; I = [i for i, e in enumerate(V) if ty in e['sites']]
    if not I: continue
    Wr = [V[i]['sites'][ty][0] for i in I]; Wa = [V[i]['sites'][ty][1] for i in I]; L = S_['L']
    F = (np.stack([PP.feat(q, L) for q in Wr]) - S_['mu']) / S_['sd']; FA = (np.stack([PP.feat(q, L) for q in Wa]) - S_['mu']) / S_['sd']; O = onehot(Wr); OA = onehot(Wa)
    for k, (a, b) in (('pyr', (F, FA)), ('cnn', (O, OA))):
        d = np.mean([pred(m, a) - pred(m, b) for m in S_['M'][k]], 0)
        for i, x in zip(I, d): V[i][k] = max(V[i][k], float(x))
    for i in I:
        w, wa, rel = V[i]['sites'][ty]; an = S_['anc']; lo, hi, f = (an - 3, an + 6, f5) if ty == 'D' else (an - 20, an + 3, f3); j = an + rel; e = V[i]
        if lo <= j < hi and 'N' not in w[lo:hi]: e['mes'] = max(e['mes'], f(w[lo:hi]) - f(wa[lo:hi])); e['motif'] = True
        e['canon'] = e['canon'] or (ty == 'D' and rel in (0, 1)) or (ty == 'A' and rel in (-2, -1)); e['dist'] = min(e['dist'], abs(rel)); e['exon'] = e['exon'] or (ty == 'D' and rel < 0 and e['dist'] == abs(rel)) or (ty == 'A' and rel >= 0 and e['dist'] == abs(rel))
y = np.array([e['label'] for e in V]); z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
S = {k: np.array([e[k] for e in V], float) for k in ('pyr', 'cnn', 'mes')}; S['dist'] = -np.array([e['dist'] for e in V], float)
S['pyr+mes'] = z(S['pyr']) + z(S['mes']); S['cnn+mes'] = z(S['cnn']) + z(S['mes']); S['pyr+cnn'] = z(S['pyr']) + z(S['cnn']); S['pyr+cnn+mes'] = z(S['pyr']) + z(S['cnn']) + z(S['mes'])
can = np.array([e['canon'] for e in V]); mot = np.array([e['motif'] for e in V]); exo = np.array([e['exon'] for e in V]); res = {}; rs = np.random.RandomState(0)
for nm, m in (('all', np.ones(len(V), bool)), ('noncanonical', ~can), ('motif_noncanon', mot & ~can), ('outside_motif', ~mot), ('outside_exonic', ~mot & exo), ('outside_intronic', ~mot & ~exo)):
    if not (0 < y[m].sum() < m.sum()): continue
    r = dict(n=int(m.sum()), n_pos=int(y[m].sum()))
    for k, v in S.items(): r['auc_' + k] = float(roc_auc_score(y[m], v[m])); r['ap_' + k] = float(AP(y[m], v[m]))
    ym = y[m]; o = []
    for _ in range(500):
        i = rs.randint(0, len(ym), len(ym))
        if 0 < ym[i].sum() < len(i): o.append([AP(ym[i], S[a][m][i]) - AP(ym[i], S[b][m][i]) for a, b in (('pyr+mes', 'mes'), ('pyr', 'mes'), ('pyr', 'cnn'), ('pyr+cnn+mes', 'cnn+mes'))])
    o = np.array(o); r['ci'] = {n_: [float(np.percentile(o[:, j], 2.5)), float(np.percentile(o[:, j], 97.5))] for j, n_ in enumerate(('pyr+mes-mes', 'pyr-mes', 'pyr-cnn', 'pyr+cnn+mes-cnn+mes'))}; res[nm] = r
    log('MF', nm, 'n', r['n'], 'pos', r['n_pos'], ' '.join('%s %.3f/%.3f' % (k, r['auc_' + k], r['ap_' + k]) for k in S)); log('MFCI', nm, {k: [round(x, 3) for x in v] for k, v in r['ci'].items()})
json.dump(dict(mapping=dict(stt), strata=res), open('results/mfass.json', 'w'), indent=1)
with open('results/mfass_scores.tsv', 'w') as fo:
    fo.write('id\tchrom\tpos\tstrand\tlabel\tdpsi\tpyr\tcnn\tmes\tdist\tmotif\tcanon\n')
    for e in V: fo.write('\t'.join(str(e[k]) for k in ('id', 'chrom', 'pos', 'strand', 'label', 'dpsi', 'pyr', 'cnn', 'mes', 'dist', 'motif', 'canon')) + '\n')
log('MFASS_DONE')
