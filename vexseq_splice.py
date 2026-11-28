# Vex-seq variants (Adamson et al. 2018) scored with the pyramid splice-site models. usage: vexseq_splice.py [seeds=3]
# Downloads the processed Vex-seq tables (change of percent spliced in and distances to the splice sites) from the
# authors' repository and the hg19 chromosomes from UCSC, recovers the boundaries and strand of each assayed exon from
# the distances (the combination that gives AG and GT at the recovered boundaries), trains the pyramid network on the
# benchmark donor and acceptor tasks, and scores every single-nucleotide variant as in mfass_splice.py.
# Writes results/vex_scores.tsv (one row per variant: exon, measured change, pyramid score, MaxEntScan) for
# vexseq_specialized.py.
import sys, os, gzip, json, time, collections, urllib.request, numpy as np, cgr_features as PP
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 3; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time()
COMP = str.maketrans('ACGTN', 'TGCAN')
def rc(s): return s.translate(COMP)[::-1]
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
def get(u, p):
    if not os.path.exists(p): open(p, 'wb').write(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'python'}), timeout=900).read())
    return p
os.makedirs('data/vex', exist_ok=True); B = 'https://raw.githubusercontent.com/scottiadamson/Vex-seq/master/processed_files/'
D = [l.rstrip('\n').split('\t') for l in open(get(B + 'delta_PSI_values.tsv', 'data/vex/delta_PSI_values.tsv'))]; dh = {h: i for i, h in enumerate(D[0])}
P = {l.split('\t')[0]: l.rstrip('\n').split('\t')[1:] for l in open(get(B + 'Vex-seq_positions.tsv', 'data/vex/Vex-seq_positions.tsv'))}
rows = []
for r in D[1:]:
    f = r[0].split('_')
    if len(f) != 4 or len(f[2]) != 1 or len(f[3]) != 1 or r[0] not in P: continue
    try: rows.append(dict(id=r[0], chrom=f[0], pos=int(f[1]), ref=f[2].upper(), alt=f[3].upper(), d3=int(P[r[0]][0]), d5=int(P[r[0]][1]), dpsi=float(r[dh['delta_PSI']])))
    except Exception: pass
log('single-nucleotide variants with positions', len(rows)); G = {}
for c in sorted(set(r['chrom'] for r in rows)):
    G[c] = ''.join(l.strip() for l in gzip.open(get('https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/%s.fa.gz' % c, 'data/vex/%s.fa.gz' % c), 'rt') if l[0] != '>').upper()
log('hg19 chromosomes', len(G)); open('data/vex/hg19_vex.fa', 'w').write(''.join('>%s\n%s\n' % (c, G[c]) for c in G))
def exon(r, sd, da, de, sg=1):
    # first and last exon base in genome order under one hypothesis of strand and offsets
    if sd == '+': a = r['pos'] - sg * r['d3'] + da; e = r['pos'] - sg * r['d5'] + de; return a, e
    a = r['pos'] + sg * r['d3'] + da; e = r['pos'] + sg * r['d5'] + de; return e, a
def canon(r, sd, s1, e1):
    g = G[r['chrom']]
    if not (10 < s1 <= e1 < len(g) - 10): return False
    return (g[s1 - 3:s1 - 1] == 'AG' and g[e1:e1 + 2] == 'GT') if sd == '+' else (g[e1:e1 + 2] == 'CT' and g[s1 - 3:s1 - 1] == 'AC')
best = {}
for sd in '+-':
    cnt = {(da, de, sg): sum(canon(r, sd, *exon(r, sd, da, de, sg)) for r in rows) for da in (-1, 0, 1) for de in (-1, 0, 1) for sg in (1, -1)}; k = max(cnt, key=cnt.get); best[sd] = k; log('strand', sd, 'best offsets and sign', k, 'canonical', cnt[k], 'second', sorted(cnt.values())[-2])
V = []
for r in rows:
    ok = []
    for sd in '+-':
        # the distances skip or count the junction differently for exonic and intronic variants, so offsets of 1 are tried per variant
        c = sorted(set(exon(r, sd, da, de, best[sd][2]) for da in (-1, 0, 1) for de in (-1, 0, 1) if canon(r, sd, *exon(r, sd, da, de, best[sd][2]))))
        if len(c) == 1: ok.append((sd,) + c[0])
        elif len(c) > 1: ok += [None, None]
    g = G[r['chrom']][r['pos'] - 1]
    if len(ok) != 1 or g != r['ref']: continue
    r['strand'], r['s1'], r['e1'] = ok[0]; V.append(r)
log('kept (one strand with AG and GT, reference allele matches)', len(V), 'of', len(rows), 'strands', dict(collections.Counter(r['strand'] for r in V)))
class Pyr(nn.Module):
    def __init__(s, DL, h=64, p=0.3):
        super().__init__(); s.DL = DL
        s.br = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)) for d in DL]); s.out = nn.Linear(h * len(DL), 1)
    def forward(s, x):
        o = []; i = 0
        for d, b in zip(s.DL, s.br): o.append(b(x[:, i:i + d])); i += d
        return s.out(torch.cat(o, 1)).squeeze(1)
def pred(m, X, bs=512):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(X), bs): o.append(m(torch.tensor(X[i:i + bs]).to(dev)).cpu().numpy())
    return np.concatenate(o)
def train(mk, X, Y, seed, ep=20, bs=128, lr=1e-3):
    torch.manual_seed(seed); rs = np.random.RandomState(seed); idx = rs.permutation(len(Y)); nv = len(Y) // 10; va, tr = idx[:nv], idx[nv:]
    m = mk().to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5); Yf = Y.astype(np.float32); bst = (-1, None)
    for e in range(ep):
        m.train()
        for b in np.array_split(rs.permutation(tr), max(1, len(tr) // bs)):
            opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(torch.tensor(X[b]).to(dev)), torch.tensor(Yf[b]).to(dev)).backward(); opt.step()
        a = roc_auc_score(Y[va], pred(m, X[va]))
        if a > bst[0]: bst = (a, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(bst[1]); return m, bst[0]
from maxentpy import maxent
try: m5, m3 = maxent.load_matrix5(), maxent.load_matrix3(); f5 = lambda q: maxent.score5(q, matrix=m5); f3 = lambda q: maxent.score3(q, matrix=m3)
except Exception: f5, f3 = maxent.score5, maxent.score3
for r in V: r['pyr'] = -1e9; r['mes'] = 0.0; r['motif'] = False; r['dist'] = 999
for ty, T, dn in (('D', 'splice_sites_donors', 'GT'), ('A', 'splice_sites_acceptors', 'AG')):
    S, Y = PP.read(T, 'train'); L = len(S[0]); fr = {c: np.mean([s[L // 2 - 20:L // 2 + 20].count(dn) > 0 for s, y in zip(S, Y) if y == c]) for c in set(Y.tolist())}
    pc = max(fr, key=fr.get); Y = (Y == pc).astype(int); pos = [s for s, y in zip(S, Y) if y == 1]
    off = max(range(L // 2 - 20, L // 2 + 20), key=lambda i: sum(s[i:i + 2] == dn for s in pos)); anc = off if ty == 'D' else off + 2
    X = np.stack([PP.feat(q, L) for q in S]); mu = X.mean(0); sd = X.std(0) + 1e-6; X = (X - mu) / sd; DL = PP.dims(L); M = []
    for s_ in range(NS):
        m, a = train(lambda: Pyr(DL), X, Y, s_); M.append(m); log(ty, 'seed', s_, 'validation AUC', round(a, 4))
    del X; I = []; Wr = []; Wa = []
    for i, r in enumerate(V):
        c = r['chrom']; an = (r['e1'] + 1 if r['strand'] == '+' else r['s1'] - 1) if ty == 'D' else (r['s1'] if r['strand'] == '+' else r['e1'])
        if r['strand'] == '+': a0 = an - anc; w = G[c][a0 - 1:a0 - 1 + L]; j = r['pos'] - a0; al = r['alt']
        else: e0 = an + anc; a0 = e0 - L + 1; w = rc(G[c][a0 - 1:e0]); j = e0 - r['pos']; al = r['alt'].translate(COMP)
        if len(w) != L or not (0 <= j < L) or w[off:off + 2] != dn: continue
        wa = w[:j] + al + w[j + 1:]; I.append(i); Wr.append(w); Wa.append(wa); rel = j - anc; lo, hi, f = (anc - 3, anc + 6, f5) if ty == 'D' else (anc - 20, anc + 3, f3)
        if lo <= j < hi and 'N' not in w[lo:hi]: r['mes'] = max(r['mes'], f(w[lo:hi]) - f(wa[lo:hi])); r['motif'] = True
        r['dist'] = min(r['dist'], abs(rel))
    F = (np.stack([PP.feat(q, L) for q in Wr]) - mu) / sd; FA = (np.stack([PP.feat(q, L) for q in Wa]) - mu) / sd; d = np.mean([pred(m, F.astype(np.float32)) - pred(m, FA.astype(np.float32)) for m in M], 0)
    for i, x in zip(I, d): V[i]['pyr'] = max(V[i]['pyr'], float(x))
    log(ty, 'variants scored', len(I))
V = [r for r in V if r['pyr'] > -1e8]; K = ('id', 'chrom', 'pos', 'ref', 'alt', 'strand', 's1', 'e1', 'dpsi', 'pyr', 'mes', 'motif', 'dist')
open('results/vex_scores.tsv', 'w').write('\t'.join(K) + '\n' + ''.join('\t'.join(str(r[k]) for k in K) + '\n' for r in V))
from scipy.stats import spearmanr
y = np.array([r['dpsi'] for r in V]); lab = (y <= -10).astype(int)
for k in ('pyr', 'mes'):
    q = np.array([r[k] for r in V]); log('VX', k, 'n', len(V), 'loss<=-10:', int(lab.sum()), 'spearman with -dPSI %.3f' % spearmanr(q, -y)[0], 'AUC %.3f' % roc_auc_score(lab, q))
log('VX_DONE')
