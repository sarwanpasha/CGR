# Pangolin on the MFASS variants scored by mfass_splice.py. usage: mfass_pangolin.py [threads]
# Needs a clone of https://github.com/tkzeng/Pangolin in ./pangolin_src (model definition and released weights).
# Each variant is scored with the 12 released models (4 tissues x 3 replicates) on 5,050 nt of context per side, in the
# orientation of the transcript. Score: the largest absolute change of the splice-site probability over the scored
# positions and the 4 tissues, after averaging the 3 replicates. Compared with SpliceAI, MaxEntScan and the pyramid
# score overall and by region, with paired bootstrap intervals of differences in average precision.
import os, sys, gzip, csv, json, time, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
sys.path.insert(0, 'pangolin_src'); from pangolin.model import Pangolin, L as PL, W as PW, AR as PAR
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 16); t0 = time.time()
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
R_ = list(csv.reader(open('data/mfass/snv_data_clean.txt'), delimiter='\t')); H = [h.strip('"') for h in R_[0]]; ix = {h: i for i, h in enumerate(H)}
info = {}
for r in R_[1:]:
    r = [x.strip('"') for x in r]; info[r[0]] = r
V = [l.rstrip('\n').split('\t') for l in open('results/mfass_scores.tsv')]; vh = V[0]; V = V[1:]; vx = {h: i for i, h in enumerate(vh)}
G = {}; cur = None; buf = []; want = set(v[vx['chrom']] for v in V)
for l in gzip.open('data/cv/GRCh38.primary_assembly.genome.fa.gz', 'rt'):
    if l[0] == '>':
        if cur in want: G[cur] = ''.join(buf).upper()
        cur = l[1:].split()[0]; buf = []
    elif cur in want: buf.append(l.rstrip())
if cur in want: G[cur] = ''.join(buf).upper()
log('variants', len(V), 'genome', len(G))
COMP = str.maketrans('ACGTN', 'TGCAN'); LUT = np.full(256, 4, np.int64)
for i, c in enumerate('ACGT'): LUT[ord(c)] = i
EYE = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)]); F = 5050
rows = []
for v in V:
    r = info.get(v[vx['id']]); c = v[vx['chrom']]; p = int(v[vx['pos']]); sd = v[vx['strand']]
    if r is None: continue
    ref = r[ix['ref_allele']].upper(); alt = r[ix['alt_allele']].upper(); g = G[c]
    if g[p - 1] != ref:
        if g[p - 1] == ref.translate(COMP): alt = alt.translate(COMP)
        else: continue
    s = g[p - 1 - F:p + F]
    if len(s) != 2 * F + 1: continue
    a = s[:F] + alt + s[F + 1:]
    if sd == '-': s = s.translate(COMP)[::-1]; a = a.translate(COMP)[::-1]
    rows.append(dict(v=v, s=s, a=a))
log('kept', len(rows))
dev = torch.device('cpu'); mods = []
for i in (0, 2, 4, 6):
    grp = []
    for j in (1, 2, 3):
        m = Pangolin(PL, PW, PAR); m.load_state_dict(torch.load('pangolin_src/pangolin/models/final.%d.%d.3.v2' % (j, i), map_location=dev)); m.eval(); grp.append(m)
    mods.append(grp)
CH = (1, 4, 7, 10); log('models', sum(len(g) for g in mods))
def oh(q): return EYE[LUT[np.frombuffer(q.encode(), np.uint8)]].T
def score(X):
    X = torch.from_numpy(X); out = []
    with torch.no_grad():
        for grp, ch in zip(mods, CH): out.append(torch.stack([m(X)[:, ch, :] for m in grp]).mean(0).numpy())
    return np.stack(out, 1)
cache = 'results/mfass_pangolin_scores.json'; out = json.load(open(cache)) if os.path.exists(cache) else {}; BS = 16
todo = [i for i, r in enumerate(rows) if r['v'][vx['id']] not in out]
for b in range(0, len(todo), BS):
    J = todo[b:b + BS]; Rr = score(np.stack([oh(rows[j]['s']) for j in J])); A = score(np.stack([oh(rows[j]['a']) for j in J]))
    d = np.abs(A - Rr).max((1, 2))
    for j, x in zip(J, d): out[rows[j]['v'][vx['id']]] = float(x)
    if (b // BS) % 40 == 0: log('done', b + len(J), 'of', len(todo)); json.dump(out, open(cache, 'w'))
json.dump(out, open(cache, 'w'))
sai = json.load(open('results/mfass_spliceai_scores.json')) if os.path.exists('results/mfass_spliceai_scores.json') else {}
rows = [r for r in rows if r['v'][vx['id']] in out and (not sai or r['v'][vx['id']] in sai)]
y = np.array([int(r['v'][vx['label']]) for r in rows]); z = lambda q: (q - q.mean()) / (q.std() + 1e-9)
S = {'pan': np.array([out[r['v'][vx['id']]] for r in rows]), 'pyr': np.array([float(r['v'][vx['pyr']]) for r in rows]), 'mes': np.array([float(r['v'][vx['mes']]) for r in rows])}
S['pyr+pan'] = z(S['pyr']) + z(S['pan']); PAIRS = [('pyr+pan', 'pan'), ('pyr', 'pan')]
if sai:
    S['sai'] = np.array([sai[r['v'][vx['id']]] for r in rows]); S['sai+pan'] = z(S['sai']) + z(S['pan']); S['pyr+sai+pan'] = z(S['pyr']) + S['sai+pan']; PAIRS.append(('pyr+sai+pan', 'sai+pan'))
mot = np.array([r['v'][vx['motif']] == 'True' for r in rows]); can = np.array([r['v'][vx['canon']] == 'True' for r in rows])
res = {}; rs = np.random.RandomState(0)
for nm, m in (('all', np.ones(len(rows), bool)), ('noncanonical', ~can), ('motif_noncanon', mot & ~can), ('outside_motif', ~mot)):
    if not (0 < y[m].sum() < m.sum()): continue
    rr = dict(n=int(m.sum()), n_pos=int(y[m].sum()))
    for k, q in S.items(): rr['auc_' + k] = float(AU(y[m], q[m])); rr['ap_' + k] = float(AP(y[m], q[m]))
    ym = y[m]; o = []
    for _ in range(500):
        i = rs.randint(0, len(ym), len(ym))
        if 0 < ym[i].sum() < len(i): o.append([AP(ym[i], S[a][m][i]) - AP(ym[i], S[bb][m][i]) for a, bb in PAIRS])
    o = np.array(o); rr['ci'] = {'%s-(%s)' % p: [float(np.percentile(o[:, j], 2.5)), float(np.percentile(o[:, j], 97.5))] for j, p in enumerate(PAIRS)}
    res[nm] = rr; log('MP', nm, 'n', rr['n'], 'pos', rr['n_pos'], ' '.join('%s %.3f/%.3f' % (k, rr['auc_' + k], rr['ap_' + k]) for k in S)); log('MPCI', nm, {k: [round(x, 3) for x in v] for k, v in rr['ci'].items()})
json.dump(res, open('results/mfass_pangolin.json', 'w'), indent=1); log('MP_DONE')
