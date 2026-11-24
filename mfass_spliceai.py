# SpliceAI on the MFASS variants scored by mfass_splice.py. usage: mfass_spliceai.py
# Reads results/mfass_scores.tsv and data/mfass/snv_data_clean.txt, scores each variant with the 5 SpliceAI models
# (10,000 nt of context, delta = largest change of acceptor or donor probability), and compares with the pyramid score
# and MaxEntScan overall and by region, with paired bootstrap intervals of differences in average precision.
import os, gzip, csv, json, glob, time, importlib.util, numpy as np
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
t0 = time.time()
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
R_ = list(csv.reader(open('data/mfass/snv_data_clean.txt'), delimiter='\t')); H = [h.strip('"') for h in R_[0]]; ix = {h: i for i, h in enumerate(H)}
info = {}
for r in R_[1:]:
    r = [x.strip('"') for x in r]; info[r[0]] = r
cS = [i for i, h in enumerate(H) if 'start_hg38' in h and 'snp' not in h][0]
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
def num(x):
    try: return int(float(x))
    except Exception: return None
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
    s0 = num(r[cS]); i1 = num(r[ix['intron1_len']]); ex = num(r[ix['exon_len']]); s1 = s0 + i1 + 1; e1 = s1 + ex - 1
    rows.append(dict(v=v, s=s, a=a, exon=(s1 <= p <= e1)))
log('kept', len(rows))
import tensorflow as tf
from tensorflow.keras.models import load_model
mods = [load_model(f, compile=False) for f in sorted(glob.glob(os.path.join(os.path.dirname(importlib.util.find_spec('spliceai').origin), 'models', 'spliceai*.h5')))]
log('models', len(mods), 'gpus', len(tf.config.list_physical_devices('GPU')))
def oh(q): return EYE[LUT[np.frombuffer(q.encode(), np.uint8)]]
def score(X): return np.mean([m.predict(X, batch_size=16, verbose=0) for m in mods], 0)
cache = 'results/mfass_spliceai_scores.json'; out = json.load(open(cache)) if os.path.exists(cache) else {}; BS = 64
todo = [i for i, r in enumerate(rows) if r['v'][vx['id']] not in out]
for b in range(0, len(todo), BS):
    J = todo[b:b + BS]; Rr = score(np.stack([oh(rows[j]['s']) for j in J])); A = score(np.stack([oh(rows[j]['a']) for j in J]))
    d = np.abs(A[:, :, 1:] - Rr[:, :, 1:]).max((1, 2))
    for j, x in zip(J, d): out[rows[j]['v'][vx['id']]] = float(x)
    if (b // BS) % 20 == 0: log('done', b + len(J), 'of', len(todo)); json.dump(out, open(cache, 'w'))
json.dump(out, open(cache, 'w'))
y = np.array([int(r['v'][vx['label']]) for r in rows]); z = lambda q: (q - q.mean()) / (q.std() + 1e-9)
S = {'sai': np.array([out[r['v'][vx['id']]] for r in rows]), 'pyr': np.array([float(r['v'][vx['pyr']]) for r in rows]), 'mes': np.array([float(r['v'][vx['mes']]) for r in rows])}
S['pyr+sai'] = z(S['pyr']) + z(S['sai']); S['sai+mes'] = z(S['sai']) + z(S['mes']); S['all3'] = z(S['pyr']) + z(S['sai']) + z(S['mes'])
mot = np.array([r['v'][vx['motif']] == 'True' for r in rows]); can = np.array([r['v'][vx['canon']] == 'True' for r in rows]); exo = np.array([r['exon'] for r in rows])
res = {}; rs = np.random.RandomState(0)
for nm, m in (('all', np.ones(len(rows), bool)), ('noncanonical', ~can), ('motif_noncanon', mot & ~can), ('outside_motif', ~mot), ('outside_exonic', ~mot & exo)):
    if not (0 < y[m].sum() < m.sum()): continue
    rr = dict(n=int(m.sum()), n_pos=int(y[m].sum()))
    for k, q in S.items(): rr['auc_' + k] = float(AU(y[m], q[m])); rr['ap_' + k] = float(AP(y[m], q[m]))
    ym = y[m]; o = []
    for _ in range(500):
        i = rs.randint(0, len(ym), len(ym))
        if 0 < ym[i].sum() < len(i): o.append([AP(ym[i], S[a][m][i]) - AP(ym[i], S[bb][m][i]) for a, bb in (('pyr+sai', 'sai'), ('pyr', 'sai'), ('all3', 'sai+mes'))])
    o = np.array(o); rr['ci'] = {n_: [float(np.percentile(o[:, j], 2.5)), float(np.percentile(o[:, j], 97.5))] for j, n_ in enumerate(('pyr+sai-sai', 'pyr-sai', 'all3-(sai+mes)'))}
    res[nm] = rr; log('MS', nm, 'n', rr['n'], 'pos', rr['n_pos'], ' '.join('%s %.3f/%.3f' % (k, rr['auc_' + k], rr['ap_' + k]) for k in S)); log('MSCI', nm, {k: [round(x, 3) for x in v] for k, v in rr['ci'].items()})
json.dump(res, open('results/mfass_spliceai.json', 'w'), indent=1); log('MS_DONE')
