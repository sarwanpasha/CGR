# Does orienting each window by genome annotation change what a model can learn? usage: orientation.py <task> [seed]
# Benchmark sequences are given on the forward genomic strand (header ">chr:start-end|label"). Using GENCODE gene
# records only (no labels), every window is optionally reverse-complemented so that
#   strand : the nearest gene is transcribed left to right          tss : the nearest TSS lies to the right of the window centre
# The base pyramid network is trained on the original and on the re-oriented sequences. Also reports how well the
# distance to the nearest TSS alone predicts the label, and for 2-class tasks the fraction of positives per orientation.
import sys, gzip, json, time, collections, numpy as np, torch, torch.nn as nn
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
try: import cgr_features as PP
except ImportError: import P as PP
T = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
GTF = 'data/cv/gencode.v44.annotation.gtf.gz'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time()
BASE = [(5, 1), (4, 4), (3, 16), (3, 64), (2, 0)]; COMP = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
def rcmap(k):
    n = 4 ** k; idx = np.arange(n); r = np.zeros(n, np.int64)
    for j in range(k): r = r * 4 + (3 - (idx // 4 ** j) % 4)
    u, inv = np.unique(np.minimum(idx, r), return_inverse=True); return inv, len(u)
CAN = not (T.startswith('splice') or T == 'promoter_tata'); W = {k: rcmap(k) for k in (2, 3, 4, 5)} if CAN else None
def load(split):
    H, S, Y = [], [], []
    for l in open('data/nt/%s_%s.fna' % (T, split)):
        l = l.strip()
        if not l: continue
        if l[0] == '>':
            a, y = l[1:].rsplit('|', 1); c, r = a.split(':'); s, e = r.split('-'); H.append((c, (int(s) + int(e)) // 2)); Y.append(int(y))
        else: S.append(l.upper())
    return H, S, np.array(Y)
def genes():
    G = collections.defaultdict(list)
    for l in gzip.open(GTF, 'rt'):
        if l[0] == '#': continue
        f = l.split('\t', 9)
        if f[2] != 'gene' or 'gene_type "protein_coding"' not in f[8]: continue
        s, e = int(f[3]), int(f[4]); G[f[0]].append((s if f[6] == '+' else e, 1 if f[6] == '+' else -1))
    return {c: (np.array(sorted(v))[:, 0], np.array(sorted(v))[:, 1]) for c, v in G.items()}
def annotate(H, G):
    d = np.zeros(len(H)); st = np.zeros(len(H), int)
    for i, (c, x) in enumerate(H):
        if c not in G: d[i] = 1e7; st[i] = 1; continue
        p, s = G[c]; j = np.searchsorted(p, x); cand = [k for k in (j - 1, j) if 0 <= k < len(p)]; k = min(cand, key=lambda k: abs(p[k] - x))
        d[i] = p[k] - x; st[i] = s[k]
    return d, st
def orient(S, flip): return [q.translate(COMP)[::-1] if f else q for q, f in zip(S, flip)]
class Pyr(nn.Module):
    def __init__(s, DL, NC, h=64, p=0.3):
        super().__init__(); s.DL = DL
        s.br = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)) for d in DL]); s.out = nn.Linear(h * len(DL), NC)
    def forward(s, x): return s.out(torch.cat([b(v) for b, v in zip(s.br, torch.split(x, s.DL, 1))], 1))
def prob(m, A, bs=1024):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(A), bs): o.append(torch.softmax(m(torch.tensor(A[i:i + bs]).to(dev)), 1).cpu().numpy())
    return np.concatenate(o)
def run(S, Y, St, Yt):
    L = len(S[0]); NC = int(Y.max()) + 1; DL = PP.dims(L, BASE, W)
    X = np.stack([PP.feat(q, L, BASE, W) for q in S]); Xt = np.stack([PP.feat(q, L, BASE, W) for q in St]); X, Xt = PP.norm(X, Xt); X = X.astype(np.float32); Xt = Xt.astype(np.float32)
    torch.manual_seed(SEED); rs = np.random.RandomState(SEED); tr, va = train_test_split(np.arange(len(Y)), test_size=0.1, stratify=Y, random_state=SEED)
    m = Pyr(DL, NC).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); best = (-2, None); pat = 0
    for ep in range(40):
        m.train()
        for b in np.array_split(rs.permutation(tr), max(1, len(tr) // 128)):
            opt.zero_grad(); nn.functional.cross_entropy(m(torch.tensor(X[b]).to(dev)), torch.tensor(Y[b]).to(dev)).backward(); opt.step()
        v = matthews_corrcoef(Y[va], prob(m, X[va]).argmax(1))
        if v > best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()}); pat = 0
        else:
            pat += 1
            if pat >= 8: break
    m.load_state_dict(best[1]); return float(matthews_corrcoef(Yt, prob(m, Xt).argmax(1)))
G = genes(); (H, S, Y), (Ht, St, Yt) = load('train'), load('test'); (d, st), (dt, stt) = annotate(H, G), annotate(Ht, G)
R = {'orig': run(S, Y, St, Yt), 'strand': run(orient(S, st < 0), Y, orient(St, stt < 0), Yt), 'tss': run(orient(S, d < 0), Y, orient(St, dt < 0), Yt)}
pos = Y == Y.max(); info = dict(n=len(Y), n_test=len(Yt), median_abs_dist_pos=float(np.median(np.abs(d[pos]))), median_abs_dist_neg=float(np.median(np.abs(d[~pos]))),
            auc_dist=float(roc_auc_score(pos, -np.abs(d))), frac_pos_minus=float(pos[st < 0].mean()), frac_pos_plus=float(pos[st > 0].mean()), frac_pos_tss_left=float(pos[d < 0].mean()), frac_pos_tss_right=float(pos[d >= 0].mean()))
json.dump(dict(task=T, seed=SEED, mcc=R, info=info, sec=time.time() - t0), open('results/orient_%s_s%d.json' % (T, SEED), 'w'))
print('ORIENT', T, {k: round(v, 3) for k, v in R.items()}, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, '(%.0fs)' % (time.time() - t0), flush=True)
