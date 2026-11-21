# Do position-binned counts help histone-mark tasks through the arrangement of composition along the window?
# usage: region_contrast.py <task> [seed].  4-mer counts (reverse-complement merged) are given to the same network as
#   R1   : one bin (whole sequence)            R4   : 4 bins in their true order
#   perm : 4 bins, order shuffled per sequence  cf   : centre (bins 2+3) and flank (bins 1+4)
#   lr   : left (bins 1+2) and right (bins 3+4)
# Also stores the class-1 minus class-0 rate of the 10 most class-enriched and 10 most depleted 4-mers in 20 bins.
import sys, json, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
try: import cgr_features as PP
except ImportError: import P as PP
T = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time(); K = 4
def rcmap(k):
    n = 4 ** k; idx = np.arange(n); r = np.zeros(n, np.int64)
    for j in range(k): r = r * 4 + (3 - (idx // 4 ** j) % 4)
    u, inv = np.unique(np.minimum(idx, r), return_inverse=True); return inv, len(u)
W = {K: rcmap(K)}; M = W[K][1]
S, Y = PP.read(T, 'train'); St, Yt = PP.read(T, 'test'); L = len(S[0]); NC = int(Y.max()) + 1
def counts(Q, R): return np.stack([np.expm1(PP.feat(q, L, [(K, R)], W)) for q in Q]).reshape(len(Q), R, M)
C4, C4t, C20 = counts(S, 4), counts(St, 4), counts(S, 20)
rs0 = np.random.RandomState(123)
def perm(C): return np.stack([c[rs0.permutation(4)] for c in C])
def views(C):
    return {'R1': C.sum(1, keepdims=True), 'R4': C, 'perm': perm(C), 'cf': np.stack([C[:, 1] + C[:, 2], C[:, 0] + C[:, 3]], 1), 'lr': np.stack([C[:, 0] + C[:, 1], C[:, 2] + C[:, 3]], 1)}
def prob(m, A, bs=1024):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(A), bs): o.append(torch.softmax(m(torch.tensor(A[i:i + bs]).to(dev)), 1).cpu().numpy())
    return np.concatenate(o)
def run(A, At):
    A = np.log1p(A.reshape(len(A), -1)); At = np.log1p(At.reshape(len(At), -1)); A, At = PP.norm(A, At); A = A.astype(np.float32); At = At.astype(np.float32)
    torch.manual_seed(SEED); rs = np.random.RandomState(SEED)
    tr, va = train_test_split(np.arange(len(Y)), test_size=0.1, stratify=Y, random_state=SEED)
    m = nn.Sequential(nn.Dropout(0.1), nn.Linear(A.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, NC)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); best = (-2, None); pat = 0
    for ep in range(40):
        m.train()
        for b in np.array_split(rs.permutation(tr), max(1, len(tr) // 128)):
            opt.zero_grad(); nn.functional.cross_entropy(m(torch.tensor(A[b]).to(dev)), torch.tensor(Y[b]).to(dev)).backward(); opt.step()
        v = matthews_corrcoef(Y[va], prob(m, A[va]).argmax(1))
        if v > best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()}); pat = 0
        else:
            pat += 1
            if pat >= 8: break
    m.load_state_dict(best[1]); return float(matthews_corrcoef(Yt, prob(m, At).argmax(1)))
V, Vt = views(C4), views(C4t); R = {k: run(V[k], Vt[k]) for k in V}
pos = Y == Y.max(); d = C20[pos].mean(0) - C20[~pos].mean(0); tot = d.sum(0); order = np.argsort(tot)
prof = {'enriched': d[:, order[-10:]].sum(1).tolist(), 'depleted': d[:, order[:10]].sum(1).tolist(), 'rate_pos': C20[pos].mean(0).sum(1).tolist()}
json.dump(dict(task=T, seed=SEED, L=L, mcc=R, profile=prof, sec=time.time() - t0), open('results/contrast_%s_s%d.json' % (T, SEED), 'w'))
print('CONTRAST', T, {k: round(v, 3) for k, v in R.items()}, 'enriched', [round(x, 3) for x in prof['enriched']], '(%.0fs)' % (time.time() - t0), flush=True)
