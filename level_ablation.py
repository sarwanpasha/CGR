# Which positional resolution carries the signal of a task? usage: level_ablation.py <task> [seed]
# Trains the pyramid network on (a) all levels, (b) each level alone, (c) all levels but one, and
# (d) single levels with the k-mer length fixed to 3 at 1, 4, 16, 64 and 256 bins, so that resolution
# is varied without changing k. Writes results/lev_<task>_s<seed>.json with the test MCC of every run.
import sys, json, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
try: import cgr_features as PP
except ImportError: import P as PP
T = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time()
BASE = [(5, 1), (4, 4), (3, 16), (3, 64), (2, 0)]; K3 = [(3, 1), (3, 4), (3, 16), (3, 64), (3, 256)]
def rcmap(k):
    n = 4 ** k; idx = np.arange(n); r = np.zeros(n, np.int64)
    for j in range(k): r = r * 4 + (3 - (idx // 4 ** j) % 4)
    u, inv = np.unique(np.minimum(idx, r), return_inverse=True); return inv, len(u)
CAN = not (T.startswith('splice') or T == 'promoter_tata')
W = {k: rcmap(k) for k in (2, 3, 4, 5)} if CAN else None
S, Y = PP.read(T, 'train'); St, Yt = PP.read(T, 'test'); L = len(S[0]); NC = int(Y.max()) + 1
LEVS = BASE + K3; DL = PP.dims(L, LEVS, W); cuts = np.cumsum([0] + DL)
X = np.stack([PP.feat(q, L, LEVS, W) for q in S]); Xt = np.stack([PP.feat(q, L, LEVS, W) for q in St]); X, Xt = PP.norm(X, Xt)
X = X.astype(np.float32); Xt = Xt.astype(np.float32)
class Pyr(nn.Module):
    def __init__(s, DL, h=64, p=0.3):
        super().__init__(); s.DL = DL
        s.br = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)) for d in DL])
        s.out = nn.Linear(h * len(DL), NC)
    def forward(s, x):
        return s.out(torch.cat([b(v) for b, v in zip(s.br, torch.split(x, s.DL, 1))], 1))
def prob(m, A, bs=1024):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(A), bs): o.append(torch.softmax(m(torch.tensor(A[i:i + bs]).to(dev)), 1).cpu().numpy())
    return np.concatenate(o)
def run(ids):
    col = np.concatenate([np.arange(cuts[i], cuts[i + 1]) for i in ids]); A = X[:, col]; At = Xt[:, col]; dl = [DL[i] for i in ids]
    torch.manual_seed(SEED); rs = np.random.RandomState(SEED)
    tr, va = train_test_split(np.arange(len(Y)), test_size=0.1, stratify=Y, random_state=SEED)
    m = Pyr(dl).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); best = (-2, None); pat = 0
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
name = lambda i: '(%d,%s)' % (LEVS[i][0], LEVS[i][1] if LEVS[i][1] else 'L')
R = {'all': run([0, 1, 2, 3, 4])}
for i in range(5): R['only' + name(i)] = run([i]); R['drop' + name(i)] = run([j for j in range(5) if j != i])
for i in range(5, 10): R['k3only' + name(i)] = run([i])
json.dump(dict(task=T, seed=SEED, L=L, canonical=CAN, mcc=R, sec=time.time() - t0), open('results/lev_%s_s%d.json' % (T, SEED), 'w'))
print('LEV', T, {k: round(v, 3) for k, v in R.items()}, '(%.0fs)' % (time.time() - t0), flush=True)
