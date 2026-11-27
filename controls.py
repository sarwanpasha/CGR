# Two controls for the pyramid on one task (CPU, PyTorch). usage: controls.py <task> select|gc [seed]
# select : train one network per pyramid level and one on all levels, keep validation and test MCC, and report the
#          test MCC of the single level with the best validation MCC
# gc     : subsample training and test sets so that positive and negative windows have the same GC histogram (1% bins),
#          then train on whole-sequence 4-mers, on 4 bins of 4-mers, and on all levels
import sys, json, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import cgr_features as PP
T, MODE = sys.argv[1], sys.argv[2]; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0; torch.set_num_threads(2); t0 = time.time()
BASE = [(5, 1), (4, 4), (3, 16), (3, 64), (2, 0)]
def rcmap(k):
    n = 4 ** k; idx = np.arange(n); r = np.zeros(n, np.int64)
    for j in range(k): r = r * 4 + (3 - (idx // 4 ** j) % 4)
    u, inv = np.unique(np.minimum(idx, r), return_inverse=True); return inv, len(u)
CAN = not (T.startswith('splice') or T == 'promoter_tata'); W = {k: rcmap(k) for k in (2, 3, 4, 5)} if CAN else None
class Pyr(nn.Module):
    def __init__(s, DL, NC, h=64, p=0.3):
        super().__init__(); s.DL = DL
        s.br = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)) for d in DL]); s.out = nn.Linear(h * len(DL), NC)
    def forward(s, x): return s.out(torch.cat([b(v) for b, v in zip(s.br, torch.split(x, s.DL, 1))], 1))
def pred(m, A):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(A), 2048): o.append(m(torch.tensor(A[i:i + 2048])).argmax(1).numpy())
    return np.concatenate(o)
def run(S, Y, St, Yt, LEV):
    L = len(S[0]); NC = int(max(Y.max(), Yt.max())) + 1; DL = PP.dims(L, LEV, W)
    X = np.stack([PP.feat(q, L, LEV, W) for q in S]); Xt = np.stack([PP.feat(q, L, LEV, W) for q in St]); X, Xt = PP.norm(X, Xt); X = X.astype(np.float32); Xt = Xt.astype(np.float32)
    torch.manual_seed(SEED); rs = np.random.RandomState(SEED); tr, va = train_test_split(np.arange(len(Y)), test_size=0.1, stratify=Y, random_state=SEED)
    m = Pyr(DL, NC); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); best = (-2, None); pat = 0
    for ep in range(40):
        m.train()
        for b in np.array_split(rs.permutation(tr), max(1, len(tr) // 128)):
            opt.zero_grad(); nn.functional.cross_entropy(m(torch.tensor(X[b])), torch.tensor(Y[b])).backward(); opt.step()
        v = matthews_corrcoef(Y[va], pred(m, X[va]))
        if v > best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()}); pat = 0
        else:
            pat += 1
            if pat >= 8: break
    m.load_state_dict(best[1]); pt = pred(m, Xt); return float(best[0]), float(matthews_corrcoef(Yt, pt)), float((pt == Yt).mean())
S, Y = PP.read(T, 'train'); St, Yt = PP.read(T, 'test'); S = [q.upper() for q in S]; St = [q.upper() for q in St]
if MODE == 'select':
    R = {'(%d,%s)' % (k, r if r else 'L'): run(S, Y, St, Yt, [(k, r)]) for k, r in BASE}; R['all'] = run(S, Y, St, Yt, BASE)
    b = max((k for k in R if k != 'all'), key=lambda k: R[k][0]); out = dict(levels=R, chosen=b, chosen_test=R[b][1], all_test=R['all'][1], best_test=max(R[k][1] for k in R if k != 'all'), chosen_acc=R[b][2], all_acc=R['all'][2])
else:
    def match(SS, YY, seed):
        g = np.array([int(100.0 * (q.count('G') + q.count('C')) / max(1, len(q))) for q in SS]); rs = np.random.RandomState(seed); keep = []
        for b in np.unique(g):
            p = np.where((g == b) & (YY == 1))[0]; n = np.where((g == b) & (YY == 0))[0]; k = min(len(p), len(n))
            if k: keep += list(rs.choice(p, k, replace=False)) + list(rs.choice(n, k, replace=False))
        keep = np.sort(np.array(keep)); return [SS[i] for i in keep], YY[keep]
    Sm, Ym = match(S, Y, 0); Stm, Ytm = match(St, Yt, 1)
    out = dict(n_train=len(Ym), n_test=len(Ytm), n_train_all=len(Y), n_test_all=len(Yt))
    for nm, lev in (('whole4', [(4, 1)]), ('bins4', [(4, 4)]), ('all', BASE)):
        out[nm + '_matched'] = run(Sm, Ym, Stm, Ytm, lev)[1]; out[nm + '_orig'] = run(S, Y, St, Yt, lev)[1]
out.update(task=T, mode=MODE, seed=SEED, sec=time.time() - t0); json.dump(out, open('results/ctl_%s_%s_s%d.json' % (MODE, T, SEED), 'w'))
print('CTL', MODE, T, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != 'levels'}, flush=True)
