# Position-aware k-mer comparators on one task (CPU). usage: comparators.py <task> pfcgr|oligo|wd
# pfcgr : 4-mer counts plus mean, s.d., skewness and kurtosis of the positions of every 4-mer; one-hidden-layer network
# oligo : trinucleotide occurrences smoothed along the sequence by a Gaussian of width sigma (oligo kernel feature map),
#         sampled every sigma nucleotides; linear SVM; sigma chosen on the validation split
# wd    : position-specific k-mers of length 1 to 3 with weighted-degree weights; linear SVM
import sys, json, time, numpy as np, scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef as M
from sklearn.model_selection import train_test_split
import cgr_features as P
T, MODE = sys.argv[1], sys.argv[2]; t0 = time.time()
S, Y = P.read(T, 'train'); St, Yt = P.read(T, 'test'); L = len(S[0])
tr, va = train_test_split(np.arange(len(Y)), test_size=0.1, stratify=Y, random_state=0)
def km(q, k):
    a = P.LUT[np.frombuffer(q.encode(), np.uint8)]; c, bad, n = P.kmers(a, k); return c, bad, n
def pf(q):
    c, bad, n = km(q, 4); pos = np.arange(n)[~bad] / float(L); c = c[~bad]; o = np.zeros((5, 256), np.float32)
    cnt = np.bincount(c, minlength=256).astype(np.float64); nz = cnt > 0
    m1 = np.bincount(c, pos, 256); mu = np.where(nz, m1 / np.maximum(cnt, 1), 0); d = pos - mu[c]
    v = np.bincount(c, d ** 2, 256) / np.maximum(cnt, 1); sd = np.sqrt(v); s3 = np.bincount(c, d ** 3, 256) / np.maximum(cnt, 1); s4 = np.bincount(c, d ** 4, 256) / np.maximum(cnt, 1)
    ok = sd > 1e-9; o[0] = np.log1p(cnt); o[1] = mu; o[2] = sd; o[3] = np.where(ok, s3 / np.maximum(sd, 1e-9) ** 3, 0); o[4] = np.where(ok, s4 / np.maximum(v, 1e-18) ** 2 - 3, 0)
    return o.ravel()
def og(q, sig):
    c, bad, n = km(q, 3); pos = np.arange(n)[~bad]; c = c[~bad]; cen = np.arange(sig / 2.0, L, sig); o = np.zeros((len(cen), 64), np.float32)
    w = np.exp(-(pos[None, :] - cen[:, None]) ** 2 / (4.0 * sig * sig))
    for j in range(len(cen)): o[j] = np.bincount(c, w[j], 64)
    return o.ravel()
def wd(SS):
    r, cidx, val = [], [], []; off = 0; D = 3
    for k in range(1, D + 1):
        b = np.sqrt(2.0 * (D - k + 1) / (D * (D + 1)))
        for i, q in enumerate(SS):
            c, bad, n = km(q, k); pos = np.arange(n)[~bad]; c = c[~bad]
            r.append(np.full(len(c), i, np.int32)); cidx.append((off + pos * 4 ** k + c).astype(np.int32)); val.append(np.full(len(c), b, np.float32))
        off += L * 4 ** k
    return sp.csr_matrix((np.concatenate(val), (np.concatenate(r), np.concatenate(cidx))), shape=(len(SS), off))
def svm(X, Xt, Cs):
    best = (-2, None)
    for C in Cs:
        m = LinearSVC(C=C, max_iter=2000).fit(X[tr], Y[tr]); v = M(Y[va], m.predict(X[va]))
        if v > best[0]: best = (v, float(M(Yt, m.predict(Xt))), C)
    return best
if MODE == 'pfcgr':
    X = np.stack([pf(q) for q in S]); Xt = np.stack([pf(q) for q in St]); X, Xt = P.norm(X, Xt)
    m = MLPClassifier((256,), early_stopping=True, validation_fraction=0.1, max_iter=80, random_state=0).fit(X, Y); r = dict(test=float(M(Yt, m.predict(Xt))))
elif MODE == 'oligo':
    r = dict(val=-2)
    for sig in (L / 64.0, L / 16.0, L / 4.0):
        X = np.stack([og(q, sig) for q in S]); Xt = np.stack([og(q, sig) for q in St]); X, Xt = P.norm(X, Xt); b = svm(X, Xt, (0.001, 0.01))
        if b[0] > r['val']: r = dict(val=float(b[0]), test=b[1], sigma=sig, C=b[2])
else:
    X = wd(S); Xt = wd(St); b = svm(X, Xt, (0.001, 0.01, 0.1)); r = dict(val=float(b[0]), test=b[1], C=b[2])
r.update(task=T, mode=MODE, sec=time.time() - t0); json.dump(r, open('results/cmp_%s_%s.json' % (MODE, T), 'w'))
print('CMP', MODE, T, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
