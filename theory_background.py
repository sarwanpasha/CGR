# Robustness of the optimal bin width to the background model. usage: theory_background.py [n_seq]
# One k-mer occurs once per positive sequence at a position drawn from N(mu, sigma^2). Background occurrences follow
#   poisson  : independent Poisson counts per position (the model of the analysis)
#   negbin   : independent negative-binomial counts per position (overdispersed, variance = 3 x mean)
#   seqrate  : Poisson with a sequence-specific rate (gamma, CV 0.7), as when composition differs between sequences
#   patches  : Poisson whose rate is 6 times higher inside patches of mean length 150 nt (two-state Markov chain)
# For each model and sigma the script measures d^2(w) for a bin of width w centred on mu and reports the best width
# and the fraction of the best d^2 that the width 2.80 sigma of the analysis retains.
import sys, json, numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000; L = 1200; MU = L // 2; LAM = 0.02; rs = np.random.RandomState(0); res = {}
def background(kind, n):
    if kind == 'poisson': return rs.poisson(LAM, (n, L))
    if kind == 'negbin': r = LAM / 2.0; return rs.negative_binomial(r, r / (r + LAM), (n, L))
    if kind == 'seqrate': g = rs.gamma(1 / 0.49, 0.49, (n, 1)); return rs.poisson(LAM * g * np.ones((1, L)))
    st = np.zeros((n, L), bool); s = rs.rand(n) < 0.2
    for x in range(L):
        st[:, x] = s; u = rs.rand(n); s = np.where(s, u > 1 / 150.0, u < 1 / 600.0)
    return rs.poisson(np.where(st, 3.0 * LAM, 0.5 * LAM))
for kind in ('poisson', 'negbin', 'seqrate', 'patches'):
    B0 = np.cumsum(np.pad(background(kind, N), ((0, 0), (1, 0))), 1); B1 = np.cumsum(np.pad(background(kind, N), ((0, 0), (1, 0))), 1)
    for sg in (5, 10, 20, 40, 80):
        p = np.clip(np.round(rs.normal(MU, sg, N)).astype(int), 0, L - 1); ws = np.unique(np.round(sg * np.exp(np.linspace(np.log(0.6), np.log(8), 60))).astype(int)); ws = ws[(ws >= 2) & (ws < L - 2)]; d2 = []
        for w in ws:
            a = MU - w // 2; b = a + w; c0 = B0[:, b] - B0[:, a]; c1 = B1[:, b] - B1[:, a] + ((p >= a) & (p < b))
            d2.append((c1.mean() - c0.mean()) ** 2 / (0.5 * (c0.var() + c1.var()) + 1e-12))
        d2 = np.array(d2); j = int(np.argmax(d2)); k = int(np.argmin(np.abs(ws - 2.80 * sg)))
        sm = np.convolve(d2, np.ones(5) / 5, 'same'); js = int(np.argmax(sm[2:-2])) + 2
        res['%s_%d' % (kind, sg)] = dict(best_w=int(ws[js]), ratio=float(ws[js] / sg), retained=float(d2[k] / d2.max()), retained_smooth=float(sm[k] / sm[2:-2].max()))
        print('BG %-8s sigma %3d  best w %4d (%.2f sigma)  d2 at 2.80 sigma / best %.3f (smoothed %.3f)' % (kind, sg, ws[js], ws[js] / sg, d2[k] / d2.max(), sm[k] / sm[2:-2].max()), flush=True)
json.dump(res, open('results/theory_background.json', 'w'), indent=1)
