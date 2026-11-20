# Test of the resolution law on real data. usage: positional_spread.py <task> [top=30]
# For every k-mer (k=3,4,5) and every class (one-vs-rest) the script estimates, from the training set,
#   mu, sigma : centre and positional spread of the class-specific excess of the k-mer (16/50/84% points of the cumulative excess)
#   q         : excess occurrences per sequence;  lam : background rate per position
# then compares the bin width predicted by the law with the bin width that maximises the measured
# signal-to-noise ratio on a dyadic grid (R = 1, 2, 4, ... bins). Writes results/spread_<task>.json.
import sys, json, numpy as np
from scipy.stats import norm, spearmanr
try: import cgr_features as PP
except ImportError: import P as PP

def profile(S, k):                       # occurrence counts per (k-mer, start position), summed over sequences
    L = len(S[0]); n = L - k + 1; O = np.zeros((4 ** k, n))
    for q in S:
        a = PP.LUT[np.frombuffer(q.encode(), np.uint8)]; km, bad, _ = PP.kmers(a, k)
        pos = np.arange(n)[~bad]; np.add.at(O, (km[~bad], pos), 1)
    return O

def analyse(S, Y, k, top):
    out = []; n = len(S[0]) - k + 1; Rs = [2 ** j for j in range(int(np.log2(n)) + 1)]
    S = np.array(S, dtype=object)
    for c in np.unique(Y):
        P1 = profile(S[Y == c], k) / (Y == c).sum(); P0 = profile(S[Y != c], k) / (Y != c).sum()
        best = np.zeros((len(Rs), 4 ** k)); sign = np.zeros((len(Rs), 4 ** k))
        for i, R in enumerate(Rs):       # measured SNR of the best bin at each resolution (Poisson variance)
            e = np.linspace(0, n, R + 1).astype(int); c1 = np.add.reduceat(P1, e[:-1], 1); c0 = np.add.reduceat(P0, e[:-1], 1)
            d = (c1 - c0) ** 2 / (0.5 * (c1 + c0) + 1e-9); j = d.argmax(1); r = np.arange(4 ** k)
            best[i] = d[r, j]; sign[i] = np.sign(c1[r, j] - c0[r, j])
        for m in np.argsort(-best.max(0))[:top]:
            i = int(best[:, m].argmax()); s = sign[i, m]
            if s <= 0: continue          # keep k-mers enriched in the class
            ex = np.cumsum(P1[m] - P0[m]); A = ex[-1]
            if A <= 0: continue
            x16, x50, x84 = [int(np.searchsorted(np.maximum.accumulate(ex), f * A)) for f in (0.16, 0.5, 0.84)]
            sig = max((x84 - x16) / 2.0, 0.5); lam = max(P0[m].mean(), 1e-6); q = min(A, 1.0)
            w = n / np.array(Rs); p = q * (2 * norm.cdf(w / (2 * sig)) - 1); pred = p ** 2 / (lam * w + p * (1 - p) / 2)
            out.append(dict(k=k, cls=int(c), kmer=''.join('ACGT'[(m >> (2 * (k - 1 - j))) & 3] for j in range(k)), mu=x50, sigma=sig, q=float(A), lam=float(lam),
                            w_meas=float(w[i]), w_pred=float(w[int(pred.argmax())]), snr=float(best[i, m])))
    return out

if __name__ == '__main__':
    T = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    S, Y = PP.read(T, 'train'); res = []
    for k in (3, 4, 5): res += analyse(S, Y, k, top)
    a = np.log2([r['w_meas'] for r in res]); b = np.log2([r['w_pred'] for r in res])
    summ = dict(task=T, n=len(res), spearman=float(spearmanr(a, b)[0]), within_factor2=float(np.mean(np.abs(a - b) <= 1)), exact=float(np.mean(a == b)), median_sigma=float(np.median([r['sigma'] for r in res])))
    json.dump(dict(summary=summ, kmers=res), open('results/spread_%s.json' % T, 'w'))
    print('SPREAD', summ, flush=True)
