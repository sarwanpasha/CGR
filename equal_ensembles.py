# Ensembles of equal size from stored test probabilities: 3 base pyramid networks, 3 CNNs, both, and the seed ensemble
# of fine-tuned DNABERT-2 (as many seeds as are stored). usage: equal_ensembles.py
import glob, os, numpy as np, cgr_features as P
from sklearn.metrics import matthews_corrcoef as M
T = [l.strip() for l in open('tasks.txt') if l.strip()]; A = []
def pr(pat):
    f = sorted(glob.glob(pat)); return (np.mean([np.load(x) for x in f], 0), len(f)) if f else (None, 0)
for t in T:
    y = P.read(t, 'test')[1]; b, nb = pr('results/pyr2_base_%s_s[0-9].npy' % t); c, nc = pr('results/pyr2_cnn_%s_s[0-9].npy' % t); f, nf = pr('results/fmft_dnabert2_%s_s[0-9].npy' % t)
    r = [M(y, b.argmax(1)), M(y, c.argmax(1)), M(y, (b + c).argmax(1)), M(y, f.argmax(1)) if nf else float('nan')]
    print('EQ %-12s %.3f %.3f %.3f %.3f n%d%d%d' % (t.replace('splice_sites_', 'ss_').replace('promoter_', 'pr_').replace('enhancers', 'enh'), r[0], r[1], r[2], r[3], nb, nc, nf)); A.append(r + [nf])
A = np.array(A); k3 = A[:, 4] >= 3; k1 = A[:, 4] >= 1
print('EQMEAN all18 pyr3 %.4f cnn3 %.4f both %.4f | tasks with FT probs (%d): pyr3 %.4f cnn3 %.4f both %.4f ft %.4f | 3 FT seeds (%d): pyr3 %.4f cnn3 %.4f both %.4f ft %.4f' % (
    A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), k1.sum(), A[k1, 0].mean(), A[k1, 1].mean(), A[k1, 2].mean(), A[k1, 3].mean(), k3.sum(), A[k3, 0].mean() if k3.any() else 0, A[k3, 1].mean() if k3.any() else 0, A[k3, 2].mean() if k3.any() else 0, A[k3, 3].mean() if k3.any() else 0))
