# Minimal working example on synthetic data. Needs only numpy and scikit-learn, no downloads, about 1 minute on a CPU.
# usage: python demo_synthetic.py
# Sequences of 2 classes are simulated. In class 1 a short motif is planted near the middle of the
# sequence with a positional spread of 2 nt, 20 nt, or at a uniformly random position.
# For each case the script reports the test MCC of every pyramid level on its own and of all levels
# combined. Level (5,1) alone is the plain FCGR. The level whose bin width matches the spread of the
# motif carries the signal, which is the point of the position-pyramid FCGR.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from cgr_features import feat, dims, LEV

L, N, MOTIF = 200, 4000, 'GTAA'

def simulate(spread, seed):
    r = np.random.RandomState(seed); S = []; Y = np.arange(N) % 2
    for y in Y:
        a = list(r.choice(list('ACGT'), L))
        if y == 1:
            p = r.randint(0, L - len(MOTIF)) if spread is None else int(np.clip(round(L / 2 + r.randn() * spread), 0, L - len(MOTIF)))
            a[p:p + len(MOTIF)] = MOTIF
        S.append(''.join(a))
    return S, Y

def run(S, Y):
    X = np.stack([feat(s, L) for s in S]); n = len(Y) * 3 // 4               # log(1+count) features of all levels
    mu = X[:n].mean(0); sd = X[:n].std(0) + 1e-6; X = (X - mu) / sd          # standardize with training statistics
    cuts = np.cumsum([0] + dims(L)); tr, te, res = [], [], []
    for i in range(len(LEV)):                                                # one linear model per level
        A = X[:, cuts[i]:cuts[i + 1]]; m = LogisticRegression(C=0.003, max_iter=300)
        tr.append(cross_val_predict(m, A[:n], Y[:n], cv=3, method='decision_function'))
        m.fit(A[:n], Y[:n]); te.append(m.decision_function(A[n:])); res.append(matthews_corrcoef(Y[n:], m.predict(A[n:])))
    top = LogisticRegression().fit(np.stack(tr, 1), Y[:n])                   # combine the levels
    return res, matthews_corrcoef(Y[n:], top.predict(np.stack(te, 1)))

if __name__ == '__main__':
    print('levels (k, bins); 0 bins means one bin per position:', LEV)
    print('features per level for L=%d:' % L, dims(L))
    print('%-22s' % 'motif position' + ''.join('%9s' % ('(%d,%s)' % (k, r if r else 'L')) for k, r in LEV) + '%10s' % 'combined')
    for name, spread in (('spread 2 nt', 2), ('spread 20 nt', 20), ('random', None)):
        S, Y = simulate(spread, seed=1); res, comb = run(S, Y)
        print('%-22s' % name + ''.join('%9.3f' % v for v in res) + '%10.3f' % comb)
