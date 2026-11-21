# Composition along the window, per class. usage: window_profile.py <task>
# Prints GC fraction and CpG density (both x1000) in 10 equal bins from the left to the right end of the window.
import sys, numpy as np
T = sys.argv[1]; S = []; Y = []
for l in open('data/nt/%s_train.fna' % T):
    l = l.strip()
    if l[:1] == '>': Y.append(int(l.rsplit('|', 1)[1])); S.append('')
    elif l: S[-1] += l.upper()
Y = np.array(Y); L = len(S[0]); B = 10; w = L // B
def f(q): return [(q[i * w:(i + 1) * w].count('G') + q[i * w:(i + 1) * w].count('C')) / w for i in range(B)] + [q[i * w:(i + 1) * w].count('CG') / w for i in range(B)]
M = np.array([f(q) for q in S])
for k in np.unique(Y):
    print('PROF', T[:14].ljust(14), k, ' '.join('%3d' % round(1000 * x) for x in M[Y == k].mean(0)[:B]), '|', ' '.join('%3d' % round(1000 * x) for x in M[Y == k].mean(0)[B:]))
