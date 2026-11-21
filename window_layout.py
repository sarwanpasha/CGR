# How are the windows of a task laid out on the genome? usage: window_layout.py <task>
# Reads the coordinates in the FASTA headers (">chr:start-end|label") of the training file and reports, for each class,
# how often a window of the other class lies within 2 kb on its left and on its right, and how often two windows abut.
import sys, collections, numpy as np
T = sys.argv[1]; C = collections.defaultdict(list)
for l in open('data/nt/%s_train.fna' % T):
    if l[0] == '>':
        a, y = l[1:].strip().rsplit('|', 1); c, r = a.split(':'); s, e = r.split('-'); C[c].append((int(s), int(e), int(y)))
n = collections.Counter(); L = collections.Counter(); R = collections.Counter(); AL = collections.Counter(); AR = collections.Counter(); SL = collections.Counter(); SR = collections.Counter()
for c, v in C.items():
    v = np.array(sorted(v)); s, e, y = v[:, 0], v[:, 1], v[:, 2]
    for k in np.unique(y):
        o = np.sort(s[y != k]); oe = np.sort(e[y != k]); m = np.sort(s[y == k]); me = np.sort(e[y == k])
        for a, b in zip(s[y == k], e[y == k]):
            n[k] += 1
            j = np.searchsorted(oe, a, 'right') - 1; L[k] += j >= 0 and a - oe[j] <= 2000; AL[k] += j >= 0 and abs(a - oe[j]) <= 1
            j = np.searchsorted(o, b, 'left'); R[k] += j < len(o) and o[j] - b <= 2000; AR[k] += j < len(o) and abs(o[j] - b) <= 1
            j = np.searchsorted(me, a, 'right') - 1; SL[k] += j >= 0 and me[j] != b and abs(a - me[j]) <= 1
            j = np.searchsorted(m, b, 'left'); SR[k] += j < len(m) and abs(m[j] - b) <= 1
for k in sorted(n):
    print('LAYOUT %-24s class %d n=%6d other<=2kb left %.3f right %.3f | abuts other left %.3f right %.3f | abuts same left %.3f right %.3f' % (T, k, n[k], L[k] / n[k], R[k] / n[k], AL[k] / n[k], AR[k] / n[k], SL[k] / n[k], SR[k] / n[k]), flush=True)
