# Print the numbers behind the tables from the stored result files. usage: final_report.py main|cnn|lev|cv|prof
import sys, glob, json, os, numpy as np
from sklearn.metrics import matthews_corrcoef as M, average_precision_score as AP, roc_auc_score as AU
W = sys.argv[1]
T = [l.strip() for l in open('tasks.txt') if l.strip()] if os.path.exists('tasks.txt') else sorted({f.split('pyr2_base_')[1].rsplit('_s', 1)[0] for f in glob.glob('results/pyr2_base_*_s0.npy')})
T = [t for t in T if os.path.exists('data/nt/%s_test.fna' % t)]
FX = [t for t in T if t.startswith('splice') or t == 'promoter_tata']
N = ['base', 'r2', 'k6', 'k6w_lr3']; F = N + ['fuse_dnabert2', 'k6fuse_dnabert2']
ab = lambda t: t.replace('splice_sites_', 'ss_').replace('promoter_', 'pr_').replace('enhancers', 'enh').ljust(11)
def pr(vs, t): return np.mean([np.load(f) for v in vs for f in sorted(glob.glob('results/pyr2_%s_%s_s[0-9].npy' % (v, t)))], 0)
def js(pat, k): return [json.load(open(f))[k] for f in sorted(glob.glob(pat))]
if W in ('main', 'cnn'):
    import cgr_features as P
    A = []
    for t in T:
        y = P.read(t, 'test')[1]; p = pr(N, t); q = pr(F, t); c = pr(['cnn'], t); nc = len(glob.glob('results/pyr2_cnn_%s_s[0-9].npy' % t))
        if W == 'main':
            f = js('results/fmft_dnabert2_%s_s[0-9].json' % t, 'mcc')
            r = [M(y, p.argmax(1)), M(y, ((p + c) / 2 if t in FX else p).argmax(1)), M(y, ((q + c) / 2 if t in FX else q).argmax(1)), np.mean(f), len(f)] + f
            print(ab(t), '%.3f %.3f %.3f | %.3f n%d |' % tuple(r[:5]), ' '.join('%.3f' % x for x in f)); A.append(r[:4] + [f[0]])
        else:
            b = js('results/pyr2_base_%s_s[0-9].json' % t, 'test_mcc'); s = js('results/pyr2_cnn_%s_s[0-9].json' % t, 'test_mcc')
            r = [np.mean(b), np.mean(s), M(y, c.argmax(1)), M(y, p.argmax(1)), M(y, (p + c).argmax(1)), nc]
            print(ab(t), 'base1 %.3f cnn1 %.3f | cnnE %.3f pyrE %.3f both %.3f n%d' % tuple(r)); A.append(r[:5])
    print('MEAN       ', ' '.join('%.4f' % x for x in np.mean(A, 0)))
if W == 'lev':
    for t in T:
        r = json.load(open('results/lev_%s_s0.json' % t))['mcc']; print(ab(t), ' '.join('%.3f' % r[k] for k in r if k.startswith('only') or k == 'all'), '|', ' '.join(k for k in r if k.startswith('only') or k == 'all') if t == T[0] else '')
if W == 'prof':
    for t in T:
        S = []; Y = []
        for l in open('data/nt/%s_train.fna' % t):
            l = l.strip()
            if l[:1] == '>': Y.append(int(l.rsplit('|', 1)[1])); S.append('')
            elif l: S[-1] += l.upper()
        Y = np.array(Y); w = len(S[0]) // 10
        g = np.array([[(q[i * w:(i + 1) * w].count('G') + q[i * w:(i + 1) * w].count('C')) / w for i in range(10)] for q in S])
        for k in np.unique(Y): print(ab(t), k, ' '.join('%3d' % round(1000 * x) for x in g[Y == k].mean(0)))
if W == 'cv':
    D = [l.rstrip('\n').split('\t') for l in open('results/cv_variants.tsv')]; h = D[0]; D = D[1:]; ix = {k: i for i, k in enumerate(h)}
    y = np.array([int(d[ix['label']]) for d in D]); rel = np.array([int(d[ix['rel']]) for d in D]); ty = np.array([d[ix['type']] for d in D])
    s = np.array([float(d[ix['d_pyr']]) for d in D]); c = np.array([float(d[ix['d_cnn']]) for d in D]); dist = -np.abs(rel).astype(float)
    can = ((ty == 'D') & np.isin(rel, [0, 1])) | ((ty == 'A') & np.isin(rel, [-2, -1])); m = ~can
    def top(yy, v, k): return float(yy[np.argsort(-v)[:k]].mean())
    print('FULL n %d path %d prev %.4f' % (m.sum(), y[m].sum(), y[m].mean()))
    for nm, v in (('pyr', s), ('cnn', c), ('dist', dist)): print(' %-5s auc %.3f ap %.4f top1000 %.3f top5000 %.3f' % (nm, AU(y[m], v[m]), AP(y[m], v[m]), top(y[m], v[m], 1000), top(y[m], v[m], 5000)))
    far = m & (np.abs(rel) > 10); print('FAR>10 n %d path %d prev %.4f' % (far.sum(), y[far].sum(), y[far].mean()))
    for nm, v in (('pyr', s), ('cnn', c), ('dist', dist)): print(' %-5s auc %.3f ap %.4f' % (nm, AU(y[far], v[far]), AP(y[far], v[far])))
    o = json.load(open('results/cv_spliceai_scores.json')); K = np.array(sorted(int(k) for k in o)); a = np.array([o[str(k)] for k in K]); z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    print('SUBSET n %d path %d prev %.4f' % (len(K), y[K].sum(), y[K].mean()))
    for nm, v in (('pyr', s[K]), ('dist', dist[K]), ('sai', a), ('p+sai', z(s[K]) + z(a))): print(' %-5s auc %.3f ap %.3f' % (nm, AU(y[K], v), AP(y[K], v)))
