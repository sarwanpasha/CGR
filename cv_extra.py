# Checks for the ClinVar study. usage: cv_extra.py leak | mes | mes strata
# leak : how many scored variants lie inside a training window of the benchmark splice-site tasks, and the AUC of the
#        pyramid score on the variants outside all training windows
# mes  : MaxEntScan baseline (needs the maxentpy package). The 9-mer at donor sites and the 23-mer at acceptor sites
#        are scored for the reference and the alternative window; the score is the drop caused by the variant.
import sys, json, collections, numpy as np
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
D = [l.rstrip('\n').split('\t') for l in open('results/cv_variants.tsv')]; h = D[0]; D = D[1:]; ix = {k: i for i, k in enumerate(h)}
y = np.array([int(d[ix['label']]) for d in D]); rel = np.array([int(d[ix['rel']]) for d in D]); ty = np.array([d[ix['type']] for d in D]); s = np.array([float(d[ix['d_pyr']]) for d in D])
o = json.load(open('results/cv_spliceai_scores.json')); K = np.array(sorted(int(k) for k in o)); sai = np.array([o[str(k)] for k in K]); w = np.where(y[K] == 1, 1.0, 687181.0 / float((y[K] == 0).sum()))
z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
if sys.argv[1] == 'leak':
    C = collections.defaultdict(list)
    for t in ('splice_sites_donors', 'splice_sites_acceptors', 'splice_sites_all'):
        for l in open('data/nt/%s_train.fna' % t):
            if l[0] == '>' and ':' in l:
                a = l[1:].strip().rsplit('|', 1)[0]; c, r = a.split(':'); b, e = r.split('-'); C[c].append((int(b), int(e)))
    I = {}
    for c, v in C.items():
        v = np.array(sorted(v)); I[c] = (v[:, 0], np.maximum.accumulate(v[:, 1]))
    def inside(c, p):
        if c not in I: return False
        b, e = I[c]; j = np.searchsorted(b, p, 'right') - 1; return bool(j >= 0 and e[j] >= p)
    ins = np.array([inside(D[k][ix['chrom']], int(D[k][ix['pos']])) for k in K]); out = ~ins
    print('LEAK subset n %d inside training windows %d (%.3f) pathogenic inside %d' % (len(K), ins.sum(), ins.mean(), y[K][ins].sum()))
    for nm, m in (('all', np.ones(len(K), bool)), ('outside', out), ('inside', ins)):
        if 0 < y[K][m].sum() < m.sum(): print('LEAK %-8s n %d path %d auc pyr %.3f sai %.3f p+sai %.3f | ap_full pyr %.3f sai %.3f p+sai %.3f' % (nm, m.sum(), y[K][m].sum(), AU(y[K][m], s[K][m]), AU(y[K][m], sai[m]), AU(y[K][m], z(s[K])[m] + z(sai)[m]), AP(y[K][m], s[K][m], sample_weight=w[m]), AP(y[K][m], sai[m], sample_weight=w[m]), AP(y[K][m], z(s[K])[m] + z(sai)[m], sample_weight=w[m])))
    print('LEAK test chromosomes of the benchmark are chr20, chr21; variants on them: %d' % sum(D[k][ix['chrom']] in ('chr20', 'chr21') for k in K))
else:
    from maxentpy import maxent
    try: m5, m3 = maxent.load_matrix5(), maxent.load_matrix3(); f5 = lambda q: maxent.score5(q, matrix=m5); f3 = lambda q: maxent.score3(q, matrix=m3)
    except Exception: f5, f3 = maxent.score5, maxent.score3
    Z = np.load('data/cv/windows.npz'); Wr, Wa = Z['w'], Z['wa']; d = np.zeros(len(K)); okc = 0
    for n, k in enumerate(K):
        a, b = str(Wr[k]), str(Wa[k]); i = [j for j in range(len(a)) if a[j] != b[j]][0]; an = i - rel[k]
        lo, hi, f = (an - 3, an + 6, f5) if ty[k] == 'D' else (an - 20, an + 3, f3)
        okc += a[an:an + 2] == 'GT' if ty[k] == 'D' else a[an - 2:an] == 'AG'
        if lo <= i < hi and 'N' not in a[lo:hi]: d[n] = f(a[lo:hi]) - f(b[lo:hi])
    if len(sys.argv) > 2:
        r = rel[K]; D_ = ty[K] == 'D'; yy = y[K]; rs = np.random.RandomState(0)
        reg = {'donor exon (<=-4)': D_ & (r <= -4), 'donor motif (-3..5)': D_ & (r >= -3) & (r <= 5), 'donor intron (>=6)': D_ & (r >= 6),
               'acc intron (<=-21)': ~D_ & (r <= -21), 'acc motif (-20..2)': ~D_ & (r >= -20) & (r <= 2), 'acc exon (>=3)': ~D_ & (r >= 3),
               'outside motifs': (D_ & ((r <= -4) | (r >= 6))) | (~D_ & ((r <= -21) | (r >= 3))), 'all': np.ones(len(K), bool)}
        for nm, m in reg.items():
            if not (0 < yy[m].sum() < m.sum()): continue
            a = z(sai[m]) + z(d[m]); b = a + z(s[K][m]); c0 = z(sai[m]); c1 = c0 + z(s[K][m]); ym = yy[m]; wm = w[m]; o = []
            for _ in range(400):
                i = rs.randint(0, len(ym), len(ym))
                if 0 < ym[i].sum() < len(i): o.append((AP(ym[i], b[i], sample_weight=wm[i]) - AP(ym[i], a[i], sample_weight=wm[i]), AU(ym[i], b[i]) - AU(ym[i], a[i]), AP(ym[i], c1[i], sample_weight=wm[i]) - AP(ym[i], c0[i], sample_weight=wm[i])))
            o = np.array(o); q = lambda j: '[%+.3f,%+.3f]' % (np.percentile(o[:, j], 2.5), np.percentile(o[:, j], 97.5))
            print('STR %-20s n %5d path %4d | ap sai %.3f s+m %.3f s+m+p %.3f pyr %.3f mes %.3f | d_ap(+p|s+m) %s d_auc %s d_ap(+p|s) %s' % (nm, m.sum(), ym.sum(), AP(ym, sai[m], sample_weight=wm), AP(ym, a, sample_weight=wm), AP(ym, b, sample_weight=wm), AP(ym, s[K][m], sample_weight=wm), AP(ym, d[m], sample_weight=wm), q(0), q(1), q(2)), flush=True)
        sys.exit()
    dist = -np.abs(rel[K]).astype(float); print('MES n %d canonical site found %.3f nonzero %.3f' % (len(K), okc / float(len(K)), (d != 0).mean()))
    for nm, v in (('mes', d), ('mes+dist', z(d) + z(dist)), ('pyr', s[K]), ('pyr+mes', z(s[K]) + z(d)), ('sai', sai), ('sai+mes', z(sai) + z(d)), ('p+sai', z(s[K]) + z(sai)), ('p+sai+mes', z(s[K]) + z(sai) + z(d))):
        print('MES %-10s auc %.3f ap_full %.3f' % (nm, AU(y[K], v), AP(y[K], v, sample_weight=w)))
