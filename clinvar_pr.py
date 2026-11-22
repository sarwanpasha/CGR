# Precision-recall on the ClinVar stratum. usage: clinvar_pr.py [n_benign_total]
# SpliceAI was run on all pathogenic variants of the stratum and a random sample of the benign ones. Weighting each
# sampled benign variant by (benign in stratum)/(benign sampled) gives estimates for the full stratum.
import sys, json, numpy as np
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AU
D = [l.rstrip('\n').split('\t') for l in open('results/cv_variants.tsv')]; h = D[0]; D = D[1:]; ix = {k: i for i, k in enumerate(h)}
o = json.load(open('results/cv_spliceai_scores.json')); K = np.array(sorted(int(k) for k in o))
y = np.array([int(D[k][ix['label']]) for k in K]); rel = np.array([int(D[k][ix['rel']]) for k in K])
S = {'pyr': np.array([float(D[k][ix['d_pyr']]) for k in K]), 'dist': -np.abs(rel).astype(float), 'sai': np.array([o[str(k)] for k in K])}
z = lambda v: (v - v.mean()) / (v.std() + 1e-9); S['p+sai'] = z(S['pyr']) + z(S['sai'])
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 687181; w = np.where(y == 1, 1.0, NB / float((y == 0).sum()))
print('n %d path %d benign %d weight %.1f prevalence %.4f' % (len(y), y.sum(), (y == 0).sum(), w.max(), y.sum() / (y.sum() + NB)))
for k, v in S.items():
    t = np.sort(v[y == 0])[::-1]; r = [float((v[y == 1] > t[int(f * len(t))]).mean()) for f in (0.001, 0.01)]
    print('%-5s auc %.3f ap_sample %.3f ap_full %.4f recall@fpr0.1%% %.3f @1%% %.3f' % (k, AU(y, v), AP(y, v), AP(y, v, sample_weight=w), r[0], r[1]))
