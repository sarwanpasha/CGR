# SpliceAI, Pangolin and MMSplice on the Vex-seq variants scored by vexseq_splice.py. usage: vexseq_specialized.py sai|pan|mms|report
# Each mode reads results/vex_scores.tsv and data/vex/hg19_vex.fa and writes results/vex_<mode>.json (variant id -> score);
# the 3 models need different Python environments, so each is run separately. Scores are defined as for the MFASS
# variants (mfass_spliceai.py, mfass_pangolin.py, mfass_mmsplice.py). report: Spearman correlation of every score with
# the measured loss of exon inclusion, AUC and average precision for a loss of at least 10 points of percent spliced
# in, and paired bootstrap intervals for the gain from adding the pyramid score.
import sys, os, json, glob, time, numpy as np
MODE = sys.argv[1]; t0 = time.time()
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
V = [l.rstrip('\n').split('\t') for l in open('results/vex_scores.tsv')]; vh = V[0]; V = [dict(zip(vh, v)) for v in V[1:]]
COMP = str.maketrans('ACGTN', 'TGCAN'); LUT = np.full(256, 4, np.int64)
for i, c in enumerate('ACGT'): LUT[ord(c)] = i
EYE = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)]); F = 5050
def genome():
    G = {}; L = open('data/vex/hg19_vex.fa').read().split('\n')
    for i in range(0, len(L) - 1, 2): G[L[i][1:]] = L[i + 1]
    return G
def windows():
    G = genome(); R = []
    for v in V:
        g = G[v['chrom']]; p = int(v['pos']); s = g[p - 1 - F:p + F]
        if len(s) != 2 * F + 1 or s[F] != v['ref']: continue
        a = s[:F] + v['alt'] + s[F + 1:]
        if v['strand'] == '-': s = s.translate(COMP)[::-1]; a = a.translate(COMP)[::-1]
        R.append((v['id'], s, a))
    return R
if MODE == 'sai':
    import importlib.util
    from tensorflow.keras.models import load_model
    mods = [load_model(f, compile=False) for f in sorted(glob.glob(os.path.join(os.path.dirname(importlib.util.find_spec('spliceai').origin), 'models', 'spliceai*.h5')))]
    oh = lambda q: EYE[LUT[np.frombuffer(q.encode(), np.uint8)]]; sc = lambda X: np.mean([m.predict(X, batch_size=16, verbose=0) for m in mods], 0); R = windows(); out = {}
    for b in range(0, len(R), 64):
        J = R[b:b + 64]; d = np.abs(sc(np.stack([oh(a) for _, _, a in J]))[:, :, 1:] - sc(np.stack([oh(s) for _, s, _ in J]))[:, :, 1:]).max((1, 2))
        for (i, _, _), x in zip(J, d): out[i] = float(x)
        log('done', b + len(J), 'of', len(R))
    json.dump(out, open('results/vex_sai.json', 'w'))
elif MODE == 'pan':
    import torch
    sys.path.insert(0, 'pangolin_src'); from pangolin.model import Pangolin, L as PL, W as PW, AR as PAR
    torch.set_num_threads(12); mods = []
    for i in (0, 2, 4, 6):
        grp = []
        for j in (1, 2, 3):
            m = Pangolin(PL, PW, PAR); m.load_state_dict(torch.load('pangolin_src/pangolin/models/final.%d.%d.3.v2' % (j, i), map_location='cpu')); m.eval(); grp.append(m)
        mods.append(grp)
    oh = lambda q: EYE[LUT[np.frombuffer(q.encode(), np.uint8)]].T
    def sc(X):
        X = torch.from_numpy(X)
        with torch.no_grad(): return np.stack([torch.stack([m(X)[:, ch, :] for m in grp]).mean(0).numpy() for grp, ch in zip(mods, (1, 4, 7, 10))], 1)
    R = windows(); out = {}
    for b in range(0, len(R), 16):
        J = R[b:b + 16]; d = np.abs(sc(np.stack([oh(a) for _, _, a in J])) - sc(np.stack([oh(s) for _, s, _ in J]))).max((1, 2))
        for (i, _, _), x in zip(J, d): out[i] = float(x)
        if (b // 16) % 10 == 0: log('done', b + len(J), 'of', len(R))
    json.dump(out, open('results/vex_pan.json', 'w'))
elif MODE == 'mms':
    import pandas as pd
    from mmsplice import MMSplice, predict_all_table
    from mmsplice.exon_dataloader import ExonDataset
    pd.DataFrame([dict(Chromosome=v['chrom'], Exon_Start=int(v['s1']), Exon_End=int(v['e1']), Strand=v['strand'], pos=int(v['pos']), ref=v['ref'], alt=v['alt'], exon_id=v['id']) for v in V]).to_csv('results/vex_mms_exons.csv', index=False)
    P = predict_all_table(MMSplice(), ExonDataset('results/vex_mms_exons.csv', 'data/vex/hg19_vex.fa'), pathogenicity=False, splicing_efficiency=False); log('predictions', P.shape, list(P.columns))
    json.dump({str(k): float(-x) for k, x in zip(P['exon_id'], P['delta_logit_psi']) if np.isfinite(x)}, open('results/vex_mms.json', 'w'))
else:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
    E = {k: json.load(open('results/vex_%s.json' % k)) for k in ('sai', 'pan', 'mms') if os.path.exists('results/vex_%s.json' % k)}
    V = [v for v in V if all(v['id'] in e for e in E.values())]; y = -np.array([float(v['dpsi']) for v in V]); lab = (y >= 10).astype(int); z = lambda q: (q - q.mean()) / (q.std() + 1e-9)
    S = {'pyr': np.array([float(v['pyr']) for v in V]), 'mes': np.array([float(v['mes']) for v in V])}
    for k, e in E.items(): S[k] = np.array([e[v['id']] for v in V])
    PAIRS = []
    for k in E: S['pyr+' + k] = z(S['pyr']) + z(S[k]); PAIRS.append(('pyr+' + k, k))
    if len(E) == 3: S['spec'] = z(S['sai']) + z(S['pan']) + z(S['mms']); S['pyr+spec'] = z(S['pyr']) + S['spec']; PAIRS.append(('pyr+spec', 'spec'))
    mot = np.array([v['motif'] == 'True' for v in V]); res = {}; rs = np.random.RandomState(0)
    for nm, m in (('all', np.ones(len(V), bool)), ('motif', mot), ('outside_motif', ~mot)):
        rr = dict(n=int(m.sum()), n_loss=int(lab[m].sum()))
        for k, q in S.items(): rr[k] = [float(spearmanr(q[m], y[m])[0]), float(AU(lab[m], q[m])), float(AP(lab[m], q[m]))]
        o = []
        for _ in range(500):
            i = rs.randint(0, m.sum(), m.sum()); yy = y[m][i]
            o.append([spearmanr(S[a][m][i], yy)[0] - spearmanr(S[b][m][i], yy)[0] for a, b in PAIRS])
        o = np.array(o); rr['ci_spearman'] = {'%s-(%s)' % p: [float(np.percentile(o[:, j], 2.5)), float(np.percentile(o[:, j], 97.5))] for j, p in enumerate(PAIRS)} if PAIRS else {}
        res[nm] = rr; log('VR', nm, 'n', rr['n'], 'loss', rr['n_loss'], ' '.join('%s %.3f/%.3f/%.3f' % (k, *rr[k]) for k in S)); log('VRCI', nm, {k: [round(x, 3) for x in v] for k, v in rr['ci_spearman'].items()})
    json.dump(res, open('results/vex_report.json', 'w'), indent=1)
log('VS_DONE', MODE)
