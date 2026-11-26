# MMSplice on the MFASS variants scored by mfass_splice.py. usage: mfass_mmsplice.py
# Needs the mmsplice package (pip install mmsplice). Each variant is given to the exon-level interface of MMSplice
# with the boundaries of its assayed exon; the score is minus the predicted change of logit(PSI), so that larger values
# mean a larger predicted loss of exon inclusion. Compared with SpliceAI, Pangolin (when their scores exist), MaxEntScan
# and the pyramid score overall and by region, with paired bootstrap intervals of differences in average precision.
import os, gzip, csv, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
t0 = time.time()
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
FA = 'data/cv/GRCh38.plain.fa'
if not os.path.exists(FA + '.done'):
    with gzip.open('data/cv/GRCh38.primary_assembly.genome.fa.gz', 'rt') as f, open(FA, 'w') as o:
        keep = False
        for l in f:
            if l[0] == '>': nm = l[1:].split()[0]; keep = nm.startswith('chr') and len(nm) < 6; l = '>' + nm + '\n'
            if keep: o.write(l)
    open(FA + '.done', 'w').write('ok'); log('fasta written')
R_ = list(csv.reader(open('data/mfass/snv_data_clean.txt'), delimiter='\t')); H = [h.strip('"') for h in R_[0]]; ix = {h: i for i, h in enumerate(H)}
info = {}
for r in R_[1:]:
    r = [x.strip('"') for x in r]; info[r[0]] = r
cS = [i for i, h in enumerate(H) if 'start_hg38' in h and 'snp' not in h][0]
V = [l.rstrip('\n').split('\t') for l in open('results/mfass_scores.tsv')]; vh = V[0]; V = V[1:]; vx = {h: i for i, h in enumerate(vh)}
from pyfaidx import Fasta
G = Fasta(FA, sequence_always_upper=True); COMP = str.maketrans('ACGT', 'TGCA'); rows = []
def num(x):
    try: return int(float(x))
    except Exception: return None
for v in V:
    r = info.get(v[vx['id']]); c = v[vx['chrom']]; p = int(v[vx['pos']])
    if r is None or c not in G.keys(): continue
    ref = r[ix['ref_allele']].upper(); alt = r[ix['alt_allele']].upper(); g = str(G[c][p - 1:p])
    if g != ref:
        if g == ref.translate(COMP): ref = g; alt = alt.translate(COMP)
        else: continue
    s1 = num(r[cS]) + num(r[ix['intron1_len']]) + 1; e1 = s1 + num(r[ix['exon_len']]) - 1
    rows.append(dict(Chromosome=c, Exon_Start=s1, Exon_End=e1, Strand=v[vx['strand']], pos=p, ref=ref, alt=alt, exon_id=v[vx['id']]))
D = pd.DataFrame(rows); D.to_csv('results/mfass_mmsplice_exons.csv', index=False); log('exon table', len(D))
from mmsplice import MMSplice, predict_all_table
from mmsplice.exon_dataloader import ExonDataset
dl = ExonDataset('results/mfass_mmsplice_exons.csv', FA); P = predict_all_table(MMSplice(), dl, pathogenicity=False, splicing_efficiency=False)
log('predictions', P.shape, list(P.columns)); print(P.head(3).to_string(), flush=True); P.to_csv('results/mfass_mmsplice_pred.csv', index=False)
key = 'exon_id' if 'exon_id' in P.columns else None
if key is None: raise SystemExit('no exon_id column in the predictions; see results/mfass_mmsplice_pred.csv')
mm = dict(zip(P[key].astype(str), -P['delta_logit_psi'].astype(float)))
ext = {}
for nm, f in (('sai', 'results/mfass_spliceai_scores.json'), ('pan', 'results/mfass_pangolin_scores.json')):
    if os.path.exists(f): ext[nm] = json.load(open(f))
V = [v for v in V if v[vx['id']] in mm and np.isfinite(mm[v[vx['id']]]) and all(v[vx['id']] in e for e in ext.values())]
y = np.array([int(v[vx['label']]) for v in V]); z = lambda q: (q - q.mean()) / (q.std() + 1e-9)
S = {'mms': np.array([mm[v[vx['id']]] for v in V]), 'pyr': np.array([float(v[vx['pyr']]) for v in V]), 'mes': np.array([float(v[vx['mes']]) for v in V])}
for nm, e in ext.items(): S[nm] = np.array([e[v[vx['id']]] for v in V])
S['pyr+mms'] = z(S['pyr']) + z(S['mms']); PAIRS = [('pyr+mms', 'mms'), ('pyr', 'mms')]
if len(ext) == 2:
    S['sai+pan+mms'] = z(S['sai']) + z(S['pan']) + z(S['mms']); S['pyr+sai+pan+mms'] = z(S['pyr']) + S['sai+pan+mms']; PAIRS.append(('pyr+sai+pan+mms', 'sai+pan+mms'))
mot = np.array([v[vx['motif']] == 'True' for v in V]); can = np.array([v[vx['canon']] == 'True' for v in V]); res = {}; rs = np.random.RandomState(0)
for nm, m in (('all', np.ones(len(V), bool)), ('noncanonical', ~can), ('motif_noncanon', mot & ~can), ('outside_motif', ~mot)):
    if not (0 < y[m].sum() < m.sum()): continue
    rr = dict(n=int(m.sum()), n_pos=int(y[m].sum()))
    for k, q in S.items(): rr['auc_' + k] = float(AU(y[m], q[m])); rr['ap_' + k] = float(AP(y[m], q[m]))
    ym = y[m]; o = []
    for _ in range(500):
        i = rs.randint(0, len(ym), len(ym))
        if 0 < ym[i].sum() < len(i): o.append([AP(ym[i], S[a][m][i]) - AP(ym[i], S[b][m][i]) for a, b in PAIRS])
    o = np.array(o); rr['ci'] = {'%s-(%s)' % p: [float(np.percentile(o[:, j], 2.5)), float(np.percentile(o[:, j], 97.5))] for j, p in enumerate(PAIRS)}
    res[nm] = rr; log('MM', nm, 'n', rr['n'], 'pos', rr['n_pos'], ' '.join('%s %.3f/%.3f' % (k, rr['auc_' + k], rr['ap_' + k]) for k in S)); log('MMCI', nm, {k: [round(x, 3) for x in v] for k, v in rr['ci'].items()})
json.dump(res, open('results/mfass_mmsplice.json', 'w'), indent=1); log('MM_DONE')
