# Combine the pyramid ensemble with a fine-tuned foundation model and attach bootstrap intervals.
# Uses saved test probabilities only. usage: combine_stats.py [model] [n_boot]
# Prints per task: fine-tuned model, pyramid ensemble (+CNN on fixed-position tasks), their average,
# and a paired bootstrap (resampling test sequences) of the MCC difference combined - fine-tuned.
import sys, glob, json, numpy as np, cgr_features as P
from sklearn.metrics import matthews_corrcoef as M
FM = sys.argv[1] if len(sys.argv) > 1 else 'dnabert2'; B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
T = sorted({f.split('pyr2_base_')[1].rsplit('_s', 1)[0] for f in glob.glob('results/pyr2_base_*_s0.npy')})
T = [t for t in T if __import__('os').path.exists('data/nt/%s_test.fna' % t)]
FX = [t for t in T if t.startswith('splice') or t == 'promoter_tata']
N = ['base', 'r2', 'k6', 'k6w_lr3']
def pr(pat):
    f = sorted(glob.glob(pat)); return (np.mean([np.load(x) for x in f], 0), len(f)) if f else (None, 0)
def sm(p):
    p = np.asarray(p, np.float64)
    if p.min() < 0 or abs(p.sum(1).mean() - 1) > 1e-3:
        e = np.exp(p - p.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    return p
rs = np.random.RandomState(0); o = {}; D = []
for t in T:
    y = P.read(t, 'test')[1]
    pp = np.mean([sm(np.load(f)) for v in N for f in glob.glob('results/pyr2_%s_%s_s[0-9].npy' % (v, t))], 0)
    if t in FX:
        c, n = pr('results/pyr2_cnn_%s_s[0-9].npy' % t)
        if n: pp = (pp + sm(c)) / 2
    f, nf = pr('results/fmft_%s_%s_s[0-9].npy' % (FM, t))
    if not nf: continue
    f = sm(f); a, b, c = f.argmax(1), pp.argmax(1), (f + pp).argmax(1)
    d = []
    for _ in range(B):
        i = rs.randint(0, len(y), len(y)); d.append((M(y[i], a[i]), M(y[i], b[i]), M(y[i], c[i])))
    d = np.array(d); D.append(d)
    r = dict(n=int(len(y)), fm_seeds=nf, fm=float(M(y, a)), pyr=float(M(y, b)), both=float(M(y, c)),
             gain_lo=float(np.percentile(d[:, 2] - d[:, 0], 2.5)), gain_hi=float(np.percentile(d[:, 2] - d[:, 0], 97.5)),
             p_gain=float(np.mean(d[:, 2] - d[:, 0] <= 0)))
    o[t] = r
    print('CMB %-26s n=%5d s=%d fm %.3f pyr %.3f both %.3f gain [%+.3f,%+.3f] p=%.3f' % (t, r['n'], nf, r['fm'], r['pyr'], r['both'], r['gain_lo'], r['gain_hi'], r['p_gain']), flush=True)
D = np.mean(D, 0); k = list(o)
s = dict(tasks=len(k), fm=float(np.mean([o[t]['fm'] for t in k])), pyr=float(np.mean([o[t]['pyr'] for t in k])), both=float(np.mean([o[t]['both'] for t in k])),
         wins_vs_fm=int(sum(o[t]['both'] > o[t]['fm'] for t in k)), wins_vs_best=int(sum(o[t]['both'] > max(o[t]['fm'], o[t]['pyr']) for t in k)),
         gain_lo=float(np.percentile(D[:, 2] - D[:, 0], 2.5)), gain_hi=float(np.percentile(D[:, 2] - D[:, 0], 97.5)),
         pyr_minus_fm_lo=float(np.percentile(D[:, 1] - D[:, 0], 2.5)), pyr_minus_fm_hi=float(np.percentile(D[:, 1] - D[:, 0], 97.5)))
o['summary'] = s
print('CMBMEAN tasks %d fm %.4f pyr %.4f both %.4f wins_fm %d wins_best %d gain [%+.4f,%+.4f] pyr-fm [%+.4f,%+.4f]' % (s['tasks'], s['fm'], s['pyr'], s['both'], s['wins_vs_fm'], s['wins_vs_best'], s['gain_lo'], s['gain_hi'], s['pyr_minus_fm_lo'], s['pyr_minus_fm_hi']), flush=True)
json.dump(o, open('results/combine_%s.json' % FM, 'w'), indent=1)
