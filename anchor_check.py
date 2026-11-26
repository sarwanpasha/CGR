# Where do peak boundaries fall inside the benchmark windows? usage: anchor_check.py
# For every histone-mark task whose sequence names carry coordinates, each window is intersected with the ENCODE K562
# peaks of the same mark (files downloaded by encode_peaks.py). For positive windows the script reports how often the
# window overlaps a peak, where the window start lies relative to the start of that peak, and where the peak summit
# and the peak-covered part of the window lie along the window.
import glob, gzip, json, numpy as np
HIS = ['H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K4me2', 'H3K9ac', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H4K20me1']; res = {}
def windows(T):
    W = []
    for sp in ('train', 'test'):
        for l in open('data/nt/%s_%s.fna' % (T, sp)):
            if l[0] != '>': continue
            nm, y = l[1:].strip().rsplit('|', 1)
            try: c, r = nm.split(':'); s, e = r.split('-'); W.append((c, int(s), int(e), int(y)))
            except Exception: pass
    return W
for T in HIS:
    f = glob.glob('data/enc/%s_*.bed.gz' % T)
    if not f: print('NOPEAKS', T, flush=True); continue
    P = {}
    for l in gzip.open(f[0], 'rt'):
        a = l.split('\t'); P.setdefault(a[0], []).append((int(a[1]), int(a[2]), int(a[9]) if len(a) > 9 and int(a[9]) >= 0 else (int(a[2]) - int(a[1])) // 2))
    I = {}
    for c, v in P.items():
        v.sort(); b = np.array([x[0] for x in v]); e = np.array([x[1] for x in v]); I[c] = (b, e, np.maximum.accumulate(e), np.array([x[2] for x in v]))
    W = windows(T)
    if not W: print('NOCOORD', T, flush=True); continue
    L = int(np.median([e - s for _, s, e, _ in W])); nb = 10; out = {}
    for y in (1, 0):
        ov = 0; n = 0; dst = []; den = []; cov = np.zeros(nb); summ = np.zeros(nb + 2); full = 0; lef = 0; rig = 0; both = 0
        for c, s, e, yy in W:
            if yy != y or c not in I: continue
            n += 1; b, pe, me, sm = I[c]; j0 = np.searchsorted(me, s, 'right'); j1 = np.searchsorted(b, e, 'left'); best = None
            m = np.zeros(e - s, bool)
            for j in range(j0, j1):
                if pe[j] > s and b[j] < e:
                    o = min(pe[j], e) - max(b[j], s); m[max(b[j], s) - s:min(pe[j], e) - s] = True
                    if best is None or o > best[0]: best = (o, j)
            if best is None: continue
            ov += 1; j = best[1]; dst.append(s - b[j]); den.append(e - pe[j])
            cov += m.reshape(nb, -1).mean(1) if (e - s) % nb == 0 else np.array([x.mean() for x in np.array_split(m, nb)])
            sp = b[j] + sm[j] - s; summ[0 if sp < 0 else nb + 1 if sp >= e - s else 1 + int(nb * sp / (e - s))] += 1
            cl = m[:max(1, (e - s) // 20)].all(); cr = m[-max(1, (e - s) // 20):].all(); full += m.all(); lef += cl and not cr; rig += cr and not cl; both += cl and cr
        if not ov: out[y] = dict(n=n, overlap=0.0); continue
        dst = np.array(dst); den = np.array(den); vals, cnt = np.unique(dst, return_counts=True); top = [(int(vals[i]), int(cnt[i])) for i in np.argsort(-cnt)[:3]]
        out[y] = dict(n=n, overlap=ov / n, start_exact=float((np.abs(dst) <= 2).mean()), end_exact=float((np.abs(den) <= 2).mean()), start_in_peak=float((dst > 2).mean()), start_before_peak=float((dst < -2).mean()),
                      med_start_offset=float(np.median(dst)), top_start_offsets=top, fully_covered=full / ov, left_edge_only=lef / ov, right_edge_only=rig / ov, both_edges=both / ov,
                      coverage_profile=[float(x) for x in cov / ov], summit_bins=[float(x) for x in summ / ov])
    res[T] = dict(window=L, peaks=f[0].split('/')[-1], pos=out[1], neg=out[0]); p = out[1]
    print('ANC', T, 'L', L, 'n+', p['n'], 'overlap+ %.3f' % p['overlap'], 'overlap- %.3f' % out[0]['overlap'], flush=True)
    if p['overlap'] > 0:
        print('ANC2', T, 'start=peakstart %.3f end=peakend %.3f start_in %.3f start_before %.3f med %.0f top' % (p['start_exact'], p['end_exact'], p['start_in_peak'], p['start_before_peak'], p['med_start_offset']), p['top_start_offsets'], flush=True)
        print('ANC3', T, 'full %.3f leftonly %.3f rightonly %.3f both %.3f' % (p['fully_covered'], p['left_edge_only'], p['right_edge_only'], p['both_edges']), 'cov', ' '.join('%.2f' % x for x in p['coverage_profile']), flush=True)
        print('ANC4', T, 'summit', ' '.join('%.2f' % x for x in p['summit_bins']), flush=True)
json.dump(res, open('results/anchor_check.json', 'w'), indent=1); print('ANC_DONE', flush=True)
