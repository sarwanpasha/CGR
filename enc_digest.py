# Summary of level selection on peak-centred windows. usage: enc_digest.py <tasks file> [<tasks file> ...]
# For each task: test MCC of each single level and of all levels (mean over the available seeds), best single level.
import sys, json, glob, numpy as np
for tf in sys.argv[1:]:
    for T in open(tf).read().split():
        F = sorted(glob.glob('results/ctl_select_%s_s*.json' % T))
        if not F: print('%-16s none' % T); continue
        D = [json.load(open(f)) for f in F]; lv = [k for k in D[0]['levels'] if k != 'all']
        m = {k: np.mean([d['levels'][k][1] for d in D]) for k in lv + ['all']}; b = max(lv, key=lambda k: m[k])
        print('%-16s n%d %s all %.3f best %s' % (T, len(D), ' '.join('%.3f' % m[k] for k in lv), m['all'], b), flush=True)
