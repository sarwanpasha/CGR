# Compact digest of the control, comparator, peak-centred and MFASS results. usage: results_digest.py sel|gc|cmp|enc|mf
import sys, glob, json, numpy as np
W = sys.argv[1]; ab = lambda t: t.replace('splice_sites_', 'ss_').replace('promoter_', 'pr_').replace('enhancers', 'enh')[:13].ljust(13)
J = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(pat))]
if W == 'sel':
    R = [j for j in J('results/ctl_select_*_s0.json') if not j['task'].startswith('enc')]
    for j in R: print(ab(j['task']), j['chosen'].ljust(6), '%.3f %.3f %.3f' % (j['chosen_test'], j['all_test'], j['best_test']))
    print('MEAN n%d chosen %.4f all %.4f oracle %.4f | all>chosen %d' % (len(R), np.mean([j['chosen_test'] for j in R]), np.mean([j['all_test'] for j in R]), np.mean([j['best_test'] for j in R]), sum(j['all_test'] > j['chosen_test'] for j in R)))
if W == 'gc':
    for j in J('results/ctl_gc_*_s0.json'): print(ab(j['task']), 'n %5d/%5d | whole %.3f>%.3f bins4 %.3f>%.3f all %.3f>%.3f' % (j['n_train'], j['n_train_all'], j['whole4_orig'], j['whole4_matched'], j['bins4_orig'], j['bins4_matched'], j['all_orig'], j['all_matched']))
if W == 'cmp':
    T = [l.strip() for l in open('tasks.txt') if l.strip()]; M = {}
    for m in ('pfcgr', 'oligo', 'wd'):
        for j in J('results/cmp_%s_*.json' % m): M[(m, j['task'])] = j['test']
    for t in T: print(ab(t), ' '.join(('%.3f' % M[(m, t)]) if (m, t) in M else '  -  ' for m in ('pfcgr', 'oligo', 'wd')))
    print('MEAN         ', ' '.join('%.4f(n%d)' % (np.mean([v for (mm, t), v in M.items() if mm == m]), sum(mm == m for mm, t in M)) for m in ('pfcgr', 'oligo', 'wd') if any(mm == m for mm, t in M)))
if W == 'enc':
    meta = json.load(open('results/encode_meta.json'))
    for t, m in meta.items():
        o = [t.ljust(9), m['accession'], 'w%d' % m['median_width']]
        for p in ('enc', 'encL'):
            f = glob.glob('results/ctl_select_%s_%s_s0.json' % (p, t))
            if f:
                j = json.load(open(f[0])); L = j['levels']; o.append(p + ' ' + ' '.join('%.3f' % L[k][1] for k in L) + ' ' + j['chosen'])
        print(' | '.join(o))
if W == 'mf':
    j = json.load(open('results/mfass.json')); print('mapping', j['mapping'])
    for s, r in j['strata'].items():
        print(s, 'n', r['n'], 'pos', r['n_pos']); print('  auc', ' '.join('%s %.3f' % (k[4:], v) for k, v in r.items() if k.startswith('auc_'))); print('  ap ', ' '.join('%s %.3f' % (k[3:], v) for k, v in r.items() if k.startswith('ap_'))); print('  ci ', {k: [round(x, 3) for x in v] for k, v in r['ci'].items()})
