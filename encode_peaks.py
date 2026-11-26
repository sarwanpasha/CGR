# Peak-centred windows from ENCODE ChIP-seq peaks with matched negatives. usage: encode_peaks.py [n_train_pos] [cell] [tag]
# Without a tag: K562, negatives matched in GC content, tasks enc_<target> and encL_<target>. With a tag (e.g. c or h):
# negatives matched in GC content (2% bins) and in the number of CpG dinucleotides, tasks <tag>enc_ and <tag>encL_;
# positives whose bin cannot be matched are dropped so that the 2 classes keep the same composition.
# For each target the script asks the ENCODE portal for a released GRCh38 narrowPeak file, downloads it, and writes
#   data/nt/enc_<target>_{train,test}.fna   windows of 1,000 nt centred on the peak summit
#   data/nt/encL_<target>_{train,test}.fna  the same peaks with the window starting at the left end of the peak
# Negatives are windows that overlap no peak of the target (within 1 kb) and have the same GC histogram (1% bins) as
# the positives. Test windows come from chr20 and chr21, training windows from the other autosomes and chrX.
import sys, os, gzip, json, time, urllib.request, numpy as np
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 15000; CELL = sys.argv[2] if len(sys.argv) > 2 else 'K562'; TAG = sys.argv[3] if len(sys.argv) > 3 else ''; L = 1000; t0 = time.time(); rs = np.random.RandomState(0)
HIS = ['H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K4me2', 'H3K9ac', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H4K20me1', 'H2AFZ']; TF = ['CTCF', 'GATA1', 'REST', 'MAX'] if CELL == 'K562' else ['CTCF', 'HNF4A', 'REST', 'MAX']
B = 'https://www.encodeproject.org'; os.makedirs('data/enc', exist_ok=True)
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={'accept': 'application/json', 'User-Agent': 'python'}), timeout=120).read()
def pick(t, assay):
    u = B + '/search/?type=File&file_format=bed&file_format_type=narrowPeak&assembly=GRCh38&status=released&biosample_ontology.term_name=%s&target.label=%s&assay_title=%s&format=json&limit=200&field=accession&field=href&field=output_type&field=preferred_default&field=biological_replicates' % (CELL, t, assay)
    g = json.loads(get(u).decode()).get('@graph', []); pref = ['replicated peaks', 'conservative IDR thresholded peaks', 'pseudo-replicated peaks', 'optimal IDR thresholded peaks', 'pseudoreplicated peaks', 'peaks']
    g = sorted(g, key=lambda f: (not f.get('preferred_default', False), pref.index(f['output_type']) if f.get('output_type') in pref else 99, -len(f.get('biological_replicates', [])), f['accession']))
    return g[0] if g else None
G = {}; cur = None; buf = []; want = set(['chr%d' % i for i in range(1, 23)] + ['chrX'])
for l in gzip.open('data/cv/GRCh38.primary_assembly.genome.fa.gz', 'rt'):
    if l[0] == '>':
        if cur in want: G[cur] = ''.join(buf).upper()
        cur = l[1:].split()[0]; buf = []
    elif cur in want: buf.append(l.rstrip())
if cur in want: G[cur] = ''.join(buf).upper()
log('genome', len(G)); TEST = ('chr20', 'chr21'); META = {}
def gc(q):
    g = 100.0 * (q.count('G') + q.count('C')) / len(q)
    if not TAG: return int(g)
    n = q.count('CG'); return (int(g / 2), 0 if n < 5 else 1 if n < 10 else 2 if n < 20 else 3 if n < 40 else 4 if n < 80 else 5)
def write(name, split, rows):
    with open('data/nt/%s_%s.fna' % (name, split), 'w') as f:
        for c, s, y in rows: f.write('>%s:%d-%d|%d\n%s\n' % (c, s, s + L, y, G[c][s:s + L]))
for t, assay in [(x, 'Histone+ChIP-seq') for x in HIS] + [(x, 'TF+ChIP-seq') for x in TF]:
    try:
        f = pick(t, assay)
        if f is None: log('NOFILE', t); continue
        p = 'data/enc/%s%s_%s.bed.gz' % ('' if CELL == 'K562' else CELL + '-', t, f['accession'])
        if not os.path.exists(p): open(p, 'wb').write(urllib.request.urlopen(urllib.request.Request(B + f['href'], headers={'User-Agent': 'python'}), timeout=600).read())
        P = []
        for l in gzip.open(p, 'rt'):
            a = l.split('\t')
            if a[0] in want: P.append((a[0], int(a[1]), int(a[2]), float(a[6]), int(a[9]) if len(a) > 9 else -1))
        P.sort(key=lambda x: -x[3]); I = {}
        for c in want:
            v = np.array(sorted((s, e) for cc, s, e, _, _ in P if cc == c)).reshape(-1, 2); I[c] = (v[:, 0], np.maximum.accumulate(v[:, 1])) if len(v) else (np.zeros(0, int), np.zeros(0, int))
        def free(c, s):
            b, e = I[c]; j = np.searchsorted(b, s + 2 * L, 'right') - 1; return not (j >= 0 and e[j] > s - L)
        wid = np.median([e - s for _, s, e, _, _ in P]); out = {}
        for nm, anchor in ((TAG + 'enc', 'summit'), (TAG + 'encL', 'left')):
            pos = {'train': [], 'test': []}; seen = set()
            for c, s, e, sig, off in P:
                a = (s + (off if off >= 0 else (e - s) // 2) - L // 2) if anchor == 'summit' else s
                sp = 'test' if c in TEST else 'train'; lim = NP if sp == 'train' else NP // 10
                if len(pos[sp]) >= lim or a < 0 or a + L > len(G[c]) or (c, a // 200) in seen: continue
                q = G[c][a:a + L]
                if 'N' in q: continue
                seen.add((c, a // 200)); pos[sp].append((c, a, 1))
            for sp in ('train', 'test'):
                need = {}
                for c, a, _ in pos[sp]: g = gc(G[c][a:a + L]); need[g] = need.get(g, 0) + 1
                chs = [c for c in sorted(want) if (c in TEST) == (sp == 'test')]; ln = np.array([len(G[c]) for c in chs], float); neg = []; tries = 0; tot = len(pos[sp])
                while len(neg) < tot and tries < (600 if TAG else 400) * tot:
                    tries += 1; c = chs[rs.choice(len(chs), p=ln / ln.sum())]; a = int(rs.randint(0, len(G[c]) - L)); q = G[c][a:a + L]
                    if 'N' in q: continue
                    g = gc(q)
                    if need.get(g, 0) > 0 and free(c, a): need[g] -= 1; neg.append((c, a, 0))
                if TAG:
                    keep = []
                    for c, a, y in pos[sp][::-1]:
                        g = gc(G[c][a:a + L])
                        if need.get(g, 0) > 0: need[g] -= 1
                        else: keep.append((c, a, y))
                    pos[sp] = keep[::-1]
                rows = pos[sp] + neg; rows = [rows[i] for i in rs.permutation(len(rows))]; write('%s_%s' % (nm, t), sp, rows); out['%s_%s' % (nm, sp)] = (len(pos[sp]), len(neg))
        META[t] = dict(accession=f['accession'], output_type=f.get('output_type'), peaks=len(P), median_width=float(wid), **{k: list(v) for k, v in out.items()}); log('ENC', t, META[t])
    except Exception as ex: log('FAIL', t, repr(ex)[:200])
json.dump(META, open('results/encode_meta%s.json' % ('_' + TAG if TAG else ''), 'w'), indent=1); open('tasks_%senc.txt' % TAG, 'w').write(''.join('%senc_%s\n%sencL_%s\n' % (TAG, t, TAG, t) for t in META)); log('DONE', len(META))
