# Download the Genomic Benchmarks suite (Gresova et al.) from the Hugging Face hub and write it in the format of the
# other tasks. usage: genomic_benchmarks.py
# Writes data/nt/gb_<name>_{train,test}.fna and tasks_gb.txt. The original train and test splits are kept. Sequences
# of a dataset with variable length are cropped around their centre or padded with N on both sides to the median
# length of its training set (at most 1,000 nt), because the position-specific levels need one length per task.
import os, io, json, urllib.request, numpy as np, pandas as pd
NAMES = ['human_enhancers_cohn', 'human_enhancers_ensembl', 'human_nontata_promoters', 'human_ocr_ensembl', 'human_ensembl_regulatory', 'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm', 'dummy_mouse_enhancers_ensembl', 'drosophila_enhancers_stark']
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'python'}), timeout=600).read()
def fit(s, n):
    s = s.upper()
    if len(s) > n: a = (len(s) - n) // 2; return s[a:a + n]
    a = (n - len(s)) // 2; return 'N' * a + s + 'N' * (n - len(s) - a)
done = []; meta = {}; os.makedirs('data/nt', exist_ok=True)
for nm in NAMES:
    try:
        ds = 'katarinagresova/Genomic_Benchmarks_' + nm; idx = json.loads(get('https://huggingface.co/api/datasets/%s/parquet' % ds).decode()); cfg = idx[sorted(idx)[0]]; D = {}
        for sp in ('train', 'test'): D[sp] = pd.concat([pd.read_parquet(io.BytesIO(get(u))) for u in cfg[sp]], ignore_index=True)
        ln = D['train']['seq'].str.len().values; n = int(min(1000, np.median(ln))); m = dict(length=n, min_len=int(ln.min()), max_len=int(ln.max()), classes=int(D['train']['label'].nunique()))
        for sp in ('train', 'test'):
            with open('data/nt/gb_%s_%s.fna' % (nm, sp), 'w') as f:
                for i, (s, y) in enumerate(zip(D[sp]['seq'].values, D[sp]['label'].values)): f.write('>s%d|%d\n%s\n' % (i, int(y), fit(s, n)))
            m[sp] = int(len(D[sp]))
        meta[nm] = m; done.append(nm); print('GB', nm, m, flush=True)
    except Exception as ex: print('GBFAIL', nm, repr(ex)[:200], flush=True)
json.dump(meta, open('results/genomic_benchmarks_meta.json', 'w'), indent=1); open('tasks_gb.txt', 'w').write(''.join('gb_%s\n' % x for x in done)); print('GB_DONE', len(done), flush=True)
