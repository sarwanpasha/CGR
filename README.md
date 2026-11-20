# CGR

Position-pyramid chaos-game / k-mer features for DNA sequence classification, with a
lightweight per-level network, a one-hot CNN branch, and scripts to compare against DNA
foundation models.

## Idea

A frequency chaos game representation (FCGR) of order k is a k-mer count table. This code
counts k-mers at several *position resolutions* (whole sequence, 4 bins, 16 bins, 64 bins,
single positions) and feeds each level to its own small branch of a network. Coarse levels
capture composition-like signals, fine levels capture motifs at fixed positions.

## Files

| File | Purpose |
| --- | --- |
| `cgr_features.py` | Feature extraction: pyramid k-mer counts, optional strand-invariant (canonical k-mer) mapping, data reader |
| `train_pyramid.py` | Train the per-level pyramid network (variants: `base`, `r2`, `k6`, `k6w_lr3`, `sp`, `fuse_<model>`) |
| `train_cnn.py` | Position-aware one-hot dilated CNN used as the hybrid branch |
| `aggregate.py` | Collect results and build probability-level ensembles |
| `fm_embed_probe.py` | Frozen foundation-model embeddings (DNABERT-2, Nucleotide Transformer) with a linear probe |
| `fm_finetune.py` | Full fine-tuning of a foundation model on one task |
| `kmer_baselines.py` | Gapped k-mer, weighted CGR and plain FCGR baselines |
| `leakage_audit.py` | MinHash train/test similarity audit |
| `strand_linear.py` | Strand-specific vs strand-invariant linear models |
| `efficiency.py` | Parameter counts and inference throughput |
| `clinvar_splice_variants.py` | Score ClinVar SNVs near splice junctions by the change in predicted splice-site strength |
| `clinvar_spliceai_baseline.py` | SpliceAI baseline on the same variants |
| `theory_resolution_law.py` | Numerical constants of the bin-width (resolution) analysis |
| `synthetic_optimal_width.py`, `synthetic_pyramid_efficiency.py`, `synthetic_strand.py` | Synthetic checks of the analysis |

## Data layout

Scripts expect FASTA files named `data/nt/<task>_train.fna` and `data/nt/<task>_test.fna`,
where each header ends with `|<integer label>`. The 18 tasks of the revised Nucleotide
Transformer benchmark can be exported to this format from the Hugging Face dataset
`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`.

The ClinVar scripts expect, under `data/cv/`: `clinvar.vcf.gz` (GRCh38),
`gencode.v44.annotation.gtf.gz` and `GRCh38.primary_assembly.genome.fa.gz`.

Results are written to `results/` and embeddings to `emb/`; create these folders first.

## Usage

```bash
pip install -r requirements.txt
mkdir -p results emb logs

# pyramid network, three seeds
for s in 0 1 2; do python train_pyramid.py enhancers base $s; done

# frozen DNABERT-2 embeddings + probe, then fusion with the pyramid
python fm_embed_probe.py dnabert2 enhancers
python train_pyramid.py enhancers fuse_dnabert2 0

# one-hot CNN branch (used for fixed-position tasks such as splice sites)
python train_cnn.py splice_sites_donors 0

# full fine-tuning baseline
python fm_finetune.py dnabert2 enhancers 0 3

# tables and ensembles
python aggregate.py
```

## License

MIT
