# CGR

Position-pyramid chaos game representation (CGR) features for DNA sequence classification.

A frequency chaos game representation (FCGR) of order k is a table of k-mer counts. It records how
often each k-mer occurs but not where. This code counts k-mers at several position resolutions
(whole sequence, 4 bins, 16 bins, 64 bins, single positions) and gives each resolution, called a
level, its own small branch of a network. Coarse levels capture composition-like signals, and fine
levels capture motifs at fixed positions such as splice sites.

## Install

Python 3.7 or later.

```bash
git clone https://github.com/sarwanpasha/CGR.git
cd CGR
pip install numpy scikit-learn        # enough for the quick start
pip install -r requirements.txt       # everything, including PyTorch and the foundation-model baselines
```

## Quick start on synthetic data

The demo needs no data download and only numpy and scikit-learn. It takes a few seconds on a laptop.

```bash
python demo_synthetic.py
```

It simulates sequences of 200 nt in 2 classes. In class 1 the motif `GTAA` is planted near the middle
of the sequence with a positional spread of 2 nt, 20 nt, or at a random position. For each case it
prints the test MCC of every level on its own and of all levels combined:

```text
levels (k, bins); 0 bins means one bin per position: [(5, 1), (4, 4), (3, 16), (3, 64), (2, 0)]
features per level for L=200: [1024, 1024, 1024, 4096, 3200]
motif position            (5,1)    (4,4)   (3,16)   (3,64)    (2,L)  combined
spread 2 nt               0.350    0.590    0.769    0.782    0.496     0.828
spread 20 nt              0.374    0.471    0.350    0.200    0.060     0.528
random                    0.334    0.290    0.160    0.066    0.036     0.354
```

How to read it: level `(5,1)` alone is the plain FCGR. When the motif sits at a nearly fixed position,
the fine levels are far better than the plain FCGR. When its position varies more, coarser levels take
over. The combination follows the best level in every case, so one feature set serves all three.
Exact numbers can differ slightly between library versions.

## Use the features in your own code

```python
import numpy as np
from cgr_features import feat, dims, norm

train_seqs = ["ACGT...", ...]          # equal-length DNA strings
test_seqs  = ["TTGA...", ...]
L = len(train_seqs[0])

X  = np.stack([feat(s, L) for s in train_seqs])    # log(1 + count), all levels concatenated
Xt = np.stack([feat(s, L) for s in test_seqs])
X, Xt = norm(X, Xt)                                # standardize with training statistics
print(dims(L))                                     # number of features per level
```

`feat(s, L, levels)` accepts a custom list of `(k, bins)` pairs, where `bins = 0` means one bin per
position. Any classifier can be trained on `X`. `train_pyramid.py` contains the per-level network.

## Train on your own data

Put two FASTA files in `data/nt/`, named `<task>_train.fna` and `<task>_test.fna`. Each header must
end with `|<integer label>`, and all sequences of a task must have the same length:

```text
>seq1|0
ACGTTGCA...
>seq2|1
GGCATCGA...
```

```bash
mkdir -p results emb logs
python train_pyramid.py <task> base 0      # pyramid network, seed 0
python train_cnn.py <task> 0               # one-hot CNN branch for fixed-position tasks
```

Each run prints the test accuracy and MCC and writes a JSON summary and the test probabilities to
`results/`. A GPU is used if available and is not required.

The 18 tasks of the Nucleotide Transformer benchmark can be exported to this format from the Hugging
Face dataset `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`.

## Full pipeline

```bash
for s in 0 1 2; do for v in base r2 k6 k6w_lr3; do python train_pyramid.py enhancers $v $s; done; done
python fm_embed_probe.py dnabert2 enhancers          # frozen DNABERT-2 embeddings and a linear probe
python train_pyramid.py enhancers fuse_dnabert2 0    # pyramid with the embeddings as an extra branch
python fm_finetune.py dnabert2 enhancers 0 3         # full fine-tuning baseline
python hybrid_ensemble.py                            # probability-level ensembles over variants and seeds
```

The ClinVar scripts expect, under `data/cv/`, the files `clinvar.vcf.gz` (GRCh38),
`gencode.v44.annotation.gtf.gz` and `GRCh38.primary_assembly.genome.fa.gz`.

## Files

| File | Purpose |
| --- | --- |
| `demo_synthetic.py` | Quick start on simulated sequences |
| `cgr_features.py` | Pyramid k-mer counts, FASTA reader, standardization |
| `train_pyramid.py` | Per-level pyramid network (variants `base`, `r2`, `k6`, `k6w_lr3`, `fuse_<model>`) |
| `train_cnn.py` | One-hot dilated CNN branch |
| `hybrid_ensemble.py`, `aggregate.py` | Ensembles and result tables |
| `fm_embed_probe.py`, `fm_finetune.py` | Foundation-model baselines (frozen probe, full fine-tuning) |
| `kmer_baselines.py` | Gapped k-mer, weighted CGR and plain FCGR baselines |
| `leakage_audit.py` | Train/test similarity check |
| `strand_linear.py` | Strand-specific versus reverse-complement-merged k-mers |
| `efficiency.py` | Parameter counts and inference speed |
| `clinvar_splice_variants.py`, `clinvar_spliceai_baseline.py` | Scoring of ClinVar variants near splice junctions, and the SpliceAI comparison |
| `combine_stats.py` | Average the pyramid ensemble with a fine-tuned foundation model, with paired bootstrap intervals |
| `level_ablation.py` | Train on one pyramid level at a time (resolution spectrum of a task) |
| `positional_spread.py` | Positional spread of the most discriminative k-mers and their best bin width |
| `region_contrast.py` | Whole-window, binned, shuffled-bin, center/flank and left/right k-mer counts |
| `orientation.py` | Re-orient windows by gene strand, by TSS side or at random, and retrain |
| `window_profile.py`, `window_layout.py` | GC and CpG content along the window per class, and genomic layout of the windows |
| `train_overlap.py`, `train_fold.py` | Variants with half-shifted bins and with strand-folded features |
| `comparators.py` | Positional FCGR (moments), oligo-kernel features and weighted-degree features as comparators |
| `controls.py` | Validation-selected single level, and GC-matched positive and negative windows |
| `equal_ensembles.py` | Ensembles of equal size from stored probabilities |
| `clinvar_pr.py`, `cv_extra.py` | Precision-recall on the ClinVar stratum, overlap with training windows, MaxEntScan baseline |
| `final_report.py` | Print the numbers behind the tables from stored result files |
| `encode_peaks.py` | Peak-centred and edge-anchored windows from ENCODE K562 peak files, with GC-matched negatives |
| `mfass_splice.py`, `mfass_spliceai.py`, `mfass_pangolin.py`, `mfass_mmsplice.py` | Score experimentally measured splicing variants (MFASS) with the splice-site models and MaxEntScan, and the SpliceAI, Pangolin and MMSplice comparisons |
| `anchor_check.py` | Intersect the benchmark histone-mark windows with ENCODE peaks to locate peak boundaries inside the windows |
| `theory_background.py` | Simulation of the optimal bin width under overdispersed and clustered background |
| `genomic_benchmarks.py` | Download the Genomic Benchmarks suite and write it in the task format |
| `fm_queue.sh` | Queue of fine-tuning runs that waits for an idle GPU |
| `results_digest.py` | Compact tables of the control, comparator, peak-centred and MFASS results |
| `theory_resolution_law.py` | Constants of the bin-width analysis |
| `synthetic_optimal_width.py`, `synthetic_pyramid_efficiency.py`, `synthetic_strand.py` | Simulations that check the analysis |

## License

MIT
