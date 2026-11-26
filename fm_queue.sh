#!/bin/bash
# Fine-tune foundation models on a list of tasks, one run at a time, each on a GPU that is idle when the run starts.
# usage: fm_queue.sh <fine-tuning script> <seed> <task> [<task> ...]
PY=$1; S=$2; shift 2; mkdir -p logs
for T in "$@"; do for M in ntv2_250 dnabert2; do
  [ -f results/fmft_${M}_${T}_s$S.json ] && continue
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<600{print $1; exit}')
    [ -n "$g" ] && break; sleep 300
  done
  CUDA_VISIBLE_DEVICES=$g python3 $PY $M $T $S 3 2>&1 | tail -n 6 > logs/ft_${M}_${T}_s$S.log
done; done
