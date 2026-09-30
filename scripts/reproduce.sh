#!/usr/bin/env bash
# Full experiment as reported in the paper: 5 models, seed 42, 30 epochs max.
# Needs a CUDA GPU (paper: one NVIDIA Tesla T4, ~254 min total training time) and the dataset (docs/DATA.md).
set -euo pipefail
DATA_PATH="${1:-./data/20k-multi-class-crop-disease-images}"
OUT="${2:-./journal_results}"
python crop_disease_benchmark.py \
  --data_path "$DATA_PATH" --output_dir "$OUT" \
  --models all --seeds 42 --split_seed 42 \
  --epochs 30 --patience 8 --img_size 288 --batch_size 32
echo "Results written to $OUT (tables/, figures/, runs/, config.json, environment.json)"
