#!/usr/bin/env bash
# Minutes-long end-to-end test on a tiny subset (40 images/class, 2 epochs, one model).
# Verifies that the environment, the dataset path and the whole pipeline work. NOT for reporting results.
set -euo pipefail
DATA_PATH="${1:-./data/20k-multi-class-crop-disease-images}"
python crop_disease_benchmark.py --data_path "$DATA_PATH" --output_dir ./smoke_results \
  --epochs 2 --seeds 42 --models ConvNeXt-Tiny --max_per_class 40 --num_workers 2
