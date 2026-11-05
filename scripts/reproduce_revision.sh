#!/usr/bin/env bash
# All revision experiments in sequence (about 70 T4 GPU-hours). Finished runs are skipped, so the script can be resumed.
# Usage: bash scripts/reproduce_revision.sh /path/to/20k-multi-class-crop-disease-images ./revision_results
set -euo pipefail
DATA="${1:-./data/20k-multi-class-crop-disease-images}"
OUT="${2:-./revision_results}"
X=(python revision_experiments.py --data_path "$DATA" --output_dir "$OUT")
ALL=(--models MobileNetV3-Small ResNet-50 EfficientNetV2-S ConvNeXt-Tiny Swin-T)
TWO=(--models MobileNetV3-Small ConvNeXt-Tiny)

"${X[@]}" --stage audit
"${X[@]}" --stage orig "${TWO[@]}" --orig_split results/splits/split_seed42.csv
"${X[@]}" --stage cv "${ALL[@]}" --variants ce_ls
"${X[@]}" --stage explain "${ALL[@]}"
"${X[@]}" --stage cv "${TWO[@]}" --variants ce cb la rs
"${X[@]}" --stage hpsweep "${ALL[@]}" --epochs 20
python revision_analysis.py --out "$OUT"
echo "All outputs in $OUT (analysis tables in $OUT/analysis)"
