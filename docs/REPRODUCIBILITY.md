# Reproducibility notes

| item | value |
|---|---|
| Script | `crop_disease_benchmark.py` (= "version 12" of the experiment script, unchanged) |
| Hardware used | 1 × NVIDIA Tesla T4 (Kaggle notebook) |
| Seeds | training seed 42, split seed 42 (single run per model) |
| Recipe | AdamW, lr 1e-4 (backbone) / 1e-3 (head), wd 0.01, batch 32, 288×288, 2 warm-up epochs + cosine to 1 %, ≤ 30 epochs, early stopping (patience 8) on validation macro-F1, label smoothing 0.1, grad-clip 1.0, fp16 AMP, head dropout 0.3 |
| Augmentation | RandomResizedCrop(0.6–1), RandAugment(2, 9), H/V flip, random erasing (p = 0.25) – training split only |
| Class weighting | **off** unless `--class_weights` is passed (all defaults are in `parse_args()`) |
| Pretrained weights | torchvision `*_Weights.DEFAULT` (ImageNet-1k) |
| Determinism | `cudnn.benchmark = True` → runs are seed-controlled but **not bit-wise deterministic**; re-running will give slightly different numbers (see the discussion of run-to-run variance in the paper) |
| Runtime | ≈ 254 min total training for the 5 models on the T4 (per-model times in `results/tables/all_runs.csv`) |

## Verify the shipped results without a GPU
```bash
python scripts/check_results.py
```

## Files produced by a fresh run (`--output_dir`)
`config.json`, `environment.json` (exact package versions), `dataset_meta.json` (cleaning counts), `class_names.json`,
`dataset_summary.csv`, `split_seed*.csv`, `tables/` (main, extended, efficiency, McNemar, per-class F1, confusion matrices,
top confusions, histories), `figures/` (PNG + PDF), `runs/<model>/seed<k>/` (history, result.json, predictions, `best.pt`).
Checkpoints (`best.pt`) and raw predictions are **not** included in this repository (size); see "Archive" below.

## Archive (recommended before submission)
Create a GitHub release (tag `v1.0.0`) and archive it with Zenodo (GitHub ↔ Zenodo integration) to obtain a citable DOI;
put that DOI into the paper's Data/Code availability statement and into `CITATION.cff`.
