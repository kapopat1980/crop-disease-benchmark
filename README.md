# Crop disease & pest recognition benchmark — code and results

Code, result tables and figures accompanying the paper

> **[TODO: paper title]**, [TODO: authors], [TODO: journal, year, DOI]

The study benchmarks five ImageNet-pretrained classifiers — MobileNetV3-Small, ResNet-50, EfficientNetV2-S,
ConvNeXt-Tiny and Swin-T — on a public, strongly imbalanced 44-class, multi-crop disease/pest image dataset
(15,201 images after cleaning; 449:1 class-imbalance ratio) under a leakage-aware protocol, with bootstrap
confidence intervals, exact McNemar tests, per-class analysis, calibration and efficiency measurements, and Grad-CAM.

## Main results (test set, single run, seed 42; brackets = 95 % bootstrap CI over test images)

| Model | Accuracy (%) | Macro-F1 (%) | Params (M) | GFLOPs | Latency bs=1 (ms) |
|---|---|---|---|---|---|
| MobileNetV3-Small | 95.22 [94.39, 96.10] | 80.86 [78.53, 84.38] | 1.56 | 0.18 | 5.11 |
| ResNet-50 | 97.11 [96.41, 97.81] | 81.37 [79.37, 87.08] | 23.6 | 13.51 | 6.47 |
| EfficientNetV2-S | 97.24 [96.58, 97.94] | 84.54 [82.38, 88.88] | 20.23 | 9.41 | 16.24 |
| ConvNeXt-Tiny | 97.76 [97.15, 98.38] | 84.63 [81.97, 90.24] | 27.85 | 14.73 | 6.8 |
| Swin-T | 97.15 [96.41, 97.85] | 85.78 [82.68, 89.60] | 27.55 | 17.29 | 11.98 |

Overall accuracy is high, but macro-F1 is much lower: six classes with ≤ 5 test images drive the gap.
The four larger models are statistically hard to separate (exact McNemar tests, see
`results/tables/mcnemar.csv`). Full discussion and caveats are in the paper.

## Repository layout

```
crop_disease_benchmark.py     complete experiment (cleaning, split, training, evaluation, efficiency, Grad-CAM)
requirements.txt
scripts/
  reproduce.sh                full experiment exactly as in the paper
  smoke_test.sh               minutes-long end-to-end test on a small subset
  check_results.py            GPU-free consistency checks of the shipped results
results/
  tables/                     main/extended tables (CSV, LaTeX), per-run metrics, McNemar, per-class F1,
                              confusion matrices, top confusions, training histories, efficiency
  figures/                    all figures (PNG + PDF)
  splits/                     cleaned file index, exact train/val/test split (seed 42), class names
docs/
  DATA.md                     dataset access, cleaning procedure, known label issues
  REPRODUCIBILITY.md          recipe, hardware, determinism notes
```

## Quick start

```bash
git clone https://github.com/[TODO-user]/crop-disease-benchmark.git
cd crop-disease-benchmark
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # choose the torch build matching your CUDA

# 1. check that the shipped results are consistent (no GPU, no images needed)
python scripts/check_results.py

# 2. get the dataset (see docs/DATA.md), then run a smoke test
bash scripts/smoke_test.sh /path/to/20k-multi-class-crop-disease-images

# 3. full experiment (~4–5 h on one Tesla T4)
bash scripts/reproduce.sh /path/to/20k-multi-class-crop-disease-images ./journal_results
```

On Kaggle: attach the dataset to a notebook and run `!python crop_disease_benchmark.py --epochs 30`
(the default `--data_path` points to the Kaggle input directory). All options: `python crop_disease_benchmark.py --help`.

## Reproducibility notes

* Seeds are fixed (training and split seed 42) but `cudnn.benchmark=True` is enabled, so re-runs are **not bit-wise
  identical**; expect small deviations. The paper reports a single training run per model.
* The dataset is **not** included (size and licence) — see `docs/DATA.md`. Use `results/splits/split_seed42.csv`
  (`rel_path` column) to recover the exact split on your copy of the data.
* Model checkpoints and per-image predictions are not stored in the repository.

## Citation

Please cite the paper and, if you use the code or result files, this repository (see `CITATION.cff`;
archived version: [TODO: Zenodo DOI]).

## Licence

Code: MIT (see `LICENSE`). Result tables and figures: [TODO: e.g. CC BY 4.0]. The image dataset has its own licence on Kaggle.

## Contact

[TODO: corresponding author, e-mail]
