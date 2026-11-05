# Crop disease & pest recognition — audited, crop-aware benchmark (code, data and results)

Code, derived data and results accompanying the paper

> **Beyond Headline Accuracy: Data Integrity, Crop Context and Imbalance-Aware Training in Multi-Crop Disease and Pest
> Recognition with Convolutional and Transformer Networks**
> Divyakant Meva, Kalpesh Popat — Faculty of Computer Applications, Marwadi University, Rajkot, India.
> Manuscript submitted for publication (2026).

The study evaluates five ImageNet-pretrained classifiers — MobileNetV3-Small, ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny
and Swin-T — on the public Kaggle dataset *20k+ Multi-Class Crop Disease Images* (five crops, 15,201 images after
cleaning) with an **audited protocol**: per-file cleaning log, corrected label taxonomy (44 → 42 classes, host crop for
every class), near-duplicate detection (perceptual hash + embedding similarity), **five-fold stratified group
cross-validation** (near-duplicate groups never cross folds), imbalance-mitigation strategies with a plain
cross-entropy control, crop-conditioned inference, calibration analysis and multi-model Grad-CAM.

## Main results

| Finding | Result |
|---|---|
| Leakage in the conventional random 70/15/15 split | 1,009 of 2,281 test images (44%) have a near-duplicate in training; retrained models score 99.9–100% on these but 91.8–95.4% on the rest |
| Architectures under group cross-validation | Accuracy 94.4–96.4%, macro-F1 81.4–85.6%; the four mid-sized networks are statistically tied (Holm-adjusted McNemar p ≥ 0.21); MobileNetV3-Small is worse |
| Rare classes (13 classes < 100 images) | Macro-F1 61–65% vs 91–95% for the rest; failures are sibling-class confusions and label errors |
| Imbalance methods vs plain cross-entropy | Class-balanced loss, logit adjustment and re-sampling help MobileNetV3-Small's rare classes (Holm p ≤ 0.02) but add nothing for ConvNeXt-Tiny |
| Crop context | Knowing the crop removes 18–23% of errors without retraining |
| Calibration | Label smoothing → under-confidence (gap −6.9 to −9.8 pts); temperature scaling → ECE 0.6–1.2% |

Pooled cross-validated results (reference recipe, 15,201 test predictions per model; brackets = 95% bootstrap CI):

| Model | Accuracy (%) | Macro-F1 (%) | Tail macro-F1 (%) |
|---|---|---|---|
| MobileNetV3-Small | 94.39 | 81.43 [80.0, 82.6] | 60.7 |
| ResNet-50 | 96.16 | 84.46 [83.2, 85.8] | 63.8 |
| EfficientNetV2-S | 96.41 | 84.57 [83.5, 85.5] | 63.1 |
| ConvNeXt-Tiny | 96.39 | 85.01 [84.0, 86.0] | 63.9 |
| Swin-T | 96.44 | 85.57 [84.1, 86.8] | 65.5 |

## Repository layout

```
crop_disease_benchmark.py       original single-split experiment (cleaning, split, training, evaluation, efficiency, Grad-CAM)
revision_experiments.py         audit, group cross-validation, imbalance variants, LR sweep, single-split retraining, Grad-CAM
revision_analysis.py            CPU-only analysis: pooled metrics, paired tests, calibration, crop-conditioned inference
notebooks/
  crop_disease_revision_kaggle.ipynb   self-contained Kaggle notebook (both revision scripts embedded; sessions 0–18)
revision_results/               outputs of all revision runs (no model checkpoints)
  audit/                        cleaning log, taxonomy, near-duplicate pairs, master index with duplicate groups
  cv_corrected/                 folds.csv, classes.json, <variant>/<model>/fold<k>/{result.json, history.csv, logits.npz}
                                variants: ce_ls (reference), ce (plain CE control), cb, la, rs
  orig_split/                   two models retrained on the original single split (leakage analysis)
  hpsweep/                      learning-rate sensitivity (5 models × 3 rates, fold 0)
  analysis/                     all tables used in the paper (cv_summary, McNemar, per-class, calibration, leakage, …)
  figures/, tables/             Grad-CAM best/worst cases, deletion AUC
paper_figures/                  figures exactly as submitted (Fig1–Fig8b, FigA1, graphical abstract)
supplementary/                  Supplementary Material (docx) and machine-readable supplementary data
results/                        original single-split results (seed 42) of the first version
scripts/                        reproduce.sh (original run), reproduce_revision.sh, smoke_test.sh, check_results.py
docs/                           DATA.md (dataset, cleaning, taxonomy, duplicates), REPRODUCIBILITY.md
```

## Reproducing the results

**Analysis only (CPU, minutes)** — all paper tables from the shipped outputs:

```bash
pip install -r requirements.txt
python revision_analysis.py --out revision_results          # writes revision_results/analysis/
```

**Full experiments (GPU)** — easiest on Kaggle: import `notebooks/crop_disease_revision_kaggle.ipynb`, attach the
dataset, set `SESSION` and *Save & Run All*. The session table at the top of the notebook lists what each session
runs (about 70 T4 GPU-hours in total). Locally, `scripts/reproduce_revision.sh` runs the same stages in sequence.
Finished runs are skipped automatically, so interrupted sessions can be resumed.

## Reproducibility notes

* Folds are deterministic (StratifiedGroupKFold, seed 42) and stored in `revision_results/cv_corrected/folds.csv`;
  runs from different machines were verified to use identical folds and test indices before pooling.
* `cudnn.benchmark=True` is enabled, so re-runs are seed-controlled but **not bit-wise identical**.
* The dataset is **not** included — see `docs/DATA.md`. `revision_results/audit/master_index.csv` lists every retained
  image (`rel_path`), its corrected label, host crop and duplicate group.
* Model checkpoints are not stored (size); out-of-fold logits are (`logits.npz`, float16).

## Citation

Please cite the paper (citation will be added on publication) and, if you use the code or result files, this
repository (see `CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). Derived data, result tables and figures: CC BY 4.0. The image dataset has its own licence on
Kaggle.

## Contact

Kalpesh Popat (corresponding author) — kapopat@gmail.com
