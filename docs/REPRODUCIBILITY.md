# Reproducibility notes

| item | value |
|---|---|
| Scripts | `crop_disease_benchmark.py` (original single split, unchanged); `revision_experiments.py`, `revision_analysis.py` (revision) |
| Notebook | `notebooks/crop_disease_revision_kaggle.ipynb` — embeds both revision scripts; one `SESSION` number per Kaggle run |
| Hardware | NVIDIA Tesla T4 (Kaggle notebooks); about 70 GPU-hours for all revision runs |
| Evaluation | 5-fold StratifiedGroupKFold (seed 42), groups = near-duplicate groups; inner group-stratified validation sixth for early stopping; predictions of the five test folds pooled |
| Reference recipe | AdamW, lr 1e-4 (backbone) / 1e-3 (head), wd 0.01, batch 32, 288×288, 2 warm-up epochs + cosine to 1%, ≤ 30 epochs, early stopping (patience 8) on validation macro-F1, label smoothing 0.1, grad-clip 1.0, fp16 AMP, head dropout 0.3 |
| Augmentation | RandomResizedCrop(0.6–1), RandAugment(2, 9), H/V flip, random erasing (p = 0.25) — training only |
| Imbalance variants | `ce` plain CE (control), `cb` class-balanced CE (β = 0.999), `la` logit-adjusted CE (τ = 1), `rs` √-frequency re-sampling; no label smoothing |
| LR sensitivity | backbone lr 3e-5, 1e-4, 3e-4 (head ×10), fold 0, 20 epochs |
| Pretrained weights | torchvision `*_Weights.DEFAULT` (ImageNet-1k) |
| Determinism | `cudnn.benchmark = True` → seed-controlled but not bit-wise deterministic |

## Notebook sessions
| SESSION | Runs |
|---|---|
| 0 | Retrain the original split (MobileNetV3-Small, ConvNeXt-Tiny) → leakage effect |
| 1 | Smoke test + data-integrity audit |
| 2–4 | 5 architectures, reference recipe, folds 0–4 (+ Grad-CAM in 4) |
| 5–7 or 11–14 | Imbalance variants cb, la, rs on MobileNetV3-Small and ConvNeXt-Tiny |
| 8 or 16–17 | Learning-rate sensitivity |
| 18 | Plain cross-entropy control, all folds |
| 10 | Final analysis (CPU) |

## Verify without a GPU
```bash
python revision_analysis.py --out revision_results      # recomputes all paper tables from the shipped logits
python scripts/check_results.py                         # consistency checks of the original single-split results
```

## Archive
Create a GitHub release and archive it with Zenodo to obtain a citable DOI for the code and result files.
