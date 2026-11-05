# Data

## Source
The experiments use the public Kaggle dataset **"20k+ Multi-Class Crop Disease Images"** (user `jawadali1045`):
<https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images> (accessed 1 November 2026).

**The images are not redistributed in this repository.** Download them from Kaggle and respect the licence stated on the
dataset page. The dataset card gives little information on image provenance or annotation; labels are unverified.

Expected layout (folder names = class labels):
```
<data_path>/
├── Train/<class name>/*.jpg
└── Validation/<class name>/*.jpg
```

## Cleaning (audit stage, `revision_experiments.py --stage audit`)
1. Every file is hashed (MD5) and decoded; unreadable files are dropped (none were found).
2. Byte-identical images under **different** labels = label conflict → all copies removed (56 files).
3. Byte-identical images under the same label → one copy kept (3,362 removed).

Result: **15,201 images** from 18,619 image files scanned. The per-file log with the reason for every removal is
`revision_results/audit/cleaning_log.csv`; per-class counts are in `cleaning_per_class.csv`. 53 of the 56 conflicting
copies involve the three bollworm classes (the same caterpillar photographs filed as American, pink and generic bollworm).

## Corrected taxonomy (44 → 42 classes)
* Merged: `Wheat Brown leaf Rust` + `Wheat Brown leaf rust` → wheat brown (leaf) rust;
  `Bacterial Blight in cotton` + `bacterial_blight in Cotton` → cotton bacterial blight.
* Host crop assigned to every class (`revision_results/audit/taxonomy.csv`). `Wilt` and `Leaf Curl` were assigned to cotton
  by inspection; `Army worm` is crop-unassigned (larvae on broad-leaved and grass hosts).
* Spelling variants (`Becterial`, `thirps`, `Wheat___Yellow_Rust`) are kept as folder names and normalised only for display.

## Near-duplicates and groups
* Detectors: 64-bit pHash (Hamming distance ≤ 6) and cosine similarity ≥ 0.95 of ImageNet ResNet-50 embeddings.
* 6,892 pHash pairs, 13,093 embedding pairs (6,357 by both); 7,123 images (47%) in 2,134 groups; largest group 65 images.
* 68 pairs carry different labels (`revision_results/audit/near_duplicate_pairs.csv`, Supplementary Table S2).
* 664 files are pre-augmented copies (names starting `contrast_`, `crop_`, `flip_`, `rotation_`, `rescale_`, `zoom_`).
* Group ids are in `revision_results/audit/master_index.csv` (`group` column); the cross-validation folds keep each group
  in a single fold (`revision_results/cv_corrected/folds.csv`).

## Original single split (first version of the study)
`results/splits/split_seed42.csv` is the stratified 70/15/15 split used in the first version. 1,009 of its 2,281 test
images have a near-duplicate in the training or validation part
(`revision_results/audit/orig_test_images_with_neardup.csv`); use it only to reproduce the leakage analysis.
