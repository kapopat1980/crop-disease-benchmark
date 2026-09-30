# Data

## Source
The experiments use the public Kaggle dataset **"20k+ Multi-Class Crop Disease Images"**
(user `jawadali1045`): <https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images>

**The images are not redistributed in this repository.** Download them from Kaggle and respect the licence
stated on the dataset page (check it before any reuse; the dataset card gives little information on image
provenance or annotation procedure).

Expected layout after download/unzip (folder names = class labels):
```
<data_path>/
├── Train/<class name>/*.jpg          # original training folder
└── Validation/<class name>/*.jpg     # original validation folder
```
The script pools both folders, cleans them and creates its own stratified 70/15/15 split
(the dataset has no test partition).

## Cleaning (done by the script before splitting)
1. every file is hashed (MD5) and decoded; unreadable files are dropped;
2. identical bytes under **different** labels = label conflict → all copies removed;
3. identical bytes under the same label → one copy kept.

Result: **15,201 images, 44 classes** (12,170 from the original training folder, 3,031 from validation).
Only *exact* duplicates are detected; near-duplicates (re-encoded/cropped copies, several photos of one plant)
are not, and may cross splits — see the limitations in the paper.

## Split files shipped here (`results/splits/`)
| file | content |
|---|---|
| `file_index.csv` | all 15,201 retained images: `path` (Kaggle path used in the run), `rel_path` (relative to dataset root), `label`, `split` (original folder: `train` = `Train/`, `val` = `Validation/`), `md5` |
| `split_seed42.csv` | the exact train/val/test assignment used in the paper (`split`, `label_idx`) |
| `class_names.json` | class names in label-index order |

Use `rel_path` to map the files onto your own copy of the dataset.

## Known label-taxonomy issues (for anyone reusing the dataset)
* Capitalisation variants that probably denote the same class: `Wheat Brown leaf Rust` (90 images) vs `Wheat Brown leaf rust` (15); `Bacterial Blight in cotton` (255) vs `bacterial_blight in Cotton` (472).
* Spelling/style inconsistencies: `Becterial Blight in Rice`, `thirps on  cotton` (misspelling + double space), `Wheat___Yellow_Rust` (underscores), mixed naming conventions across classes.
* Extremely small classes (3–22 images: `bollrot on Cotton` 3, `bollworm on Cotton` 9, `Wheat Brown leaf rust` 15, `pink bollworm in cotton` 22) → per-class metrics on the test split rest on 1–5 images.
* At least one image carries an overlaid caption contradicting its label (see Grad-CAM figure in the paper).
Classes were used **as provided**; they were not merged.
