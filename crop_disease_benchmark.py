#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crop-disease image classification benchmark (journal-grade, reproducible).

Kaggle usage
------------
    !python crop_disease_benchmark.py --epochs 30            # single seed (42), 5 models
    # quick smoke test (~minutes):
    !python crop_disease_benchmark.py --epochs 2 --seeds 42 --models ConvNeXt-Tiny --max_per_class 40

What this script guarantees (each item fixes a problem in the earlier version_11.py)
-----------------------------------------------------------------------------------
 1. Separate Dataset objects per split -> augmentation only on train, never on val/test.
 2. Stratified train/val/test split (or the dataset's own split), saved to CSV.
 3. Exact-duplicate + corrupt-image detection (MD5) BEFORE splitting -> no train/test leakage.
 4. Model selection on the VALIDATION set only; test set is evaluated once per run.
 5. Same training recipe for every model (fair comparison), warmup + cosine LR, AMP,
    grad clipping, early stopping.
 6. Single seed by default (--seeds 42): uncertainty is reported with 95% bootstrap CIs over
    the test images, plus exact McNemar tests between models. Several seeds are still supported
    (--seeds 42 123 2024) and then give mean +/- std.
 7. Metrics that fail are reported as NaN with a warning (never silently 0.0):
    accuracy, balanced acc., macro/weighted P/R/F1, ROC-AUC (OvR), MCC, kappa, top-3, ECE, NLL.
 8. Efficiency: params, FLOPs (torch FlopCounterMode), latency, throughput, peak VRAM,
    measured per model on a clean GPU with inference_mode.
 9. Full-label confusion matrices, per-class F1, top confusions, learning curves,
    accuracy-vs-cost plot, correct Grad-CAM (tensor hooks, no in-place edits).
10. Resumable: finished (model, seed) runs are skipped on restart (Kaggle session limits).
11. Saves config, environment info, split files and predictions for reproducibility.

Note: cudnn.benchmark=True is used for speed, so runs are seed-controlled but not bit-wise
deterministic. Report this in the paper.
"""
from __future__ import annotations

import argparse
import glob
import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

ALL_MODELS = ["MobileNetV3-Small", "ResNet-50", "EfficientNetV2-S", "ConvNeXt-Tiny", "Swin-T"]


# =====================================================================================
# 0. ARGUMENTS / UTILITIES
# =====================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Crop disease benchmark")
    p.add_argument("--data_path", default="/kaggle/input/datasets/jawadali1045/20k-multi-class-crop-disease-images")
    p.add_argument("--output_dir", default="./journal_results")
    p.add_argument("--models", nargs="+", default=["all"], help=f"subset of {ALL_MODELS} or 'all'")
    p.add_argument("--focus_model", default="ConvNeXt-Tiny", help="model used for Grad-CAM / per-class plots")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="training seeds (default: one). More seeds -> mean +/- std in the tables")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--resplit_per_seed", action="store_true",
                   help="use a different stratified split for every seed (captures data variability)")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--img_size", type=int, default=288)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8, help="early stopping on val macro-F1")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--head_dropout", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--class_weights", action="store_true", help="effective-number class weights in the loss")
    p.add_argument("--no_dedupe", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_ckpt", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore cached indices / finished runs")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--gradcam_samples", type=int, default=8)
    p.add_argument("--max_per_class", type=int, default=0, help="debug: cap images per class (0 = all)")
    args, _ = p.parse_known_args()  # tolerant of Jupyter/Kaggle extra args
    if args.models == ["all"] or "all" in args.models:
        args.models = list(ALL_MODELS)
    lookup = {m.lower(): m for m in ALL_MODELS}
    fixed = []
    for m in args.models:
        if m.lower() not in lookup:
            raise ValueError(f"Unknown model '{m}'. Choose from {ALL_MODELS}")
        fixed.append(lookup[m.lower()])
    args.models = fixed
    if args.focus_model.lower() in lookup:
        args.focus_model = lookup[args.focus_model.lower()]
    return args


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def to_py(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=to_py)


def save_fig(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def environment_info(device):
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    try:
        import torchvision, sklearn
        info["torchvision"] = torchvision.__version__
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


# =====================================================================================
# 1. DATASET DISCOVERY, CLEANING, SPLITTING
# =====================================================================================
def find_dataset(args) -> Path:
    cands = [args.data_path, "/kaggle/input/20k-multi-class-crop-disease-images", "./data"]
    for c in cands:
        if os.path.isdir(c):
            return Path(c)
    hits = [h for h in glob.glob("/kaggle/input/**/*20k*", recursive=True) if os.path.isdir(h)]
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(
        f"Dataset not found at '{args.data_path}'. Attach it in Kaggle (Add Input) or pass --data_path.")


def _has_images(d: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in IMG_EXT for p in d.iterdir())
    except OSError:
        return False


def _is_class_root(d: Path) -> bool:
    try:
        subs = [s for s in d.iterdir() if s.is_dir()]
    except OSError:
        return False
    if len(subs) < 2:
        return False
    n = sum(_has_images(s) for s in subs)
    return n >= 2 and n >= 0.8 * len(subs)


def discover_layout(base: Path, max_depth: int = 4):
    """Returns ('predefined', {split: Path}) or ('flat', Path)."""
    queue = [(base, 0)]
    while queue:
        d, depth = queue.pop(0)
        subs = {s.name.lower(): s for s in sorted(d.iterdir()) if s.is_dir()}
        tr = subs.get("train")
        if tr is not None and _is_class_root(tr):
            va = subs.get("val") or subs.get("valid") or subs.get("validation")
            te = subs.get("test")
            return "predefined", {
                "train": tr,
                "val": va if (va is not None and _is_class_root(va)) else None,
                "test": te if (te is not None and _is_class_root(te)) else None,
            }
        if _is_class_root(d):
            return "flat", d
        if depth < max_depth:
            queue.extend((s, depth + 1) for s in sorted(d.iterdir()) if s.is_dir())
    raise RuntimeError(f"No class-folder structure found under {base}")


def _scan(root: Path, split: str):
    rows = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cdir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                rows.append((str(f), cdir.name, split))
    return rows


def _md5_and_check(path: str):
    try:
        with open(path, "rb") as f:
            data = f.read()
        h = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return h, True
    except Exception:
        return "", False


def build_index(args, out: Path):
    """Scan -> hash/verify -> remove corrupt & duplicate & label-conflicting images."""
    base = find_dataset(args)
    mode, layout = discover_layout(base)
    cache = out / "file_index.csv"
    meta_path = out / "dataset_meta.json"

    if cache.exists() and meta_path.exists() and not args.force:
        meta = json.load(open(meta_path))
        if (meta.get("base") == str(base) and meta.get("mode") == mode and meta.get("dedupe") == (not args.no_dedupe)
                and meta.get("max_per_class", 0) == args.max_per_class):
            print(f"[data] reusing cached index ({cache})")
            return pd.read_csv(cache), meta

    rows = []
    if mode == "predefined":
        for sp, root in layout.items():
            if root is not None:
                rows += _scan(root, sp)
    else:
        rows += _scan(layout, "all")
    df = pd.DataFrame(rows, columns=["path", "label", "split"])
    print(f"[data] layout={mode}; scanned {len(df):,} images in {df.label.nunique()} classes under {base}")

    hashes = [_md5_and_check(p) for p in tqdm(df.path, desc="hash+verify", leave=False)]
    df["md5"] = [h for h, _ in hashes]
    df["ok"] = [ok for _, ok in hashes]

    report = {"n_scanned": int(len(df))}
    report["corrupt_removed"] = int((~df.ok).sum())
    df = df[df.ok].copy()

    if not args.no_dedupe:
        # same bytes, different class -> label noise, drop all copies
        g = df.groupby("md5")["label"].nunique()
        conflict = set(g[g > 1].index)
        report["label_conflict_removed"] = int(df.md5.isin(conflict).sum())
        df = df[~df.md5.isin(conflict)]
        if mode == "predefined":
            cross = 0
            for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
                ha, hb = set(df[df.split == a].md5), set(df[df.split == b].md5)
                cross += len(ha & hb)
            report["cross_split_duplicate_hashes"] = int(cross)
            prio = {"test": 0, "val": 1, "train": 2}
            df = df.assign(_o=df.split.map(prio)).sort_values(["_o", "path"]).drop(columns="_o")
        before = len(df)
        df = df.drop_duplicates(subset="md5", keep="first")
        report["exact_duplicates_removed"] = int(before - len(df))
    df = df.drop(columns=["ok"]).reset_index(drop=True)

    if args.max_per_class > 0:
        df = df.groupby("label", group_keys=False).head(args.max_per_class).reset_index(drop=True)
        report["debug_max_per_class"] = args.max_per_class
    report["n_final"] = int(len(df))
    report["n_classes"] = int(df.label.nunique())
    meta = {"base": str(base), "mode": mode, "dedupe": not args.no_dedupe, "max_per_class": args.max_per_class,
            "cleaning_report": report}
    df.to_csv(cache, index=False)
    save_json(meta, meta_path)
    print("[data] cleaning report:", json.dumps(report))
    return df, meta


def make_splits(df: pd.DataFrame, meta: dict, args, seed: int):
    """Returns df with columns [path,label,label_idx,split] and the class-name list."""
    class_names = sorted(df.label.unique())
    cmap = {c: i for i, c in enumerate(class_names)}
    d = df.copy()
    d["label_idx"] = d.label.map(cmap).astype(int)

    if meta["mode"] == "predefined" and {"train", "val", "test"} <= set(d.split.unique()):
        print("[split] using the dataset's own train/val/test folders (after de-duplication)")
        return d[["path", "label", "label_idx", "split"]].reset_index(drop=True), class_names

    if meta["mode"] == "predefined":
        print("[split] dataset has missing val/test folders -> pooling and re-splitting (stratified)")
    counts = d.label.value_counts()
    if counts.min() < 3:
        raise ValueError(f"Class '{counts.idxmin()}' has only {counts.min()} images; need >=3 for stratified split.")
    idx, y = np.arange(len(d)), d.label_idx.values
    tr_val, te = train_test_split(idx, test_size=args.test_frac, stratify=y, random_state=seed)
    rel = args.val_frac / (1.0 - args.test_frac)
    tr, va = train_test_split(tr_val, test_size=rel, stratify=y[tr_val], random_state=seed)
    d["split"] = ""
    d.loc[d.index[tr], "split"] = "train"
    d.loc[d.index[va], "split"] = "val"
    d.loc[d.index[te], "split"] = "test"
    return d[["path", "label", "label_idx", "split"]].reset_index(drop=True), class_names


class ImgDS(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths, self.labels, self.tfm = list(paths), list(labels), tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.tfm(img), int(self.labels[i])


def get_transforms(S: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.6, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
        transforms.RandomErasing(p=0.25),
    ])
    evalt = transforms.Compose([
        transforms.Resize((S, S)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return train, evalt


def make_loaders(split_df, args, seed, device):
    tf_tr, tf_ev = get_transforms(args.img_size)
    dfs = {s: split_df[split_df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
              worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0)
    train = DataLoader(ImgDS(dfs["train"].path, dfs["train"].label_idx, tf_tr), batch_size=args.batch_size,
                       shuffle=True, drop_last=True, generator=g, **kw)
    val = DataLoader(ImgDS(dfs["val"].path, dfs["val"].label_idx, tf_ev), batch_size=args.batch_size * 2,
                     shuffle=False, **kw)
    test = DataLoader(ImgDS(dfs["test"].path, dfs["test"].label_idx, tf_ev), batch_size=args.batch_size * 2,
                      shuffle=False, **kw)
    return train, val, test, dfs


# =====================================================================================
# 2. MODEL FACTORY (identical head/optimiser policy for every model)
# =====================================================================================
def build_model(name: str, num_classes: int, pretrained: bool, head_dropout: float):
    """Returns (model, head_module, gradcam_target_layer, target_is_channels_last)."""
    if name == "MobileNetV3-Small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)  # native dropout kept
        return m, m.classifier[3], m.features[-1], False
    if name == "ResNet-50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(m.fc.in_features, num_classes))
        return m, m.fc, m.layer4[-1], False
    if name == "EfficientNetV2-S":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)  # native dropout kept
        return m, m.classifier[1], m.features[-1], False
    if name == "ConvNeXt-Tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2] = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(m.classifier[2].in_features, num_classes))
        return m, m.classifier[2], m.features[-1], False
    if name == "Swin-T":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        m.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(m.head.in_features, num_classes))
        return m, m.head, m.features[-1], True  # Swin blocks output B,H,W,C
    raise ValueError(name)


# =====================================================================================
# 3. EFFICIENCY PROFILING (clean, per model, inference mode)
# =====================================================================================
def _time_ms(fn, device, warmup, runs):
    if device.type == "cuda":
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(runs):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / runs
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / runs


def profile_efficiency(name, args, num_classes, device):
    from torch.utils.flop_counter import FlopCounterMode
    model, _, _, _ = build_model(name, num_classes, pretrained=False, head_dropout=args.head_dropout)
    model = model.to(device).eval()
    S = args.img_size
    total = sum(p.numel() for p in model.parameters())
    x1 = torch.randn(1, 3, S, S, device=device)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            model(x1)
        flops = fc.get_total_flops()
        out = {"Model": name, "Params (M)": total / 1e6, "Size (MB, fp32)": total * 4 / 2 ** 20,
               "FLOPs (G)": flops / 1e9, "MACs (G)": flops / 2e9}
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out["Latency bs=1 (ms)"] = _time_ms(lambda: model(x1), device, 20 if device.type == "cuda" else 2,
                                                100 if device.type == "cuda" else 8)
            if device.type == "cuda":
                bs = 32
                xb = torch.randn(bs, 3, S, S, device=device)
                ms = _time_ms(lambda: model(xb), device, 5, 20)
                out["Throughput bs=32 (img/s)"] = bs * 1000.0 / ms
                out["Peak VRAM inference bs=32 (MB)"] = torch.cuda.max_memory_allocated() / 2 ** 20
                del xb
            else:
                out["Throughput bs=32 (img/s)"] = float("nan")
                out["Peak VRAM inference bs=32 (MB)"] = float("nan")
    out["Device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    del model, x1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# =====================================================================================
# 4. METRICS
# =====================================================================================
def expected_calibration_error(probs, y_true, n_bins=15):
    conf = probs.max(1)
    acc = (probs.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true, y_pred, probs, C):
    m = {"accuracy": float((y_true == y_pred).mean()),
         "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    m.update(precision_macro=float(p), recall_macro=float(r), f1_macro=float(f))
    m["f1_weighted"] = float(precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)[2])
    m["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    m["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    m["ece"] = expected_calibration_error(probs, y_true)

    present = np.unique(y_true)
    m["roc_auc_macro"] = float("nan")
    if len(present) >= 2:
        try:
            sub = probs[:, present]
            sub = sub / sub.sum(1, keepdims=True).clip(1e-12)
            m["roc_auc_macro"] = float(roc_auc_score(y_true, sub, multi_class="ovr", average="macro", labels=present))
        except Exception as e:  # never fail silently
            warnings.warn(f"ROC-AUC could not be computed: {e}")
    m["top3_acc"] = float("nan")
    if C > 3:
        try:
            m["top3_acc"] = float(top_k_accuracy_score(y_true, probs, k=3, labels=np.arange(C)))
        except Exception as e:
            warnings.warn(f"Top-3 accuracy could not be computed: {e}")
    if len(present) < C:
        warnings.warn(f"{C - len(present)} class(es) absent from this evaluation split.")
    return m


def bootstrap_ci(y_true, y_pred, B, seed=0):
    """95% percentile bootstrap CIs (resampling test images) for the headline metrics."""
    if B <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, prec, rec, f1 = [], [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        p, r, f, _ = precision_recall_fscore_support(y_true[i], y_pred[i], average="macro", zero_division=0)
        acc.append((y_true[i] == y_pred[i]).mean())
        prec.append(p)
        rec.append(r)
        f1.append(f)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy_ci95": q(acc), "precision_macro_ci95": q(prec),
            "recall_macro_ci95": q(rec), "f1_macro_ci95": q(f1)}


@torch.inference_mode()
def predict(model, loader, device, use_amp):
    model.eval()
    nll, n = 0.0, 0
    P, Y = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        nll += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        P.append(F.softmax(logits, 1).cpu())
        Y.append(y.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), nll / max(n, 1)


# =====================================================================================
# 5. TRAINING (one model, one seed)
# =====================================================================================
def effective_number_weights(labels, C, beta=0.999):
    counts = np.bincount(labels, minlength=C).astype(float)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.clip(eff, 1e-12, None)
    return torch.tensor(w / w.sum() * C, dtype=torch.float32)


def run_one(name, seed, loaders, dfs, class_names, args, device, run_dir: Path):
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        print(f"  [skip] {name} seed {seed}: result found")
        return json.load(open(result_path))
    run_dir.mkdir(parents=True, exist_ok=True)

    C = len(class_names)
    train_loader, val_loader, test_loader = loaders
    set_seed(seed)
    model, head, _, _ = build_model(name, C, not args.no_pretrained, args.head_dropout)
    model = model.to(device)
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr_backbone}, {"params": list(head.parameters()), "lr": args.lr_head}],
        weight_decay=args.weight_decay)

    spe = len(train_loader)
    total_steps, warm = args.epochs * spe, max(1, args.warmup_epochs * spe)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weight = None
    if args.class_weights:
        weight = effective_number_weights(dfs["train"].label_idx.values, C).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    history, best = [], {"f1": -1.0, "loss": 1e9, "epoch": 0, "state": None}
    bad_epochs, t_start = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()
            loss_sum += loss.detach() * y.size(0)
            correct += (out.argmax(1) == y).sum()
            seen += y.size(0)

        vp, vy, vloss = predict(model, val_loader, device, use_amp)
        vm = compute_metrics(vy, vp.argmax(1), vp, C)
        row = {"epoch": epoch, "lr_backbone": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum.item() / seen, "train_acc": correct.item() / seen,
               "val_loss": vloss, "val_acc": vm["accuracy"], "val_f1_macro": vm["f1_macro"],
               "epoch_time_s": time.time() - t0}
        history.append(row)
        improved = (vm["f1_macro"] > best["f1"] + 1e-6) or (abs(vm["f1_macro"] - best["f1"]) <= 1e-6 and vloss < best["loss"])
        if improved:
            best.update(f1=vm["f1_macro"], loss=vloss, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"  [{name} s{seed}] ep {epoch:02d}/{args.epochs} trL {row['train_loss']:.4f} "
              f"vaL {vloss:.4f} vaAcc {vm['accuracy']*100:.2f} vaF1 {vm['f1_macro']*100:.2f} "
              f"({row['epoch_time_s']:.0f}s){' *' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"  early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break
    train_time = time.time() - t_start

    model.load_state_dict(best["state"])
    if not args.no_ckpt:
        torch.save(best["state"], run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    vp, vy, vloss = predict(model, val_loader, device, use_amp)
    tp, ty, tloss = predict(model, test_loader, device, use_amp)  # test set touched ONCE, after selection
    val_m = compute_metrics(vy, vp.argmax(1), vp, C)
    test_m = compute_metrics(ty, tp.argmax(1), tp, C)
    val_m["loss"], test_m["loss"] = vloss, tloss
    ci = bootstrap_ci(ty, tp.argmax(1), args.bootstrap)
    np.savez_compressed(run_dir / "test_predictions.npz", y_true=ty, y_pred=tp.argmax(1),
                        probs=tp.astype(np.float16))

    res = {"model": name, "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history),
           "train_time_s": train_time, "n_train": len(dfs["train"]), "n_val": len(dfs["val"]),
           "n_test": len(dfs["test"]), "val": val_m, "test": test_m, "test_ci": ci}
    save_json(res, result_path)
    print(f"  => TEST {name} s{seed}: acc {test_m['accuracy']*100:.2f} | macro-F1 {test_m['f1_macro']*100:.2f} "
          f"| AUC {test_m['roc_auc_macro']*100:.2f} | ECE {test_m['ece']*100:.2f}")
    del model, optimizer, scaler, best
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


# =====================================================================================
# 6. GRAD-CAM (tensor hooks; no in-place edits; works for CNNs and Swin)
# =====================================================================================
class GradCAM:
    def __init__(self, model, target_layer, channels_last=False):
        self.model, self.cl = model, channels_last
        self.acts, self.grads = None, None
        self.h = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self.h.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(1)
            logits.gather(1, class_idx.view(-1, 1)).sum().backward()
        a, g = self.acts, self.grads
        if self.cl:
            a, g = a.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1))
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return cam.detach().cpu().numpy(), logits.detach()


def run_gradcam(name, seed, split_df, class_names, args, device, out: Path):
    run_dir = out / "runs" / slug(name) / f"seed{seed}"
    ckpt, pred_file = run_dir / "best.pt", run_dir / "test_predictions.npz"
    if not (ckpt.exists() and pred_file.exists()):
        print("[gradcam] checkpoint/predictions missing -> skipped")
        return
    C = len(class_names)
    model, _, layer, cl = build_model(name, C, pretrained=False, head_dropout=args.head_dropout)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()
    cam_engine = GradCAM(model, layer, cl)

    P = np.load(pred_file)
    y_true, y_pred, probs = P["y_true"], P["y_pred"], P["probs"].astype(np.float32)
    test_df = split_df[split_df.split == "test"].reset_index(drop=True)
    conf = probs.max(1)
    rng = np.random.default_rng(0)
    N = args.gradcam_samples

    correct_idx = rng.permutation(np.where(y_true == y_pred)[0])
    seen, chosen_c = set(), []
    for i in correct_idx:  # prefer distinct classes
        if y_true[i] not in seen:
            seen.add(y_true[i])
            chosen_c.append(i)
        if len(chosen_c) == N:
            break
    wrong_idx = np.where(y_true != y_pred)[0]
    chosen_w = list(wrong_idx[np.argsort(-conf[wrong_idx])][:N])  # most confident mistakes

    _, tf_ev = get_transforms(args.img_size)
    mean, std = np.array(NORM_MEAN), np.array(NORM_STD)
    cmap = plt.get_cmap("jet")

    def render(indices):
        imgs, overlays, titles = [], [], []
        for i in indices:
            with Image.open(test_df.path[i]) as im:
                x = tf_ev(im.convert("RGB")).unsqueeze(0).to(device)
            cam, logits = cam_engine(x)
            rgb = np.clip(x[0].detach().cpu().permute(1, 2, 0).numpy() * std + mean, 0, 1)
            heat = cmap(cam[0])[..., :3]
            imgs.append(rgb)
            overlays.append(np.clip(0.55 * rgb + 0.45 * heat, 0, 1))
            titles.append(textwrap.fill(f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]} ({conf[i]:.2f})", 22))
        return imgs, overlays, titles

    groups = [("Correct", chosen_c), ("Misclassified (most confident)", chosen_w)]
    groups = [(t, ix) for t, ix in groups if len(ix) > 0]
    if not groups:
        return
    ncol = max(len(ix) for _, ix in groups)
    fig, axes = plt.subplots(2 * len(groups), ncol, figsize=(2.3 * ncol, 2.9 * 2 * len(groups)), squeeze=False)
    for g, (title, ix) in enumerate(groups):
        imgs, overlays, titles = render(ix)
        for c in range(ncol):
            for r in (0, 1):
                axes[2 * g + r, c].axis("off")
        for c in range(len(ix)):
            axes[2 * g, c].imshow(imgs[c])
            axes[2 * g, c].set_title(titles[c], fontsize=6.5)
            axes[2 * g + 1, c].imshow(overlays[c])
        axes[2 * g, 0].text(-0.05, 1.28, title, transform=axes[2 * g, 0].transAxes, fontsize=10, fontweight="bold")
        axes[2 * g + 1, 0].text(-0.05, -0.08, "Grad-CAM", transform=axes[2 * g + 1, 0].transAxes, fontsize=8, va="top")
    fig.suptitle(f"Grad-CAM - {name} (seed {seed}, test set)", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, out / "figures" / f"gradcam_{slug(name)}")
    cam_engine.remove()


# =====================================================================================
# 7. AGGREGATION, TABLES, FIGURES
# =====================================================================================
METRIC_COLS = [  # key, header, scale, higher_is_better
    ("accuracy", "Accuracy (%)", 100, True), ("balanced_accuracy", "Balanced Acc. (%)", 100, True),
    ("precision_macro", "Precision (%)", 100, True), ("recall_macro", "Recall (%)", 100, True),
    ("f1_macro", "Macro-F1 (%)", 100, True), ("f1_weighted", "Weighted-F1 (%)", 100, True),
    ("roc_auc_macro", "ROC-AUC (%)", 100, True), ("mcc", "MCC", 1, True), ("kappa", "Kappa", 1, True),
    ("top3_acc", "Top-3 Acc. (%)", 100, True), ("ece", "ECE (%)", 100, False), ("loss", "NLL", 1, False),
]


def _fmt(mean, std, scale, n):
    if not np.isfinite(mean):
        return "--"
    if n > 1 and np.isfinite(std):
        return f"{mean*scale:.2f} ± {std*scale:.2f}"
    return f"{mean*scale:.2f}"


def _tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def latex_table(df_str, bold_mask, caption, label):
    cols = list(df_str.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for i in range(len(df_str)):
        cells = []
        for j, c in enumerate(cols):
            v = str(df_str.iloc[i, j]).replace("±", r"$\pm$")
            v = _tex_escape(v).replace(r"$\pm$", r"$\pm$") if j == 0 else v
            if j > 0 and bold_mask.iloc[i, j]:
                v = r"\textbf{" + v + "}"
            cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def plot_confusion(cm, class_names, title, stem: Path):
    C = len(class_names)
    mat = cm.astype(float)
    mat = mat / mat.sum(1, keepdims=True).clip(1)
    side = max(7, 0.30 * C)
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    sns.heatmap(mat, cmap="Blues", vmin=0, vmax=1, square=True, ax=ax, linewidths=0.1, linecolor="#eee",
                xticklabels=class_names, yticklabels=class_names, annot=(C <= 20), fmt=".2f",
                cbar_kws={"shrink": 0.6, "label": "Row-normalised rate"})
    fs = max(4.5, 9 - C / 8)
    ax.tick_params(axis="both", labelsize=fs)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    save_fig(fig, stem)


def aggregate(args, out: Path, class_names, split_dfs_by_seed, eff_df):
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    C = len(class_names)

    results = []
    for f in sorted(glob.glob(str(out / "runs" / "*" / "seed*" / "result.json"))):
        results.append(json.load(open(f)))
    results = [r for r in results if r["model"] in args.models]
    if not results:
        print("[aggregate] no finished runs")
        return
    rows = []
    for r in results:
        row = {"Model": r["model"], "Seed": r["seed"], "Best epoch": r["best_epoch"],
               "Train time (min)": r["train_time_s"] / 60}
        row.update({f"test_{k}": v for k, v in r["test"].items()})
        row.update({f"val_{k}": v for k, v in r["val"].items()})
        for ck, cv in r.get("test_ci", {}).items():
            row[f"test_{ck}"] = str(cv)
        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(tables / "all_runs.csv", index=False)

    order = [m for m in ALL_MODELS if m in runs.Model.unique()]
    n_seeds = runs.groupby("Model").Seed.nunique()
    single = bool((n_seeds <= 1).all())
    ci_lookup = {(r["model"], k): v for r in results for k, v in r.get("test_ci", {}).items()}

    def cell(m, key, mu, sd, scale):
        n = n_seeds.get(m, 1)
        if n > 1:
            return _fmt(mu, sd, scale, n)
        if not np.isfinite(mu):
            return "--"
        ci = ci_lookup.get((m, f"{key}_ci95"))
        if ci:
            return f"{mu*scale:.2f} [{ci[0]*scale:.2f}, {ci[1]*scale:.2f}]"
        return f"{mu*scale:.2f}"

    # ---- mean +/- std table
    mean_df = pd.DataFrame(index=order)
    str_df = pd.DataFrame(index=order)
    for key, head, scale, _ in METRIC_COLS:
        g = runs.groupby("Model")[f"test_{key}"]
        mu, sd = g.mean(), g.std(ddof=1)
        mean_df[head] = [mu.get(m, np.nan) * scale for m in order]
        str_df[head] = [cell(m, key, mu.get(m, np.nan), sd.get(m, np.nan), scale) for m in order]
    tm = runs.groupby("Model")["Train time (min)"].mean()
    str_df["Train time (min)"] = [f"{tm.get(m, np.nan):.1f}" for m in order]
    mean_df["Train time (min)"] = [tm.get(m, np.nan) for m in order]
    str_df.insert(0, "Model", order)
    mean_df.insert(0, "Model", order)
    str_df.to_csv(tables / "summary_mean_std.csv", index=False)

    # ---- efficiency merge + LaTeX
    eff = eff_df.set_index("Model") if eff_df is not None and len(eff_df) else None
    main_cols = ["Accuracy (%)", "Precision (%)", "Recall (%)", "Macro-F1 (%)", "ROC-AUC (%)"]
    main = str_df[["Model"] + main_cols].copy().reset_index(drop=True)
    mask = pd.DataFrame(False, index=main.index, columns=main.columns)
    for c in main_cols:
        vals = pd.to_numeric(mean_df[c].reset_index(drop=True), errors="coerce")
        if vals.notna().any():
            mask.loc[vals.idxmax(), c] = True
    if eff is not None:
        for c, nd in [("Params (M)", 2), ("FLOPs (G)", 2), ("Latency bs=1 (ms)", 2)]:
            main[c] = [f"{eff.loc[m, c]:.{nd}f}" if m in eff.index and np.isfinite(eff.loc[m, c]) else "--" for m in order]
            mask[c] = False
    (tables / "main_table.tex").write_text(latex_table(
        main, mask, "Test-set performance (" + (
            f"single run, seed {min(args.seeds)}; brackets: 95\\% bootstrap CI over test images"
            if single else "mean $\\pm$ std over seeds") + ") and computational cost.", "tab:main"))
    main.to_csv(tables / "main_table.csv", index=False)

    extra_cols = ["Balanced Acc. (%)", "Weighted-F1 (%)", "MCC", "Kappa", "Top-3 Acc. (%)", "ECE (%)", "NLL"]
    ex = str_df[["Model"] + extra_cols].copy().reset_index(drop=True)
    exm = pd.DataFrame(False, index=range(len(ex)), columns=ex.columns)
    for _, head, _, hib in METRIC_COLS:
        if head in extra_cols:
            vals = pd.to_numeric(mean_df[head].reset_index(drop=True), errors="coerce")
            if vals.notna().any():
                exm.loc[vals.idxmax() if hib else vals.idxmin(), head] = True
    (tables / "extended_table.tex").write_text(latex_table(
        ex, exm, "Additional test-set metrics (" + (
            f"single run, seed {min(args.seeds)}; point estimates" if single else "mean $\\pm$ std over seeds") + ").",
        "tab:extended"))

    if eff is not None:
        eff.reset_index().to_csv(tables / "efficiency.csv", index=False)

    # ---- model ranking by VALIDATION macro-F1 (not test) -> reference for significance tests
    val_rank = runs.groupby("Model")["val_f1_macro"].mean().reindex(order).sort_values(ascending=False)
    best_model = val_rank.index[0]

    # ---- per-class F1 (mean over seeds), confusion matrices, McNemar
    per_class = {}
    first_seed = min(args.seeds)
    preds_first = {}
    for m in order:
        f1s, cms = [], []
        for f in sorted(glob.glob(str(out / "runs" / slug(m) / "seed*" / "test_predictions.npz"))):
            P = np.load(f)
            yt, yp = P["y_true"], P["y_pred"]
            f1s.append(f1_score(yt, yp, labels=np.arange(C), average=None, zero_division=0))
            cms.append(confusion_matrix(yt, yp, labels=np.arange(C)))
            if f.endswith(f"seed{first_seed}{os.sep}test_predictions.npz"):
                preds_first[m] = (yt, yp)
        if not f1s:
            continue
        per_class[m] = np.mean(f1s, axis=0)
        cm_sum = np.sum(cms, axis=0)  # pooled over seeds
        pd.DataFrame(cm_sum, index=class_names, columns=class_names).to_csv(tables / f"confusion_{slug(m)}.csv")
        plot_confusion(cm_sum, class_names, f"Confusion matrix - {m} (pooled over {len(cms)} seed(s))",
                       figs / f"confusion_{slug(m)}")
        off = cm_sum.copy()
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:20], off.shape))[0]
        pd.DataFrame([{"true": class_names[i], "predicted": class_names[j], "count": int(off[i, j]),
                       "share_of_true_class_%": 100 * off[i, j] / max(cm_sum[i].sum(), 1)}
                      for i, j in top if off[i, j] > 0]).to_csv(tables / f"top_confusions_{slug(m)}.csv", index=False)
    if per_class:
        pc = pd.DataFrame(per_class, index=class_names)
        pc.to_csv(tables / "per_class_f1_all_models.csv")
        fm = args.focus_model if args.focus_model in pc.columns else pc.columns[0]
        s = pc[fm].sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * C)))
        ax.barh(s.index, s.values * 100, color=plt.cm.RdYlGn(s.values))
        ax.set_xlabel("F1-score (%)")
        ax.set_title(f"Per-class F1 - {fm} (mean over seeds)", fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlim(0, 100)
        save_fig(fig, figs / f"per_class_f1_{slug(fm)}")

    if best_model in preds_first:
        yt_b, yp_b = preds_first[best_model]
        mc = []
        for m in order:
            if m == best_model or m not in preds_first:
                continue
            yt_o, yp_o = preds_first[m]
            if len(yt_o) != len(yt_b) or not np.array_equal(yt_o, yt_b):
                continue
            cb, co = yp_b == yt_b, yp_o == yt_o
            b, c = int((cb & ~co).sum()), int((~cb & co).sum())
            p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
            mc.append({"reference (best val F1)": best_model, "compared with": m, "seed": first_seed,
                       "ref right / other wrong": b, "ref wrong / other right": c, "p (exact McNemar)": p,
                       "p Bonferroni": min(1.0, p * max(1, len(order) - 1))})
        if mc:
            pd.DataFrame(mc).to_csv(tables / "mcnemar.csv", index=False)

    # ---- learning curves
    hist = []
    for f in sorted(glob.glob(str(out / "runs" / "*" / "seed*" / "history.csv"))):
        h = pd.read_csv(f)
        h["Model"] = Path(f).parents[1].name
        hist.append(h)
    if hist:
        H = pd.concat(hist)
        name_by_slug = {slug(m): m for m in ALL_MODELS}
        H["Model"] = H.Model.map(name_by_slug).fillna(H.Model)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for m in order:
            hm = H[H.Model == m].groupby("epoch").mean(numeric_only=True)
            if hm.empty:
                continue
            axes[0].plot(hm.index, hm.train_loss, label=m)
            axes[1].plot(hm.index, hm.val_loss, label=m)
            axes[2].plot(hm.index, hm.val_f1_macro * 100, label=m)
        for ax, t, yl in zip(axes, ["Training loss", "Validation NLL", "Validation macro-F1"],
                             ["Loss (label-smoothed CE)", "NLL", "Macro-F1 (%)"]):
            ax.set_title(t, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, figs / "learning_curves")

    # ---- accuracy vs cost
    if eff is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, xc in zip(axes, ["FLOPs (G)", "Latency bs=1 (ms)"]):
            for m in order:
                if m not in eff.index or not np.isfinite(eff.loc[m, xc]):
                    continue
                y = mean_df.set_index("Model").loc[m, "Macro-F1 (%)"]
                ax.scatter(eff.loc[m, xc], y, s=30 + 12 * eff.loc[m, "Params (M)"], alpha=0.7, label=m)
                ax.annotate(m, (eff.loc[m, xc], y), fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel(xc)
            ax.set_ylabel("Test macro-F1 (%)")
            ax.grid(alpha=0.3)
        axes[0].set_title("Accuracy-cost trade-off (bubble = #params)", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, figs / "tradeoff")

    print("\n" + "=" * 90)
    print("TEST RESULTS (single run; [95% bootstrap CI over test images])" if single
          else "TEST RESULTS (mean ± std over seeds)")
    print("=" * 90)
    print(str_df[["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "Macro-F1 (%)", "ROC-AUC (%)"]].to_string(index=False))
    print(f"\nBest model by mean VALIDATION macro-F1: {best_model}")
    print(f"Tables -> {tables}\nFigures -> {figs}")


# =====================================================================================
# 8. MAIN
# =====================================================================================
def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"--- device: {device} | models: {args.models} | seeds: {args.seeds} ---")
    if len(args.seeds) == 1:
        print(f"[note] single seed ({args.seeds[0]}): report bootstrap CIs / McNemar tests, and state in the paper "
              "that results come from one training run (no seed-to-seed variance).")
    save_json(vars(args), out / "config.json")
    save_json(environment_info(device), out / "environment.json")

    df, meta = build_index(args, out)
    first_split, class_names = make_splits(df, meta, args, args.split_seed)
    C = len(class_names)
    save_json(class_names, out / "class_names.json")

    # dataset summary (table + figure)
    summ = first_split.pivot_table(index="label", columns="split", values="path", aggfunc="count", fill_value=0)
    summ = summ[[c for c in ["train", "val", "test"] if c in summ.columns]]
    summ["total"] = summ.sum(1)
    summ.to_csv(out / "dataset_summary.csv")
    print(f"[data] {C} classes | train/val/test = "
          f"{(first_split.split=='train').sum():,}/{(first_split.split=='val').sum():,}/{(first_split.split=='test').sum():,} "
          f"| imbalance ratio (max/min class) = {summ.total.max()/summ.total.min():.1f}")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.24 * C)))
    summ.drop(columns="total").sort_values(summ.columns[0]).plot.barh(stacked=True, ax=ax, width=0.85)
    ax.set_xlabel("Images")
    ax.set_title("Class distribution per split", fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    save_fig(fig, out / "figures" / "class_distribution")

    # efficiency (once per model, before any training, on a clean device)
    eff_path = out / "tables" / "efficiency.csv"
    (out / "tables").mkdir(exist_ok=True)
    if eff_path.exists() and not args.force:
        eff_df = pd.read_csv(eff_path)
        eff_df = eff_df[eff_df.Model.isin(args.models)]
        missing = [m for m in args.models if m not in set(eff_df.Model)]
    else:
        eff_df, missing = pd.DataFrame(), list(args.models)
    new_rows = []
    for m in missing:
        print(f"[profile] {m}")
        new_rows.append(profile_efficiency(m, args, C, device))
    if new_rows:
        eff_df = pd.concat([eff_df, pd.DataFrame(new_rows)], ignore_index=True)
        eff_df.to_csv(eff_path, index=False)

    # training
    split_dfs = {}
    for seed in args.seeds:
        sseed = seed if args.resplit_per_seed else args.split_seed
        split_df, cn = (first_split, class_names) if sseed == args.split_seed else make_splits(df, meta, args, sseed)
        assert cn == class_names
        split_df.to_csv(out / f"split_seed{sseed}.csv", index=False)
        split_dfs[seed] = split_df
        train_l, val_l, test_l, dfs = make_loaders(split_df, args, seed, device)
        for name in args.models:
            print(f"\n{'='*70}\n {name} | seed {seed}\n{'='*70}")
            run_one(name, seed, (train_l, val_l, test_l), dfs, class_names, args, device,
                    out / "runs" / slug(name) / f"seed{seed}")
        del train_l, val_l, test_l
        gc.collect()

    aggregate(args, out, class_names, split_dfs, eff_df)

    fm = args.focus_model if args.focus_model in args.models else args.models[0]
    seed0 = min(args.seeds)
    run_gradcam(fm, seed0, split_dfs[seed0], class_names, args, device, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
