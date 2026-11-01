#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Revision experiments for the crop disease/pest benchmark (resubmission after The Visual Computer review).

Place this file next to crop_disease_benchmark.py (it re-uses its data scan, model factory,
transforms, metrics and Grad-CAM) and run the stages in order on Kaggle (Tesla T4).

Stages
------
  audit    (GPU optional, ~15 min)  cleaning audit per class and per reason, corrected label taxonomy,
                                    near-duplicate detection (pHash + ImageNet embeddings), duplicate
                                    groups, cross-split leakage check of the ORIGINAL seed-42 split,
                                    a montage of near-duplicate pairs for visual verification.
  cv       (GPU)                    5-fold stratified *group* cross-validation (near-duplicate groups never
                                    cross folds) on the corrected taxonomy, for any (model, variant).
                                    Saves val/test logits for every fold -> pooled out-of-fold evaluation.
  hpsweep  (GPU)                    learning-rate sensitivity per architecture on fold 0 (val macro-F1 only).
  explain  (GPU, ~10 min)           Grad-CAM for ALL models on the SAME fold-0 images: best cases (most
                                    confident correct) and worst cases (most confident errors), plus the
                                    deletion-AUC faithfulness score on a fixed random sample.

Training variants (--variants)
------------------------------
  ce_ls  cross-entropy + label smoothing 0.1          (= original recipe, the reference)
  ce     plain cross-entropy                          (isolates the effect of label smoothing on calibration)
  cb     class-balanced cross-entropy (effective number of samples, beta = 0.999) [Cui et al. 2019]
  focal  focal loss, gamma = 2 [Lin et al. 2017]
  la     logit-adjusted cross-entropy, tau = 1 [Menon et al. 2021]
  rs     square-root-frequency re-sampling of the training set (WeightedRandomSampler)

Suggested Kaggle commands (each block fits one 12-h session; runs are resumable)
------------------------------------------------------------------------------
  !python revision_experiments.py --stage audit
  !python revision_experiments.py --stage cv --models all --variants ce_ls                       # ~21 h, 2 sessions
  !python revision_experiments.py --stage cv --models MobileNetV3-Small ConvNeXt-Tiny \\
                                  --variants ce cb focal la rs                                    # ~35 h, 3-4 sessions
  !python revision_experiments.py --stage hpsweep --models all --epochs 20                         # ~9 h
  !python revision_experiments.py --stage explain --models all
Then, on any CPU:  python revision_analysis.py --out ./revision_results
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_disease_benchmark as cdb  # noqa: E402

ALL_MODELS = cdb.ALL_MODELS
VARIANTS = ["ce_ls", "ce", "cb", "focal", "la", "rs"]

# -------------------------------------------------------------------------------------------------
# Corrected label taxonomy.
#   * merges: capitalisation / naming variants that denote the same disease
#   * crop: host crop for every class. Entries marked VERIFY were assigned from the dataset's folder
#     structure and confusion patterns; confirm them by inspecting the images before submission.
# -------------------------------------------------------------------------------------------------
MERGE = {
    "Wheat Brown leaf rust": "Wheat Brown leaf Rust",
    "bacterial_blight in Cotton": "Bacterial Blight in cotton",
}
RENAME = {  # cosmetic, for tables/figures only (does not change the task)
    "Wheat Brown leaf Rust": "Wheat brown (leaf) rust",
    "Bacterial Blight in cotton": "Cotton bacterial blight",
    "Becterial Blight in Rice": "Rice bacterial blight",
    "thirps on  cotton": "Cotton thrips",
    "Wheat___Yellow_Rust": "Wheat yellow rust",
    "Common_Rust": "Maize common rust",
    "Gray_Leaf_Spot": "Maize gray leaf spot",
    "Brownspot": "Rice brown spot",
    "Tungro": "Rice tungro",
    "Leaf smut": "Rice leaf smut",
    "Flag Smut": "Wheat flag smut",
    "Wilt": "Cotton wilt",
    "Leaf Curl": "Cotton leaf curl",
    "Army worm": "Armyworm",
}
CROP = {
    "Brownspot": "Rice", "Tungro": "Rice", "Leaf smut": "Rice", "Becterial Blight in Rice": "Rice", "Rice Blast": "Rice",
    "Common_Rust": "Maize", "Gray_Leaf_Spot": "Maize", "Healthy Maize": "Maize",
    "Flag Smut": "Wheat",
    "Wilt": "Cotton",        # VERIFY
    "Leaf Curl": "Cotton",   # VERIFY
    "Army worm": "Any",      # VERIFY: folder does not name a crop; "Any" = allowed under every crop mask
}


def crop_of(label: str) -> str:
    if label in CROP:
        return CROP[label]
    low = label.lower()
    for k in ["cotton", "wheat", "maize", "sugarcane", "rice"]:
        if k in low:
            return k.capitalize()
    return "Any"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["audit", "cv", "hpsweep", "explain"])
    p.add_argument("--data_path", default="/kaggle/input/datasets/jawadali1045/20k-multi-class-crop-disease-images")
    p.add_argument("--output_dir", default="./revision_results")
    p.add_argument("--orig_split", default="results/splits/split_seed42.csv",
                   help="split file of the original submission (for the leakage check)")
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--variants", nargs="+", default=["ce_ls"])
    p.add_argument("--taxonomy", choices=["corrected", "original"], default="corrected")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold_ids", nargs="+", type=int, default=None, help="subset of folds to run")
    p.add_argument("--cv_seed", type=int, default=42)
    p.add_argument("--phash_thr", type=int, default=6, help="max Hamming distance (64-bit pHash)")
    p.add_argument("--emb_thr", type=float, default=0.95, help="min cosine similarity of ImageNet embeddings")
    p.add_argument("--max_group_frac", type=float, default=0.01,
                   help="if a duplicate group exceeds this share of the data, group on pHash only")
    p.add_argument("--hp_lrs", nargs="+", type=float, default=[3e-5, 1e-4, 3e-4])
    # recipe (identical to the original submission unless overridden)
    p.add_argument("--img_size", type=int, default=288)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head_mult", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--head_dropout", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--explain_n", type=int, default=6, help="best and worst cases per model")
    p.add_argument("--deletion_n", type=int, default=200)
    p.add_argument("--max_per_class", type=int, default=0, help="debug only")
    p.add_argument("--no_pretrained", action="store_true", help="debug only")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.models == ["all"]:
        a.models = list(ALL_MODELS)
    bad = set(a.variants) - set(VARIANTS)
    if bad:
        p.error(f"unknown variants {bad}")
    return a


# =================================================================================================
# STAGE 1: AUDIT
# =================================================================================================
def _scan_all(args):
    base = cdb.find_dataset(argparse.Namespace(data_path=args.data_path))
    mode, layout = cdb.discover_layout(Path(base))
    rows = []
    if mode == "predefined":
        for sp, root in layout.items():
            if root is not None:
                rows += cdb._scan(root, sp)
    else:
        rows += cdb._scan(layout, "all")
    return pd.DataFrame(rows, columns=["path", "label", "split"])


def cleaning_audit(args, out: Path) -> pd.DataFrame:
    """Re-runs the original cleaning but records the reason for every removed file."""
    df = _scan_all(args)
    res = [cdb._md5_and_check(p) for p in cdb.tqdm(df.path, desc="hash+verify")]
    df["md5"] = [h for h, _ in res]
    df["ok"] = [ok for _, ok in res]
    df["reason"] = np.where(df.ok, "kept", "corrupt")
    okm = df.ok
    nlab = df[okm].groupby("md5").label.nunique()
    conflict = set(nlab[nlab > 1].index)
    df.loc[okm & df.md5.isin(conflict), "reason"] = "label_conflict"
    live = df.reason == "kept"
    prio = {"test": 0, "val": 1, "train": 2}
    order = df[live].assign(_o=df.split.map(prio).fillna(3)).sort_values(["_o", "path"])
    dup = order.duplicated("md5", keep="first")
    df.loc[order.index[dup.values], "reason"] = "exact_duplicate_same_label"
    df.to_csv(out / "audit" / "cleaning_log.csv", index=False)
    per_class = df.pivot_table(index="label", columns="reason", values="path", aggfunc="count", fill_value=0)
    per_class["scanned"] = per_class.sum(1)
    per_class["removed_%"] = 100 * (1 - per_class.get("kept", 0) / per_class.scanned)
    per_class.sort_values("scanned", ascending=False).to_csv(out / "audit" / "cleaning_per_class.csv")
    summary = df.reason.value_counts().to_dict()
    summary["scanned"] = int(len(df))
    # where do conflicting labels come from? (pairs of labels sharing identical bytes)
    pairs = (df[okm & df.md5.isin(conflict)].groupby("md5").label.apply(lambda s: " | ".join(sorted(set(s))))
             .value_counts().rename_axis("label_set").reset_index(name="n_hashes"))
    pairs.to_csv(out / "audit" / "label_conflict_pairs.csv", index=False)
    cdb.save_json(summary, out / "audit" / "cleaning_summary.json")
    print("[audit] cleaning:", summary)
    return df[df.reason == "kept"].drop(columns=["ok", "reason"]).reset_index(drop=True)


def _phash_bits(paths):
    import imagehash
    H = np.zeros((len(paths), 64), dtype=np.uint8)
    for i, p in enumerate(cdb.tqdm(paths, desc="pHash")):
        with Image.open(p) as im:
            H[i] = imagehash.phash(im.convert("RGB")).hash.flatten().astype(np.uint8)
    return np.packbits(H, axis=1)  # (n, 8) uint8


def _hamming_pairs(packed, thr, chunk=512):
    n = len(packed)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = []
    for s in range(0, n, chunk):
        a = packed[s:s + chunk]
        d = lut[np.bitwise_xor(a[:, None, :], packed[None, :, :])].sum(-1)
        ii, jj = np.where(d <= thr)
        keep = (ii + s) < jj
        out += [(int(i + s), int(j), int(d[i, j])) for i, j in zip(ii[keep], jj[keep])]
    return out


@torch.inference_mode()
def _embeddings(paths, device, bs=64):
    from torchvision import models, transforms
    try:
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception as e:  # offline smoke test only
        print(f"[audit] WARNING: could not load ImageNet weights ({e}); embeddings are NOT meaningful")
        m = models.resnet50(weights=None)
    m.fc = nn.Identity()
    m = m.to(device).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(cdb.NORM_MEAN, cdb.NORM_STD)])
    ds = cdb.ImgDS(paths, [0] * len(paths), tf)
    E = []
    for x, _ in cdb.tqdm(DataLoader(ds, batch_size=bs, num_workers=2), desc="embed"):
        E.append(F.normalize(m(x.to(device)).float(), dim=1).cpu())
    return torch.cat(E)


def _cosine_pairs(E, thr, chunk=1024):
    out = []
    for s in range(0, len(E), chunk):
        sim = E[s:s + chunk] @ E.T
        ii, jj = torch.where(sim >= thr)
        keep = (ii + s) < jj
        out += [(int(i + s), int(j), float(sim[i, j])) for i, j in zip(ii[keep], jj[keep])]
    return out


def _union_find(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def near_duplicate_audit(df, args, out: Path, device):
    paths = df.path.tolist()
    ph = _hamming_pairs(_phash_bits(paths), args.phash_thr)
    E = _embeddings(paths, device)
    cs = _cosine_pairs(E, args.emb_thr)
    pairs = {}
    for i, j, d in ph:
        pairs.setdefault((i, j), {})["hamming"] = d
    for i, j, s in cs:
        pairs.setdefault((i, j), {})["cosine"] = s
    rows = []
    for (i, j), v in pairs.items():
        rows.append(dict(i=i, j=j, hamming=v.get("hamming", np.nan), cosine=v.get("cosine", np.nan),
                         label_i=df.label[i], label_j=df.label[j], path_i=df.rel_path[i], path_j=df.rel_path[j]))
    P = pd.DataFrame(rows)
    group = _union_find(len(df), [(r.i, r.j) for r in P.itertuples()]) if len(P) else np.arange(len(df))
    biggest = np.bincount(group).max() if len(group) else 1
    if biggest > args.max_group_frac * len(df) and len(P):
        # transitive chaining of embedding neighbours can merge many merely similar leaves into one group;
        # fall back to pHash-only grouping (still reported as such in the summary)
        print(f"[audit] WARNING: largest group {biggest} images > {args.max_group_frac:.0%} of data; "
              "grouping on pHash pairs only (embedding pairs are still reported)")
        ph_edges = [(r.i, r.j) for r in P.itertuples() if not np.isnan(r.hamming)]
        group = _union_find(len(df), ph_edges)
        grouping_basis = "phash_only"
    else:
        grouping_basis = "phash_or_embedding"
    df = df.assign(group=group)
    sizes = df.group.map(df.group.value_counts())
    summ = {"pairs_total": int(len(P)),
            "pairs_phash": int(P.hamming.notna().sum()) if len(P) else 0,
            "pairs_embedding": int(P.cosine.notna().sum()) if len(P) else 0,
            "pairs_both": int((P.hamming.notna() & P.cosine.notna()).sum()) if len(P) else 0,
            "pairs_cross_label": int((P.label_i != P.label_j).sum()) if len(P) else 0,
            "images_in_groups_gt1": int((sizes > 1).sum()),
            "groups_gt1": int(df[sizes > 1].group.nunique()),
            "largest_group": int(sizes.max()), "grouping_basis": grouping_basis}
    # leakage in the ORIGINAL seed-42 split
    osp = Path(args.orig_split)
    if osp.exists() and len(P):
        o = pd.read_csv(osp)
        key = "rel_path" if "rel_path" in o.columns else "path"
        smap = dict(zip(o[key], o.split))
        P["split_i"] = P.path_i.map(smap) if key == "rel_path" else P.i.map(lambda k: smap.get(df.path[k]))
        P["split_j"] = P.path_j.map(smap) if key == "rel_path" else P.j.map(lambda k: smap.get(df.path[k]))
        cross = P[(P.split_i != P.split_j) & P.split_i.notna() & P.split_j.notna()]
        summ["orig_split_cross_pairs"] = int(len(cross))
        tst = set(cross[cross.split_i == "test"].path_i) | set(cross[cross.split_j == "test"].path_j)
        summ["orig_test_images_with_neardup_outside_test"] = int(len(tst))
        pd.Series(sorted(tst), name="rel_path").to_csv(out / "audit" / "orig_test_images_with_neardup.csv",
                                                        index=False)
    P.to_csv(out / "audit" / "near_duplicate_pairs.csv", index=False)
    per_class = df.assign(in_group=sizes > 1).groupby("label").in_group.agg(["sum", "count"])
    per_class.columns = ["images_in_dup_groups", "images"]
    per_class.to_csv(out / "audit" / "near_duplicates_per_class.csv")
    cdb.save_json(summ, out / "audit" / "near_duplicate_summary.json")
    print("[audit] near-duplicates:", summ)
    _montage(P, df, out / "audit" / "near_duplicate_examples")
    return df


def _montage(P, df, stem, n=12):
    import matplotlib.pyplot as plt
    if not len(P):
        return
    sel = pd.concat([P[P.label_i != P.label_j].head(n // 2), P[P.label_i == P.label_j].sample(
        min(n // 2, (P.label_i == P.label_j).sum()), random_state=0)])
    fig, ax = plt.subplots(len(sel), 2, figsize=(5, 2.4 * len(sel)))
    ax = np.atleast_2d(ax)
    for r, row in enumerate(sel.itertuples()):
        for c, (k, lab) in enumerate([(row.i, row.label_i), (row.j, row.label_j)]):
            with Image.open(df.path[k]) as im:
                ax[r, c].imshow(im.convert("RGB"))
            ax[r, c].set_title(f"{lab[:28]}", fontsize=8)
            ax[r, c].axis("off")
    fig.tight_layout()
    cdb.save_fig(fig, stem)
    plt.close(fig)


def stage_audit(args, out: Path, device):
    (out / "audit").mkdir(parents=True, exist_ok=True)
    kept = cleaning_audit(args, out)
    if args.max_per_class:
        kept = kept.groupby("label", group_keys=False).head(args.max_per_class).reset_index(drop=True)
    base = Path(cdb.find_dataset(argparse.Namespace(data_path=args.data_path)))
    kept["rel_path"] = [str(Path(p).relative_to(base)) if str(p).startswith(str(base)) else p for p in kept.path]
    kept = near_duplicate_audit(kept, args, out, device)
    kept["label_corrected"] = kept.label.map(lambda s: MERGE.get(s, s))
    kept["crop"] = kept.label.map(crop_of)
    kept.to_csv(out / "audit" / "master_index.csv", index=False)
    tax = (kept.groupby(["label", "label_corrected", "crop"]).size().rename("images").reset_index())
    tax["display_name"] = tax.label_corrected.map(lambda s: RENAME.get(s, s))
    tax.to_csv(out / "audit" / "taxonomy.csv", index=False)
    print(f"[audit] master index: {len(kept)} images | original classes {kept.label.nunique()} | "
          f"corrected classes {kept.label_corrected.nunique()}")


# =================================================================================================
# STAGE 2: GROUP-AWARE CROSS-VALIDATION
# =================================================================================================
def load_master(args, out: Path):
    m = pd.read_csv(out / "audit" / "master_index.csv")
    col = "label_corrected" if args.taxonomy == "corrected" else "label"
    classes = sorted(m[col].unique())
    m["y"] = m[col].map({c: i for i, c in enumerate(classes)})
    crops = [crop_of(c) for c in classes]
    return m, classes, crops


def fold_assignment(m, args):
    """Outer test fold via StratifiedGroupKFold; inner validation = one group-stratified sixth of the rest."""
    f = np.full(len(m), -1)
    sgk = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.cv_seed)
    for k, (_, te) in enumerate(sgk.split(np.zeros(len(m)), m.y, m.group)):
        f[te] = k
    return f


def inner_val(m, train_idx, seed):
    sgk = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    sub = m.iloc[train_idx]
    tr, va = next(sgk.split(np.zeros(len(sub)), sub.y, sub.group))
    return train_idx[tr], train_idx[va]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.g = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), 1).gather(1, y[:, None])[:, 0]
        return (-(1 - logp.exp()) ** self.g * logp).mean()


class LogitAdjustedLoss(nn.Module):
    def __init__(self, prior, tau=1.0):
        super().__init__()
        self.register_buffer("adj", tau * torch.log(prior.clamp_min(1e-12)))

    def forward(self, logits, y):
        return F.cross_entropy(logits.float() + self.adj, y)


def make_criterion(variant, ytr, C, device):
    counts = torch.tensor(np.bincount(ytr, minlength=C), dtype=torch.float32)
    if variant == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if variant in ("ce", "rs"):
        return nn.CrossEntropyLoss()
    if variant == "cb":
        return nn.CrossEntropyLoss(weight=cdb.effective_number_weights(ytr, C, 0.999).to(device))
    if variant == "focal":
        return FocalLoss(2.0)
    if variant == "la":
        return LogitAdjustedLoss((counts / counts.sum()).to(device))
    raise ValueError(variant)


@torch.inference_mode()
def predict_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            L.append(model(x.to(device, non_blocking=True)).float().cpu())
        Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def macro_f1(y, p, C):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0))


def train_eval(name, variant, m, tr, va, te, C, args, device, run_dir: Path, lr_backbone=None, save_test=True):
    res_path = run_dir / "result.json"
    if res_path.exists() and not args.force:
        print(f"  [skip] {run_dir}")
        return json.load(open(res_path))
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = args.cv_seed
    cdb.set_seed(seed)
    tf_tr, tf_ev = cdb.get_transforms(args.img_size)
    ytr = m.y.values[tr]
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
              worker_init_fn=cdb.seed_worker, persistent_workers=args.num_workers > 0)
    g = torch.Generator()
    g.manual_seed(seed)
    ds_tr = cdb.ImgDS(m.path.values[tr], ytr, tf_tr)
    if variant == "rs":
        cnt = np.bincount(ytr, minlength=C).astype(float)
        w = 1.0 / np.sqrt(cnt[ytr])
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(tr),
                                        replacement=True, generator=g)
        train_l = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, drop_last=True, **kw)
    else:
        train_l = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g, **kw)
    val_l = DataLoader(cdb.ImgDS(m.path.values[va], m.y.values[va], tf_ev), batch_size=64, shuffle=False, **kw)

    model, head, _, _ = cdb.build_model(name, C, not args.no_pretrained, args.head_dropout)
    model = model.to(device)
    hid = {id(p) for p in head.parameters()}
    lrb = lr_backbone or args.lr_backbone
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in hid], "lr": lrb},
                             {"params": list(head.parameters()), "lr": lrb * args.lr_head_mult}],
                            weight_decay=args.weight_decay)
    total, warm = args.epochs * len(train_l), max(1, args.warmup_epochs * len(train_l))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(
            math.pi * (s - warm) / max(1, total - warm))))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = make_criterion(variant, ytr, C, device)

    best, bad, hist, t0 = {"f1": -1, "loss": 1e9, "epoch": 0, "state": None}, 0, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, seen = 0.0, 0
        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sched.step()
            tl += loss.item() * y.size(0)
            seen += y.size(0)
        vl, vy = predict_logits(model, val_l, device)
        vf1 = macro_f1(vy, vl.argmax(1), C)
        vloss = float(F.cross_entropy(torch.tensor(vl), torch.tensor(vy)))
        hist.append(dict(epoch=ep, train_loss=tl / max(seen, 1), val_nll=vloss, val_f1=vf1))
        if vf1 > best["f1"] + 1e-6 or (abs(vf1 - best["f1"]) <= 1e-6 and vloss < best["loss"]):
            best.update(f1=vf1, loss=vloss, epoch=ep,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        print(f"  [{name}/{variant}] ep {ep:02d} trL {hist[-1]['train_loss']:.4f} vaNLL {vloss:.4f} "
              f"vaF1 {vf1*100:.2f}{' *' if bad == 0 else ''}")
        if bad >= args.patience:
            break
    model.load_state_dict(best["state"])
    vl, vy = predict_logits(model, val_l, device)
    out = dict(model=name, variant=variant, lr_backbone=lrb, best_epoch=best["epoch"], epochs_run=len(hist),
               train_time_min=(time.time() - t0) / 60, n_train=len(tr), n_val=len(va), n_test=len(te),
               val_f1=macro_f1(vy, vl.argmax(1), C), val_acc=float((vl.argmax(1) == vy).mean()))
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    arrays = dict(val_idx=va, val_logits=vl.astype(np.float16))
    if save_test:
        te_l = DataLoader(cdb.ImgDS(m.path.values[te], m.y.values[te], tf_ev), batch_size=64, shuffle=False, **kw)
        tl_, ty = predict_logits(model, te_l, device)
        arrays.update(test_idx=te, test_logits=tl_.astype(np.float16))
        out.update(test_f1=macro_f1(ty, tl_.argmax(1), C), test_acc=float((tl_.argmax(1) == ty).mean()))
        torch.save(best["state"], run_dir / "best.pt")
    np.savez_compressed(run_dir / "logits.npz", **arrays)
    cdb.save_json(out, res_path)
    print(f"  => {name}/{variant}: " + ", ".join(f"{k} {v:.4f}" for k, v in out.items() if k.startswith(("val_", "test_"))))
    del model, opt, best
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def stage_cv(args, out: Path, device):
    m, classes, crops = load_master(args, out)
    C = len(classes)
    fold = fold_assignment(m, args)
    cvdir = out / f"cv_{args.taxonomy}"
    cvdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rel_path": m.rel_path, "y": m.y, "group": m.group, "fold": fold}).to_csv(
        cvdir / "folds.csv", index=False)
    cdb.save_json({"classes": classes, "crops": crops}, cvdir / "classes.json")
    for k in (args.fold_ids if args.fold_ids is not None else range(args.folds)):
        te = np.where(fold == k)[0]
        tr, va = inner_val(m, np.where(fold != k)[0], args.cv_seed + k)
        assert not (set(m.group[te]) & set(m.group[tr])), "group leakage"
        for name in args.models:
            for v in args.variants:
                print(f"\n=== fold {k} | {name} | {v} ===")
                train_eval(name, v, m, tr, va, te, C, args, device, cvdir / v / cdb.slug(name) / f"fold{k}")


def stage_hpsweep(args, out: Path, device):
    m, classes, _ = load_master(args, out)
    fold = fold_assignment(m, args)
    tr, va = inner_val(m, np.where(fold != 0)[0], args.cv_seed)
    for name in args.models:
        for lr in args.hp_lrs:
            train_eval(name, "ce_ls", m, tr, va, np.array([], int), len(classes), args, device,
                       out / "hpsweep" / cdb.slug(name) / f"lr{lr:g}", lr_backbone=lr, save_test=False)


# =================================================================================================
# STAGE 4: EXPLANATIONS (Grad-CAM best/worst, all models, same images) + deletion AUC
# =================================================================================================
def stage_explain(args, out: Path, device):
    import matplotlib.pyplot as plt
    m, classes, _ = load_master(args, out)
    C = len(classes)
    cvdir = out / f"cv_{args.taxonomy}"
    _, tf_ev = cdb.get_transforms(args.img_size)
    P = {}
    for name in args.models:
        z = np.load(cvdir / "ce_ls" / cdb.slug(name) / "fold0" / "logits.npz")
        P[name] = (z["test_idx"], torch.softmax(torch.tensor(z["test_logits"].astype(np.float32)), 1).numpy())
    idx = P[args.models[0]][0]
    y = m.y.values[idx]
    probs = np.stack([P[n][1] for n in args.models])            # (M, N, C)
    pt = probs[:, np.arange(len(idx)), y]                        # prob of true class
    allcorrect = (probs.argmax(2) == y).all(0)
    allwrong = (probs.argmax(2) != y).all(0)
    # consensus best/worst cases; relax to "majority of models" if too few unanimous cases exist
    maj_c = (probs.argmax(2) == y).mean(0) > 0.5
    best = [i for i in np.argsort(-pt.mean(0)) if allcorrect[i]] or [i for i in np.argsort(-pt.mean(0)) if maj_c[i]]
    worst = [i for i in np.argsort(pt.mean(0)) if allwrong[i]] or [i for i in np.argsort(pt.mean(0)) if not maj_c[i]]
    # one image per class where possible
    def distinct(lst, k):
        seen, sel = set(), []
        for i in lst:
            if y[i] not in seen:
                sel.append(i)
                seen.add(y[i])
            if len(sel) == k:
                break
        return sel
    sel = {"best": distinct(best, args.explain_n), "worst": distinct(worst, args.explain_n)}
    rng = np.random.default_rng(0)
    del_sample = rng.choice(len(idx), size=min(args.deletion_n, len(idx)), replace=False)
    rows = []
    for kind, ii in sel.items():
        if not ii:
            print(f"[explain] no {kind} cases found; skipped")
            continue
        fig, ax = plt.subplots(len(ii), len(args.models) + 1, figsize=(2.3 * (len(args.models) + 1), 2.5 * len(ii)))
        ax = np.atleast_2d(ax)
        for r, i in enumerate(ii):
            with Image.open(m.path.values[idx[i]]) as im:
                img = im.convert("RGB").resize((args.img_size, args.img_size))
            ax[r, 0].imshow(img)
            ax[r, 0].set_title(f"T: {classes[y[i]][:24]}", fontsize=8)
            ax[r, 0].axis("off")
        for c, name in enumerate(args.models, 1):
            model, _, layer, cl = cdb.build_model(name, C, False, args.head_dropout)
            model.load_state_dict(torch.load(cvdir / "ce_ls" / cdb.slug(name) / "fold0" / "best.pt",
                                             map_location="cpu"))
            model = model.to(device).eval()
            gc_ = cdb.GradCAM(model, layer, cl)
            for r, i in enumerate(ii):
                with Image.open(m.path.values[idx[i]]) as im:
                    x = tf_ev(im.convert("RGB"))[None].to(device)
                cam, logits = gc_(x)
                pr = torch.softmax(logits, 1)[0]
                ax[r, c].imshow(np.asarray(Image.open(m.path.values[idx[i]]).convert("RGB")
                                           .resize((args.img_size, args.img_size))))
                ax[r, c].imshow(cam[0], cmap="jet", alpha=0.45)
                ax[r, c].set_title(f"{name}\nP: {classes[int(pr.argmax())][:18]} ({float(pr.max()):.2f})", fontsize=7)
                ax[r, c].axis("off")
            gc_.remove()
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        fig.tight_layout()
        cdb.save_fig(fig, out / "figures" / f"gradcam_{kind}_all_models")
        plt.close(fig)
    for name in args.models:  # deletion AUC on the same random fold-0 sample for every model
        model, _, layer, cl = cdb.build_model(name, C, False, args.head_dropout)
        model.load_state_dict(torch.load(cvdir / "ce_ls" / cdb.slug(name) / "fold0" / "best.pt", map_location="cpu"))
        model = model.to(device).eval()
        gc_ = cdb.GradCAM(model, layer, cl)
        rows.append(dict(model=name, n_images=len(del_sample),
                         deletion_auc=_deletion_auc(model, gc_, m.path.values[idx[del_sample]], tf_ev, device)))
        gc_.remove()
        del model
    pd.DataFrame(rows).to_csv(out / "tables" / "deletion_auc.csv", index=False)
    print(pd.DataFrame(rows))


def _deletion_auc(model, gc_, paths, tf_ev, device, steps=10):
    """Mean area under the class-probability curve when the most salient pixels are removed first
    (lower = explanation is more faithful) [Petsiuk et al. 2018]. Removed pixels are set to 0 (= mean colour)."""
    xs_all, cams, cls = [], [], []
    for p in paths:
        with Image.open(p) as im:
            x = tf_ev(im.convert("RGB"))[None].to(device)
        cam, logits = gc_(x)
        xs_all.append(x)
        cams.append(cam[0])
        cls.append(int(logits.argmax()))
    gc_.remove()  # the Grad-CAM hook must not be active during the no-grad deletion passes
    aucs = []
    for x, cam, c in zip(xs_all, cams, cls):
        order = np.argsort(-cam.ravel())
        n = order.size
        xs = x.clone()
        probs = []
        with torch.inference_mode():
            for s in range(steps + 1):
                if s:
                    k = order[(s - 1) * n // steps: s * n // steps]
                    xs.view(1, 3, -1)[:, :, torch.as_tensor(k, device=device)] = 0
                probs.append(float(torch.softmax(model(xs), 1)[0, c]))
        aucs.append(np.trapezoid(probs, dx=1 / steps) if hasattr(np, "trapezoid") else np.trapz(probs, dx=1 / steps))
    return float(np.mean(aucs))


def main():
    args = parse_args()
    out = Path(args.output_dir)
    for d in ["audit", "tables", "figures"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    cdb.save_json(vars(args), out / f"config_{args.stage}.json")
    {"audit": stage_audit, "cv": stage_cv, "hpsweep": stage_hpsweep, "explain": stage_explain}[args.stage](
        args, out, device)
    print("Done.")


if __name__ == "__main__":
    main()
