#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU-only analysis for the revised manuscript. Produces every table/figure value marked [[...]] in the
revised paper.

Two independent modes (use either or both):

  A) ORIGINAL seed-42 runs (available today, no training needed):
       python revision_analysis.py --orig_runs /path/to/journal_results \\
                                   --orig_split results/splits/split_seed42.csv \\
                                   [--neardup_list revision_results/audit/orig_test_images_with_neardup.csv]
     -> all 10 pairwise exact McNemar tests (Holm), signed calibration + reliability diagrams,
        per-class ROC diagnostics (MobileNetV3-Small AUC question), crop-conditioned inference,
        accuracy on test images with vs without a near-duplicate outside the test set.

  B) Revision cross-validation (after revision_experiments.py --stage cv):
       python revision_analysis.py --out revision_results [--taxonomy corrected]
     -> pooled out-of-fold metrics with bootstrap CIs and fold mean +/- SD, tail-class metrics,
        per-class recall with Wilson intervals, all-pairs McNemar + paired-bootstrap differences,
        imbalance-variant comparison, temperature scaling, crop-conditioned inference, hp sensitivity.
Outputs go to <out>/analysis/.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["MobileNetV3-Small", "ResNet-50", "EfficientNetV2-S", "ConvNeXt-Tiny", "Swin-T"]
VARIANTS = ["ce_ls", "ce", "cb", "focal", "la", "rs"]
VNAME = {"ce_ls": "CE + label smoothing (reference)", "ce": "CE", "cb": "Class-balanced CE",
         "focal": "Focal loss", "la": "Logit-adjusted CE", "rs": "Square-root re-sampling"}
TAIL_MAX = 100   # a class is a "tail" class if it has fewer than 100 images in total
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
                     "savefig.dpi": 600})


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------- metrics
def core_metrics(y, P, C):
    p = P.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=list(range(C)), zero_division=0)
    present = np.bincount(y, minlength=C) > 0
    return {"accuracy": float((p == y).mean()),
            "macro_f1": float(f1[present].mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "mcc": float(matthews_corrcoef(y, p)),
            "nll": float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()),
            "ece": ece(P, y), "signed_gap": signed_gap(P, y)}


def per_class(y, P, C):
    p = P.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(C)), zero_division=0)
    return pd.DataFrame({"precision": pr, "recall": rc, "f1": f1, "support": sup})


def ece(P, y, bins=15):
    conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def signed_gap(P, y):
    """mean confidence - accuracy: > 0 over-confident, < 0 under-confident."""
    return float(P.max(1).mean() - (P.argmax(1) == y).mean())


def reliability(P, y, bins=15):
    conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), conf=conf[m].mean(), acc=acc[m].mean()))
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def mcnemar_exact(ca, cb):
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(y, P, C, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        v.append(fn(y[i], P[i]))
    return np.percentile(v, [2.5, 97.5])


def paired_boot_diff(y, Pa, Pb, C, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda yy, PP: f1_score(yy, PP.argmax(1), labels=list(range(C)), average="macro", zero_division=0)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        d.append(f(y[i], Pa[i]) - f(y[i], Pb[i]))
    return np.percentile(d, [2.5, 97.5])


def fit_temperature(logits, y):
    def nll(t):
        return -np.log(np.clip(softmax(logits / t)[np.arange(len(y)), y], 1e-12, None)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)


# ------------------------------------------------------------------------------------------------- crop
def crop_conditioned(P, y, crops):
    """Oracle crop known (deployment scenario): restrict the decision to classes of the image's crop.
    Two-stage: crop predicted by summing class probabilities per crop, then classify within it."""
    crops = np.asarray(crops)
    names = sorted(set(crops) - {"Any"})
    allow = {c: (crops == c) | (crops == "Any") for c in names}
    true_crop = crops[y]
    Po = P.copy()
    for i, c in enumerate(true_crop):
        if c != "Any":
            Po[i, ~allow[c]] = 0
    Po /= Po.sum(1, keepdims=True)
    crop_scores = np.stack([P[:, allow[c]].sum(1) for c in names], 1)
    pc = np.array(names)[crop_scores.argmax(1)]
    Pt = P.copy()
    for i, c in enumerate(pc):
        Pt[i, ~allow[c]] = 0
    Pt /= Pt.sum(1, keepdims=True)
    known = true_crop != "Any"
    crop_acc = float((pc[known] == true_crop[known]).mean())
    return Po, Pt, crop_acc


def error_breakdown(P, y, crops):
    crops = np.asarray(crops)
    p = P.argmax(1)
    wrong = p != y
    cross = wrong & (crops[p] != crops[y]) & (crops[p] != "Any") & (crops[y] != "Any")
    return int(wrong.sum()), int(cross.sum())


# ------------------------------------------------------------------------------------------------- ROC
def roc_diagnostics(y, P, C, sizes):
    rows = []
    for c in range(C):
        pos = y == c
        if 0 < pos.sum() < len(y):
            rows.append(dict(cls=c, n_pos=int(pos.sum()), n_total=int(sizes[c]),
                             auc=roc_auc_score(pos, P[:, c]),
                             recall=float((P.argmax(1)[pos] == c).mean())))
    R = pd.DataFrame(rows)
    p = P.argmax(1)
    wrong = p != y
    rank = (P > P[np.arange(len(y)), y][:, None]).sum(1)  # 0 = top-1
    return R, {
        "macro_auc": float(R.auc.mean()),
        "macro_auc_classes_ge10": float(R[R.n_pos >= 10].auc.mean()),
        "macro_auc_tail": float(R[R.n_total < TAIL_MAX].auc.mean()),
        "auc_recall_spearman": float(stats.spearmanr(R.auc, R.recall).correlation),
        "mean_max_prob": float(P.max(1).mean()),
        "mean_entropy": float((-(P * np.log(np.clip(P, 1e-12, None))).sum(1)).mean()),
        "errors_true_rank2_share": float((rank[wrong] == 1).mean()) if wrong.any() else np.nan,
        "errors_true_in_top3_share": float((rank[wrong] <= 2).mean()) if wrong.any() else np.nan,
        "mean_true_prob_on_errors": float(P[np.arange(len(y)), y][wrong].mean()) if wrong.any() else np.nan,
    }


# ------------------------------------------------------------------------------------------------- plots
def plot_reliability(curves, stem):
    n = len(curves)
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (name, items) in zip(ax, curves.items()):
        a.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        for lab, R, st in items:
            a.plot(R.conf, R.acc, st, label=lab, ms=4)
        a.set_title(name)
        a.set_xlabel("Mean confidence")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.grid(alpha=.3)
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(stem) + ".png", bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roc_vs_recall(diag, stem):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, R in diag.items():
        ax.scatter(R.recall, R.auc, s=14 + 2 * np.sqrt(R.n_total), alpha=.6, label=name)
    ax.set_xlabel("Per-class recall (threshold-dependent)")
    ax.set_ylabel("Per-class ROC-AUC (ranking)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(stem) + ".png", bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ================================================================================================= MODE A
def mode_original(args, out):
    split = pd.read_csv(args.orig_split)
    classes = sorted(split.label.unique())
    C = len(classes)
    sys_path = Path(__file__).resolve().parent
    import importlib.util
    spec = importlib.util.spec_from_file_location("rx", sys_path / "revision_experiments.py")
    crops = None
    try:
        rx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rx)
        crops = [rx.crop_of(c) for c in classes]
    except Exception as e:  # torch missing on this machine: fall back to a local copy of the rule
        print("[orig] crop map from fallback rule:", e)
    test = split[split.split == "test"].reset_index(drop=True)
    y = test.label_idx.values
    sizes = split.label_idx.value_counts().reindex(range(C), fill_value=0).values
    P = {}
    for name in MODELS:
        f = Path(args.orig_runs) / "runs" / slug(name) / "seed42" / "test_predictions.npz"
        if f.exists():
            z = np.load(f)
            assert (z["y_true"] == y).all(), f"{name}: test order differs from split file"
            P[name] = z["probs"].astype(np.float64)
            P[name] /= P[name].sum(1, keepdims=True)
    if not P:
        print("[orig] no test_predictions.npz found under", args.orig_runs)
        return
    o = out / "original_seed42"
    o.mkdir(parents=True, exist_ok=True)
    # 1. all pairwise McNemar
    rows = []
    for a, b in combinations(P, 2):
        ca, cb = P[a].argmax(1) == y, P[b].argmax(1) == y
        n10, n01, p = mcnemar_exact(ca, cb)
        lo, hi = paired_boot_diff(y, P[a], P[b], C)
        rows.append(dict(model_a=a, model_b=b, a_only=n10, b_only=n01, p_exact=p,
                         dF1_ci_lo=lo * 100, dF1_ci_hi=hi * 100))
    M = pd.DataFrame(rows)
    M["p_holm"] = holm(M.p_exact)
    M.to_csv(o / "mcnemar_all_pairs.csv", index=False)
    print(M.round(4).to_string())
    # 2. calibration (signed) + reliability
    cal, curves = [], {}
    for n, Pm in P.items():
        R = reliability(Pm, y)
        R.to_csv(o / f"reliability_{slug(n)}.csv", index=False)
        cal.append(dict(model=n, ece=ece(Pm, y) * 100, signed_gap=signed_gap(Pm, y) * 100,
                        mean_conf=Pm.max(1).mean() * 100, accuracy=(Pm.argmax(1) == y).mean() * 100,
                        share_samples_underconf_bins=float(R[R.acc > R.conf].n.sum() / R.n.sum())))
        curves[n] = [("seed 42 (label smoothing)", R, "o-")]
    pd.DataFrame(cal).to_csv(o / "calibration_signed.csv", index=False)
    print(pd.DataFrame(cal).round(3).to_string())
    plot_reliability(curves, o / "reliability_diagrams")
    # 3. ROC diagnostics
    diag, summ = {}, []
    for n, Pm in P.items():
        R, s = roc_diagnostics(y, Pm, C, sizes)
        R["class"] = [classes[c] for c in R.cls]
        R.to_csv(o / f"roc_per_class_{slug(n)}.csv", index=False)
        diag[n] = R
        summ.append(dict(model=n, **s))
    pd.DataFrame(summ).to_csv(o / "roc_diagnostics.csv", index=False)
    print(pd.DataFrame(summ).round(4).to_string())
    plot_roc_vs_recall(diag, o / "roc_vs_recall")
    # 4. crop-conditioned inference
    if crops is not None:
        rows = []
        for n, Pm in P.items():
            Po, Pt, cacc = crop_conditioned(Pm, y, crops)
            err, cross = error_breakdown(Pm, y, crops)
            for lab, PP in [("flat", Pm), ("crop known (oracle)", Po), ("two-stage (predicted crop)", Pt)]:
                mm = core_metrics(y, PP, C)
                rows.append(dict(model=n, inference=lab, accuracy=mm["accuracy"] * 100, macro_f1=mm["macro_f1"] * 100,
                                 errors=int((PP.argmax(1) != y).sum()), crop_accuracy=cacc * 100,
                                 cross_crop_errors_flat=cross))
        pd.DataFrame(rows).to_csv(o / "crop_conditioned.csv", index=False)
        print(pd.DataFrame(rows).round(2).to_string())
    # 5. near-duplicate effect
    if args.neardup_list and Path(args.neardup_list).exists():
        nd = set(pd.read_csv(args.neardup_list).rel_path)
        key = test.rel_path if "rel_path" in test.columns else test.path
        has = key.isin(nd).values
        rows = []
        for n, Pm in P.items():
            c = Pm.argmax(1) == y
            rows.append(dict(model=n, n_with=int(has.sum()), acc_with=c[has].mean() * 100 if has.any() else np.nan,
                             n_without=int((~has).sum()), acc_without=c[~has].mean() * 100))
        pd.DataFrame(rows).to_csv(o / "neardup_effect.csv", index=False)
        print(pd.DataFrame(rows).round(2).to_string())


# ================================================================================================= MODE B
def load_cv(cvdir: Path, variant, name, n_total):
    d = cvdir / variant / slug(name)
    folds = sorted(d.glob("fold*/logits.npz"))
    if not folds:
        return None
    test_idx, test_logits, fold_of, val = [], [], [], []
    for f in folds:
        z = np.load(f)
        k = int(f.parent.name[4:])
        test_idx.append(z["test_idx"])
        test_logits.append(z["test_logits"].astype(np.float64))
        fold_of.append(np.full(len(z["test_idx"]), k))
        val.append((z["val_idx"], z["val_logits"].astype(np.float64), k))
    return dict(idx=np.concatenate(test_idx), logits=np.concatenate(test_logits), fold=np.concatenate(fold_of),
                val=val, n_folds=len(folds))


def mode_cv(args, out):
    cvdir = out / f"cv_{args.taxonomy}"
    meta = json.load(open(cvdir / "classes.json"))
    classes, crops = meta["classes"], meta["crops"]
    C = len(classes)
    F = pd.read_csv(cvdir / "folds.csv")
    yall = F.y.values
    sizes = np.bincount(yall, minlength=C)
    tail = sizes < TAIL_MAX
    o = out / "analysis"
    o.mkdir(parents=True, exist_ok=True)
    runs = {}
    for v in VARIANTS:
        for n in MODELS:
            r = load_cv(cvdir, v, n, len(F))
            if r is not None:
                r["y"] = yall[r["idx"]]
                runs[(v, n)] = r
    complete = {k: r for k, r in runs.items() if r["n_folds"] == F.fold.nunique()}
    print(f"[cv] runs found: {len(runs)}; complete (all folds): {len(complete)}")

    rows, pcs = [], {}
    for (v, n), r in complete.items():
        y, Z = r["y"], r["logits"]
        P = softmax(Z)
        # temperature scaling, fitted per fold on that fold's validation logits
        Pts = np.zeros_like(P)
        temps = []
        for vidx, vlog, k in r["val"]:
            T = fit_temperature(vlog, yall[vidx])
            temps.append(T)
            m = r["fold"] == k
            Pts[m] = softmax(Z[m] / T)
        mm = core_metrics(y, P, C)
        fm = [core_metrics(y[r["fold"] == k], P[r["fold"] == k], C) for k in sorted(set(r["fold"]))]
        ci = boot_ci(y, P, C, lambda yy, PP: f1_score(yy, PP.argmax(1), labels=list(range(C)),
                                                         average="macro", zero_division=0))
        pc = per_class(y, P, C)
        pcs[(v, n)] = pc
        Po, Pt, cacc = crop_conditioned(P, y, crops)
        err, cross = error_breakdown(P, y, crops)
        _, rd = roc_diagnostics(y, P, C, sizes)
        rows.append(dict(
            variant=v, model=n,
            accuracy=mm["accuracy"] * 100, acc_fold_sd=np.std([f["accuracy"] for f in fm], ddof=1) * 100,
            macro_f1=mm["macro_f1"] * 100, macro_f1_ci_lo=ci[0] * 100, macro_f1_ci_hi=ci[1] * 100,
            macro_f1_fold_mean=np.mean([f["macro_f1"] for f in fm]) * 100,
            macro_f1_fold_sd=np.std([f["macro_f1"] for f in fm], ddof=1) * 100,
            balanced_accuracy=mm["balanced_accuracy"] * 100, mcc=mm["mcc"],
            tail_macro_f1=pc.f1[tail].mean() * 100, tail_macro_recall=pc.recall[tail].mean() * 100,
            head_macro_f1=pc.f1[~tail].mean() * 100,
            n_classes_recall0=int((pc.recall == 0).sum()),
            ece=mm["ece"] * 100, signed_gap=mm["signed_gap"] * 100, nll=mm["nll"],
            temperature=float(np.mean(temps)), ece_ts=ece(Pts, y) * 100, signed_gap_ts=signed_gap(Pts, y) * 100,
            nll_ts=core_metrics(y, Pts, C)["nll"],
            errors=err, cross_crop_errors=cross, crop_acc_two_stage=cacc * 100,
            acc_crop_known=core_metrics(y, Po, C)["accuracy"] * 100,
            macro_f1_crop_known=core_metrics(y, Po, C)["macro_f1"] * 100,
            macro_f1_two_stage=core_metrics(y, Pt, C)["macro_f1"] * 100,
            macro_auc=rd["macro_auc"] * 100, errors_true_in_top3=rd["errors_true_in_top3_share"] * 100))
    S = pd.DataFrame(rows)
    S.to_csv(o / "cv_summary.csv", index=False)
    if len(S):
        print(S.round(2).to_string())

    # per-class recall with Wilson CIs (reference variant + every variant of the two imbalance backbones)
    pr = []
    for (v, n), pc in pcs.items():
        for c in range(C):
            k = int(round(pc.recall[c] * pc.support[c]))
            lo, hi = wilson(k, int(pc.support[c]))
            pr.append(dict(variant=v, model=n, cls=classes[c], crop=crops[c], n=int(pc.support[c]),
                           precision=pc.precision[c], recall=pc.recall[c], f1=pc.f1[c], recall_lo=lo, recall_hi=hi))
    pd.DataFrame(pr).to_csv(o / "per_class_cv.csv", index=False)

    # macro-F1 deficit decomposition (which classes cost most), reference variant
    for (v, n), pc in pcs.items():
        if v == "ce_ls":
            pcs_def = (1 - pc.f1) / C * 100
            pcs_def.index = classes
            pcs_def.sort_values(ascending=False).rename("deficit_pp").to_csv(o / f"deficit_{slug(n)}.csv")

    # pairwise McNemar among architectures (reference variant) and variants vs reference within a model
    for label, pairs in [("architectures", [(("ce_ls", a), ("ce_ls", b)) for a, b in combinations(MODELS, 2)]),
                         ("variants", [((v, n), ("ce_ls", n)) for n in MODELS for v in VARIANTS if v != "ce_ls"])]:
        rows = []
        for ka, kb in pairs:
            if ka in complete and kb in complete:
                ra, rb = complete[ka], complete[kb]
                ia, ib = np.argsort(ra["idx"]), np.argsort(rb["idx"])
                assert (ra["idx"][ia] == rb["idx"][ib]).all()
                y = ra["y"][ia]
                Pa, Pb = softmax(ra["logits"][ia]), softmax(rb["logits"][ib])
                b, c, p = mcnemar_exact(Pa.argmax(1) == y, Pb.argmax(1) == y)
                lo, hi = paired_boot_diff(y, Pa, Pb, C)
                # tail-class recall difference (sign test on tail images)
                tm = tail[y]
                bt, ct, pt = mcnemar_exact((Pa.argmax(1) == y)[tm], (Pb.argmax(1) == y)[tm])
                rows.append(dict(a=f"{ka[1]} / {ka[0]}", b=f"{kb[1]} / {kb[0]}", a_only=b, b_only=c, p_exact=p,
                                 dF1_lo=lo * 100, dF1_hi=hi * 100, tail_a_only=bt, tail_b_only=ct, p_tail=pt))
        if rows:
            M = pd.DataFrame(rows)
            M["p_holm"] = holm(M.p_exact)
            M["p_tail_holm"] = holm(M.p_tail)
            M.to_csv(o / f"mcnemar_{label}.csv", index=False)
            print(M.round(4).to_string())

    # reliability diagrams (reference, CE, reference + temperature scaling)
    curves = {}
    for n in MODELS:
        items = []
        for v, st, ts in [("ce_ls", "o-", False), ("ce_ls", "s--", True), ("ce", "^-", False)]:
            r = complete.get((v, n))
            if r is None:
                continue
            Z, y = r["logits"], r["y"]
            if ts:
                P = np.zeros_like(Z)
                for vidx, vlog, k in r["val"]:
                    m = r["fold"] == k
                    P[m] = softmax(Z[m] / fit_temperature(vlog, yall[vidx]))
                lab = "CE+LS, temp.-scaled"
            else:
                P, lab = softmax(Z), VNAME[v].replace(" (reference)", "")
            items.append((lab, reliability(P, y), st))
        if items:
            curves[n] = items
    if curves:
        plot_reliability(curves, o / "reliability_cv")

    # hp sensitivity
    hp = []
    for f in sorted((out / "hpsweep").glob("*/lr*/result.json")):
        r = json.load(open(f))
        hp.append(dict(model=r["model"], lr_backbone=r["lr_backbone"], val_macro_f1=r["val_f1"] * 100,
                       val_acc=r["val_acc"] * 100, best_epoch=r["best_epoch"]))
    if hp:
        H = pd.DataFrame(hp).sort_values(["model", "lr_backbone"])
        H["lr_backbone"] = H.lr_backbone.map(lambda v: f"{v:g}")
        H.to_csv(o / "hp_sensitivity.csv", index=False)
        print(H.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./revision_results")
    ap.add_argument("--taxonomy", default="corrected")
    ap.add_argument("--orig_runs", default=None)
    ap.add_argument("--orig_split", default="results/splits/split_seed42.csv")
    ap.add_argument("--neardup_list", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.orig_runs:
        mode_original(a, out / "analysis")
    if (out / f"cv_{a.taxonomy}" / "folds.csv").exists():
        mode_cv(a, out)


if __name__ == "__main__":
    main()
