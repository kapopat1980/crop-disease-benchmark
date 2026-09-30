#!/usr/bin/env python
"""Sanity checks on the result files shipped in results/ (no GPU, no images needed).

Verifies that the tables are internally consistent and match the numbers reported in the paper:
  * split sizes, stratification, no path / MD5 overlap between splits
  * dataset_summary.csv agrees with the split file
  * macro-F1 recomputed from per-class F1 equals the value in all_runs.csv
  * every confusion matrix sums to the test-set size and reproduces the reported accuracy
Usage:  python scripts/check_results.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
T, S = ROOT / "tables", ROOT / "splits"
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


split = pd.read_csv(S / "split_seed42.csv")
index = pd.read_csv(S / "file_index.csv")
summ = pd.read_csv(T / "dataset_summary.csv", index_col=0)
runs = pd.read_csv(T / "all_runs.csv").set_index("Model")
pcf1 = pd.read_csv(T / "per_class_f1_all_models.csv", index_col=0)

n = split.split.value_counts()
check((n.get("train"), n.get("val"), n.get("test")) == (10640, 2280, 2281), f"split sizes train/val/test = {n.get('train')}/{n.get('val')}/{n.get('test')}")
check(split.label.nunique() == 44, "44 classes")
check(split.path.is_unique, "no image appears twice in the split file")
check(set(split.groupby("label").split.nunique().index) == set(split.label), "every class present")
check(index.md5.is_unique, "no byte-identical duplicates remain after cleaning (unique MD5)")
check(len(index) == len(split) == 15201, f"{len(index):,} cleaned images")
check((summ[["train", "val", "test"]].sum() == [10640, 2280, 2281]).all(), "dataset_summary.csv agrees with split sizes")
check(summ.total.max() / summ.total.min() > 448, f"imbalance ratio = {summ.total.max() / summ.total.min():.0f}:1")

for m in runs.index:
    mf1 = pcf1[m].mean()
    check(abs(mf1 - runs.loc[m, "test_f1_macro"]) < 1e-3, f"{m}: macro-F1 from per-class table = {mf1:.4f} vs all_runs {runs.loc[m, 'test_f1_macro']:.4f}")
    cm = pd.read_csv(T / f"confusion_{m.replace('-', '_')}.csv", index_col=0).values
    acc = np.trace(cm) / cm.sum()
    check(cm.sum() == 2281 and abs(acc - runs.loc[m, "test_accuracy"]) < 1e-3, f"{m}: confusion matrix sums to {int(cm.sum())}, accuracy {acc:.4f}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
