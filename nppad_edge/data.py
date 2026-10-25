"""NPPAD data loading and preprocessing for NPPAD-Edge.

The protocol is identical to the Homework 2 study pipeline (the TaskE
package / ``adv_data.py`` used by NPPAD_Advanced_SOTA): same run-level
split rule, same variable sets, same preprocessing (LogHeavy ->
ChannelScaler -> clip), with every fitted transform fitted on the
training split only. This module is a clean, self-contained
re-implementation so the edge project has no dependency on the study
workspace.

Data layout: ``<root>/<CLASS>/<severity>.csv`` with a TIME column and
96 plant variables sampled every 10 s.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
CLASSES = ["LOCA", "LOCAC", "SLBIC", "SLBOC", "SGATR", "SGBTR",
           "RW", "RI", "FLB", "MD", "LR", "LLB"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
# Oracle / leak variables excluded from the "realistic" variable set.
ORACLE = ["WLR", "HLW", "WBK", "MBK", "EBK", "WTRA", "WTRB", "WFLB"]
DT = 10.0            # seconds between samples
CLIP = 10.0          # clip bound applied after scaling (study value)
DEFAULT_T0 = 100.0   # onset-aligned window start (seconds)


def load_runs(root, classes=CLASSES):
    """Load every run CSV under ``root`` (identical rules to the study)."""
    root = Path(root)
    runs = []
    for c in classes:
        folder = root / c
        for f in sorted(folder.glob("*.csv")):
            try:
                sev = float(f.stem)
            except ValueError:
                continue
            d = pd.read_csv(f)
            d.columns = [str(col).strip() for col in d.columns]
            d = d.apply(pd.to_numeric, errors="coerce").sort_values("TIME").reset_index(drop=True)
            d = d.ffill().bfill().fillna(0.0)
            runs.append({"label": c, "severity": sev,
                         "file": f"{c}/{f.name}", "df": d})
    return runs


def assign_splits(runs):
    """Study split rule: per class, sorted by severity, k%5==0 -> test,
    k%5==1 -> val, otherwise train. Yields 725/242/244 runs."""
    for c in CLASSES:
        cls = sorted((r for r in runs if r["label"] == c),
                     key=lambda r: r["severity"])
        for k, r in enumerate(cls):
            r["split"] = "test" if k % 5 == 0 else ("val" if k % 5 == 1 else "train")
    return runs


def variable_sets(runs):
    """(all_vars, realistic_vars): variables present in every run."""
    all_vars = sorted(set.intersection(*(set(r["df"].columns) for r in runs)) - {"TIME"})
    real = [v for v in all_vars if v not in ORACLE]
    return all_vars, real


def build_windows(runs, split, W_s, cols, t0=DEFAULT_T0):
    """Windows of ``W_s`` seconds starting at ``t0`` (study rule).

    Returns (X, y, meta) with X of shape (n, n_steps, n_cols) float32.
    Runs shorter than the window are skipped (as in the study)."""
    n_steps = int(round(W_s / DT))
    X, y, meta = [], [], []
    for r in runs:
        if r["split"] != split:
            continue
        d = r["df"]
        seg = d.loc[d["TIME"] >= t0 - 1e-6, cols].iloc[:n_steps]
        if len(seg) < n_steps:
            continue
        X.append(seg.to_numpy(np.float32))
        y.append(CLASS_TO_ID[r["label"]])
        meta.append((r["label"], r["severity"], r["file"]))
    return (np.stack(X), np.array(y),
            pd.DataFrame(meta, columns=["label", "severity", "file"]))


class ChannelScaler:
    """Per-channel standardisation (study implementation)."""

    def fit(self, X):
        self.mu = X.mean(axis=(0, 1))
        self.sd = X.std(axis=(0, 1))
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    @classmethod
    def from_params(cls, mu, sd):
        obj = cls()
        obj.mu = np.asarray(mu)
        obj.sd = np.asarray(sd)
        return obj


class LogHeavy:
    """log10(1 + x/s) on heavy-tailed non-negative channels (study impl.)."""

    def fit(self, X, ratio=1e3):
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        lo, med, hi = flat.min(0), np.median(np.abs(flat), 0), np.abs(flat).max(0)
        self.idx = np.where((lo >= 0) & (hi > ratio * np.maximum(med, 1e-12)) & (hi > 0))[0]
        base = np.median(np.abs(X[:, 0, :]), 0)
        s = np.where(base[self.idx] > 0, base[self.idx], 1e-3 * hi[self.idx])
        self.s = np.maximum(s, 1e-9)
        return self

    def transform(self, X):
        X = X.copy()
        if len(self.idx):
            X[:, :, self.idx] = np.log10(1.0 + np.maximum(X[:, :, self.idx], 0) / self.s)
        return X.astype(np.float32)

    @classmethod
    def from_params(cls, idx, s):
        obj = cls()
        obj.idx = np.asarray(idx)
        obj.s = np.asarray(s)
        return obj


def fit_preprocessing(X_train_raw):
    """Fit LogHeavy + ChannelScaler on raw training windows (study order)."""
    lg = LogHeavy().fit(X_train_raw)
    sc = ChannelScaler().fit(lg.transform(X_train_raw))
    return lg, sc


def apply_preprocessing(X_raw, lg, sc):
    return np.clip(sc.transform(lg.transform(X_raw)), -CLIP, CLIP)


def summary_features(X):
    """Package summary features: per-variable [last, mean, slope] -> (n, 3*C).

    Used by the LogReg+ safe-mode fallback model."""
    t = np.arange(X.shape[1], dtype=np.float32)
    t = (t - t.mean())[None, :, None]
    xm = X.mean(axis=1, keepdims=True)
    slope = (t * (X - xm)).sum(axis=1) / max(float((t ** 2).sum()), 1e-8)
    return np.concatenate([X[:, -1, :], xm[:, 0, :], slope], axis=1)


def macro_f1(y_true, y_pred, n_classes=len(CLASSES)):
    """Macro-F1 over the fixed class set (no sklearn dependency)."""
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def splits_summary(runs):
    counts = {}
    for r in runs:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return counts


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
