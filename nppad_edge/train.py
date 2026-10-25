"""Ground-side training + export of the NPPAD-Edge uplink bundle.

Run with the TRAINING environment (aeon + scikit-learn installed):

    python -m nppad_edge.train --data-root DIR --bundle bundle

For each window size W it:
  1. rebuilds the study's exact dataset (splits, 88 realistic channels,
     LogHeavy -> ChannelScaler -> clip, fitted on train only) and pads
     scaled windows to 12 steps (the study's fix for short series);
  2. fits aeon's MultiRocket + HYDRA transforms (the study's models)
     and caches the features + extracted kernel parameters;
  3. trains a standardised linear head — logistic regression and ridge
     candidates, selected on validation macro-F1 only (the study
     protocol; the study's own classifier used a ridge head) — and the
     LogReg+ safe-mode fallback on summary features;
  4. verifies the pure-NumPy edge transforms in features.py reproduce
     aeon's features to numerical tolerance;
  5. exports a compact uplink bundle (float16 head / kernel weights,
     SHA-256 manifest, model card) and records honest metrics.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import data as D
from .bundle import load_bundle, write_manifest
from .features import HydraFeatures, MultiRocketFeatures, SparseScaler, pad_windows

REFERENCE_STUDY = {60: 0.9838, 120: 0.9876}   # MultiRocketHydra, t0=100
REFERENCE_LOGREGPLUS = {60: 0.9675, 120: 0.9677}
CACHE_DIR = Path("artifacts")


def _to_aeon(X_scaled):
    """(n, T, C) scaled windows -> padded (n, C, T) float32 for aeon."""
    Xp = pad_windows(X_scaled.transpose(0, 2, 1))
    return np.ascontiguousarray(Xp, dtype=np.float32)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _extract_multirocket(mr):
    from aeon.transformations.collection.convolution_based._multirocket import (
        MultiRocket as _MR)
    indices = np.asarray(_MR._indices, dtype=np.int64)
    return indices, mr.parameter, mr.parameter1


def _extract_hydra(hy):
    inner = hy._hydra
    W = inner.W.detach().cpu().numpy()[:, :, :, 0, :].astype(np.float32)
    idx = np.stack([t.cpu().numpy() for t in inner.idx]).astype(np.int64)
    dil = inner.dilations.cpu().numpy().astype(np.int64)
    pad = inner.paddings.cpu().numpy().astype(np.int64)
    return W, idx, dil, pad, int(hy.n_groups), int(hy.n_kernels)


def compute_features(runs, cols, W, n_kernels=6250, seed=2026):
    """aeon features + extracted parameters for one W (cached)."""
    cache = CACHE_DIR / f"feature_cache_w{W}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    from aeon.transformations.collection.convolution_based import (
        HydraTransformer, MultiRocket)

    Xtr_r, ytr, _ = D.build_windows(runs, "train", W, cols, t0=100.0)
    Xva_r, yva, _ = D.build_windows(runs, "val", W, cols, t0=100.0)
    Xte_r, yte, _ = D.build_windows(runs, "test", W, cols, t0=100.0)
    lg, sc = D.fit_preprocessing(Xtr_r)
    pre = lambda X: D.apply_preprocessing(X, lg, sc)          # (n, T, C)
    Xtr, Xva, Xte = pre(Xtr_r), pre(Xva_r), pre(Xte_r)
    Atr, Ava, Ate = _to_aeon(Xtr), _to_aeon(Xva), _to_aeon(Xte)

    mr = MultiRocket(n_kernels=n_kernels, random_state=seed)
    Mtr = _np(mr.fit_transform(Atr, ytr)).astype(np.float32)
    Mva = _np(mr.transform(Ava)).astype(np.float32)
    Mte = _np(mr.transform(Ate)).astype(np.float32)
    hy = HydraTransformer(random_state=seed)
    Htr = _np(hy.fit_transform(Atr, ytr)).astype(np.float32)
    Hva = _np(hy.transform(Ava)).astype(np.float32)
    Hte = _np(hy.transform(Ate)).astype(np.float32)

    indices, par, par1 = _extract_multirocket(mr)
    hW, hidx, hdil, hpad, hg, hk = _extract_hydra(hy)
    out = dict(Mtr=Mtr, Mva=Mva, Mte=Mte, Htr=Htr, Hva=Hva, Hte=Hte,
               ytr=ytr, yva=yva, yte=yte, Xte_raw=Xte_r,
               indices=indices, par=par, par1=par1,
               hW=hW, hidx=hidx, hdil=hdil, hpad=hpad, hg=hg, hk=hk,
               log_idx=lg.idx, log_s=lg.s, sc_mu=sc.mu, sc_sd=sc.sd,
               Str=D.summary_features(Xtr), Sva=D.summary_features(Xva),
               Ste=D.summary_features(Xte))
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def _candidates():
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    cands = []
    for C in (0.003, 0.01, 0.03, 0.1, 0.3):
        cands.append((f"logreg_C{C}",
                      LogisticRegression(C=C, max_iter=3000, solver="lbfgs")))
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        cands.append((f"ridge_a{a}", RidgeClassifier(alpha=a)))
    return cands


def _select_head(Ztr, ytr, Zva, yva):
    """Train all candidates; select on validation macro-F1 only."""
    scores = {}
    best = None
    for name, clf in _candidates():
        clf.fit(Ztr, ytr)
        vf1 = D.macro_f1(yva, clf.predict(Zva))
        scores[name] = float(vf1)
        if best is None or vf1 > best[2]:
            best = (name, clf, vf1)
    return best, scores


def build_for_W(fc, W, bundle_dir):
    from sklearn.preprocessing import StandardScaler

    t_start = time.time()
    ytr, yva, yte = fc["ytr"], fc["yva"], fc["yte"]

    # --- verify the NumPy edge transforms against aeon ---
    mr_np = MultiRocketFeatures(fc["indices"], fc["par"], fc["par1"])
    hy_np = HydraFeatures(fc["hW"], fc["hidx"], fc["hdil"], fc["hpad"],
                          fc["hg"], fc["hk"])
    # aeon-space test windows, padded, for a like-for-like check
    lg = D.LogHeavy.from_params(fc["log_idx"], fc["log_s"])
    sc = D.ChannelScaler.from_params(fc["sc_mu"], fc["sc_sd"])
    Ate = _to_aeon(D.apply_preprocessing(fc["Xte_raw"], lg, sc))
    chk = min(24, len(Ate))
    diff_mr = np.abs(mr_np.transform(Ate[:chk]) - fc["Mte"][:chk]).max()
    diff_hy = np.abs(hy_np.transform(Ate[:chk]) - fc["Hte"][:chk]).max()

    # --- scalers + head (study structure) ---
    feat_scaler = StandardScaler().fit(fc["Mtr"])
    sparse = SparseScaler().fit(fc["Htr"])
    Z = lambda M, H: np.concatenate(
        [feat_scaler.transform(M), sparse.transform(H)], axis=1)
    Ztr, Zva, Zte = Z(fc["Mtr"], fc["Htr"]), Z(fc["Mva"], fc["Hva"]), Z(fc["Mte"], fc["Hte"])
    (head_name, clf, val_f1), cand_scores = _select_head(Ztr, ytr, Zva, yva)
    test_f1_head = D.macro_f1(yte, clf.predict(Zte))

    # --- safe-mode fallback: LogReg+ on summary features ---
    sum_scaler = StandardScaler().fit(fc["Str"])
    from sklearn.linear_model import LogisticRegression
    fbest, fscores = None, {}
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        fclf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        fclf.fit(sum_scaler.transform(fc["Str"]), ytr)
        vf1 = D.macro_f1(yva, fclf.predict(sum_scaler.transform(fc["Sva"])))
        fscores[f"logreg_C{C}"] = float(vf1)
        if fbest is None or vf1 > fbest[2]:
            fbest = (f"C={C}", fclf, vf1)
    _, fclf, fval_f1 = fbest
    test_f1_fallback = D.macro_f1(yte, fclf.predict(sum_scaler.transform(fc["Ste"])))

    # --- export (float16 for the big weight arrays) ---
    par, par1 = fc["par"], fc["par1"]
    pre_kw = dict(
        pre_log_idx=fc["log_idx"].astype(np.int64),
        pre_log_s=fc["log_s"].astype(np.float32),
        pre_mu=fc["sc_mu"].astype(np.float32),
        pre_sd=fc["sc_sd"].astype(np.float32),
        classes=np.array(D.CLASSES))
    np.savez_compressed(
        bundle_dir / f"primary_w{W}.npz",
        mr_indices=fc["indices"],
        mr0_nchan=par[0], mr0_chan=par[1], mr0_dil=par[2], mr0_nfpd=par[3],
        mr0_bias=par[4].astype(np.float32),
        mr1_nchan=par1[0], mr1_chan=par1[1], mr1_dil=par1[2],
        mr1_nfpd=par1[3], mr1_bias=par1[4].astype(np.float32),
        hy_W=fc["hW"].astype(np.float16), hy_idx=fc["hidx"],
        hy_dil=fc["hdil"], hy_pad=fc["hpad"],
        hy_g=np.int64(fc["hg"]), hy_k=np.int64(fc["hk"]),
        feat_mu=feat_scaler.mean_.astype(np.float32),
        feat_sd=feat_scaler.scale_.astype(np.float32),
        hy_mu=sparse.mu.astype(np.float32),
        hy_sigma=sparse.sigma.astype(np.float32),
        head_coef=clf.coef_.T.astype(np.float16),          # (n_feat, 12)
        head_intercept=clf.intercept_.astype(np.float32),
        **pre_kw)
    np.savez_compressed(
        bundle_dir / f"fallback_w{W}.npz",
        sum_mu=sum_scaler.mean_.astype(np.float32),
        sum_sd=sum_scaler.scale_.astype(np.float32),
        head_coef=fclf.coef_.T.astype(np.float16),
        head_intercept=fclf.intercept_.astype(np.float32),
        **pre_kw)

    return {
        "W": W, "n_windows": {"train": int(len(ytr)), "val": int(len(yva)),
                              "test": int(len(yte))},
        "features": {"multirocket": int(fc["Mtr"].shape[1]),
                     "hydra": int(fc["Htr"].shape[1])},
        "numpy_vs_aeon_max_abs_diff": {"multirocket": float(diff_mr),
                                       "hydra": float(diff_hy)},
        "head": {"selected": head_name, "val_macro_F1": float(val_f1),
                 "test_macro_F1_float32": float(test_f1_head),
                 "candidate_val_scores": cand_scores},
        "fallback": {"selected": fbest[0], "val_macro_F1": float(fval_f1),
                     "test_macro_F1": float(test_f1_fallback),
                     "candidate_val_scores": fscores},
        "reference_study_macro_F1": REFERENCE_STUDY.get(W),
        "reference_logregplus_macro_F1": REFERENCE_LOGREGPLUS.get(W),
        "seconds": round(time.time() - t_start, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nppad-edge-train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--windows", default="60,120")
    ap.add_argument("--n-kernels", type=int, default=6250,
                    help="MultiRocket feature budget (study default 6250)")
    args = ap.parse_args(argv)

    bundle_dir = Path(args.bundle)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    runs = D.assign_splits(D.load_runs(args.data_root))
    _, cols = D.variable_sets(runs)
    assert len(cols) == 88, f"expected 88 realistic channels, got {len(cols)}"

    metrics = {}
    for W in [int(w) for w in args.windows.split(",")]:
        fc = compute_features(runs, cols, W, n_kernels=args.n_kernels)
        metrics[f"W{W}"] = build_for_W(fc, W, bundle_dir)
        print(f"W={W} done: head={metrics[f'W{W}']['head']['selected']} "
              f"test={metrics[f'W{W}']['head']['test_macro_F1_float32']:.4f}",
              flush=True)

    config = {"task": "NPPAD accident classification", "t0_s": 100.0,
              "channels": 88, "classes": D.CLASSES,
              "windows_s": [int(w) for w in args.windows.split(",")],
              "seed": 2026, "multirocket_n_kernels": args.n_kernels}
    card = {"bundle": "NPPAD-Edge uplink bundle", "config": config,
            "metrics": metrics}
    (bundle_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    write_manifest(bundle_dir, config)

    # --- end-to-end check of the exported bundle via the edge runtime ---
    for W in [int(w) for w in args.windows.split(",")]:
        primary, _ = load_bundle(bundle_dir, W, verify=True)
        Xte_r, yte, _ = D.build_windows(runs, "test", W, cols, t0=100.0)
        pred = np.array([int(np.argmax(primary.predict_proba(Xte_r[i])))
                         for i in range(len(yte))])
        metrics[f"W{W}"]["edge_bundle_test_macro_F1"] = D.macro_f1(yte, pred)
        print(f"W={W} edge bundle test macro-F1: "
              f"{metrics[f'W{W}']['edge_bundle_test_macro_F1']:.4f}", flush=True)
    (bundle_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    write_manifest(bundle_dir, config)   # re-hash after card update

    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
