"""Measured evaluation of the NASA-technique modules (harden,
quantize, checkpoint, sensitivity) — prints a JSON report.

Ground-side tool (not flown): it computes standardised features once
and measures each technique against the float baseline on the same
held-out test windows.

Calibration provenance is stated in the output, never hidden:
  * with ``--cache`` (study feature-cache pickle): INT8 scales and
    sensitivity scores are calibrated on the TRAIN split and evaluated
    on the TEST split — the flight-honest arrangement;
  * without: features are computed from the replay windows under
    ``--data-root``; the first 80 windows calibrate and the remaining
    160 evaluate (a weaker split, labelled as such).

    python -m nppad_edge.nasa_bench --bundle bundle --W 60 \
        --data-root DIR [--cache artifacts/feature_cache_w60.pkl]
"""
from __future__ import annotations

import argparse
import io
import json
import resource
import time

import numpy as np

from .bundle import load_bundle
from .checkpoint import CheckpointManager, GoldenRestorer
from .data import macro_f1
from .harden import VotingClassifier
from .quantize import evaluate as quantize_evaluate
from .runtime import load_test_windows
from .sensitivity import SelectiveProtection, validate_by_campaign


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _feats(model, X, n):
    return np.stack([model.features(X[i].astype(np.float32))
                     for i in range(n)])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nppad-edge-nasa-bench")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--W", type=int, default=60)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--n-latency", type=int, default=40)
    args = ap.parse_args(argv)
    W = args.W
    primary, _ = load_bundle(args.bundle, W, verify=True)
    X, y, _ = load_test_windows(args.data_root, W)
    n = len(y)
    report = {"W": W, "n_test_windows": n}

    # ---- features: cache (train-calib) or computed (split-calib) ----
    if args.cache:
        import pickle
        from pathlib import Path

        from .features import SparseScaler

        with open(args.cache, "rb") as fh:
            d = pickle.load(fh)
        z = np.load(Path(args.bundle) / f"primary_w{W}.npz")
        sp = SparseScaler.from_params(z["hy_mu"], z["hy_sigma"])
        mu, sd = primary._arrays["feat_mu"], primary._arrays["feat_sd"]

        def std(M, H):
            return np.concatenate([(M - mu) / sd, sp.transform(H)],
                                  axis=1).astype(np.float32)

        calib_feats = std(d["Mtr"], d["Htr"])
        eval_feats, y_eval = std(d["Mte"], d["Hte"]), d["yte"]
        calib_src = f"train split of {args.cache} ({len(calib_feats)} windows)"
    else:
        all_feats = _feats(primary, X, n)
        # seeded spread: replay order is grouped by class, so a
        # contiguous calibration slice would miss most classes
        perm = np.random.default_rng(0).permutation(n)
        calib_feats = all_feats[perm[:80]]
        eval_feats, y_eval = all_feats[perm[80:]], y[perm[80:]]
        calib_src = "80 of the replay windows (seeded spread; weaker split, no train data)"
    report["calibration_source"] = calib_src

    # ---- (a) TMR voting ------------------------------------------------
    voter = VotingClassifier(primary)
    rss_after_voter = _rss_mb()
    lat_f, lat_t = [], []
    agree = 0
    for i in range(args.n_latency):
        w = X[i].astype(np.float32)
        t0 = time.perf_counter(); pf = primary.predict_proba(w)
        lat_f.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); pt = voter.predict_proba(w)
        lat_t.append((time.perf_counter() - t0) * 1e3)
        agree += int(np.argmax(pf) == np.argmax(pt))
    clean_proba = voter.predict_proba(X[0].astype(np.float32))
    voter.corrupt_replica_for_test(1, 12345, bit=30)
    masked_proba = voter.predict_proba(X[0].astype(np.float32))
    report["tmr_head"] = {
        "latency_ms_mean_float": float(np.mean(lat_f)),
        "latency_ms_mean_tmr": float(np.mean(lat_t)),
        "argmax_agreement_clean": agree / args.n_latency,
        "proba_max_abs_diff_after_1_replica_bitflip":
            float(np.max(np.abs(clean_proba - masked_proba))),
        "replica_check_after_flip": voter.check_replicas(),
        "extra_head_bytes_2_extra_replicas":
            int(2 * (primary._arrays["head_coef"].nbytes
                     + primary._arrays["head_intercept"].nbytes)),
    }

    # ---- (b) INT8 head ---------------------------------------------------
    report["int8_head"] = quantize_evaluate(
        args.bundle, W, calib_feats, eval_feats, y_eval)

    # ---- (c) checkpoint / golden restore ---------------------------------
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        cm = CheckpointManager(_P(td) / "ckpt.json")
        cm.save({"mode": "primary", "seq": 42})
        ckpt_bytes = (_P(td) / "ckpt.json").stat().st_size
        state = cm.load()
    restorer = GoldenRestorer(args.bundle, W)
    before = primary.predict_proba(X[0].astype(np.float32))
    primary.corrupt_for_test("head_coef", 999)
    t0 = time.perf_counter()
    ok = restorer.restore(primary)
    restore_ms = (time.perf_counter() - t0) * 1e3
    after = primary.predict_proba(X[0].astype(np.float32))
    report["checkpoint_restore"] = {
        "checkpoint_file_bytes": ckpt_bytes,
        "state_roundtrip_ok": state == {"mode": "primary", "seq": 42},
        "restore_ok": bool(ok),
        "restore_ms": restore_ms,
        "predictions_identical_after_restore":
            bool(np.array_equal(before, after)),
    }

    # ---- (d) selective protection -----------------------------------------
    camp1 = validate_by_campaign(primary, eval_feats, y_eval,
                                 trials=16, flips_per_trial=1)
    camp32 = validate_by_campaign(primary, eval_feats, y_eval,
                                  trials=16, flips_per_trial=32)
    prot = SelectiveProtection.from_calibration(primary, calib_feats,
                                                frac=0.05)
    prot_bytes = prot.golden_bytes
    full_head_bytes = int(primary._arrays["head_coef"].nbytes
                          + primary._arrays["head_intercept"].nbytes)
    # scrub demo: corrupt one protected entry, scrub, verify
    idx = int(prot.indices[0])
    orig = primary._arrays["head_coef"].ravel()[idx]
    u = primary._arrays["head_coef"].reshape(-1).view(np.uint32)
    u[idx] = u[idx] ^ np.uint32(1 << 30)
    repaired = prot.scrub()
    report["selective_protection"] = {
        "campaign_single_flip": camp1,
        "campaign_32_flips": camp32,
        "protected_frac": 0.05,
        "protected_entries": prot.protected_entries,
        "golden_bytes": prot_bytes,
        "full_head_golden_bytes": full_head_bytes,
        "golden_bytes_ratio": prot_bytes / full_head_bytes,
        "scrub_repaired_corrupted_protected_entry": repaired,
        "value_restored": bool(primary._arrays["head_coef"].ravel()[idx]
                               == orig),
    }
    report["peak_rss_mb"] = _rss_mb()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
