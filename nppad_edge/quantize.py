"""INT8-quantized inference path for the primary model's linear head.

Onboard ML processors for space — the Intel Movidius Myriad X VPU in
Ubotica's CogniSAT payload (flown with JPL on the ISS in 2022 and on
CogniSAT-6 for JPL's Dynamic Targeting tests) is the current example —
execute neural inference in low-precision integer arithmetic: NASA's
SpaceCube overview (GSFC Code 587, January 2020) quotes commercial
devices at ~148 GOPS/W for 8-bit integer vs ~60 GOPS/W for float32.
The model that flies is therefore an INT8 model, adapted from the
float model on the ground (in the Spinoff account of CogniSAT, "I
would send them a JPL model and they would adapt it to their
processor" — Emily Dunkel, JPL).

This module performs that adaptation for NPPAD-Edge's linear head:

* Calibration: per-feature activation scales ``s_j`` are estimated
  from a calibration set of standardised features (the 99.9th
  percentile of |f_j|, floored) — use TRAIN features for flight
  calibration, never the evaluation set.
* The activation scales are folded into the weights,
  ``W'_jk = W_jk * s_j``, and ``W'`` is quantised per output class
  (per-column scale) to int8.
* Inference: features are quantised to int8 with the same scales, the
  matmul accumulates exactly in int32, and logits are recovered as
  ``scale_k * (q_f . q_W)_k + intercept_k``.

Honest scope: the MultiRocket+HYDRA feature extraction stays float —
on a flight VPU the convolutions would be integer too, but here the
measurable wins are head storage (4x fewer weight bytes than float32)
and exact integer accumulation of the 50,752 x 12 matmul. On this
Earth CPU NumPy's int8 path is *not* faster than float32 (it upcasts
to int32 for the dot product); the latency delta is measured and
reported as-is rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .bundle import load_bundle
from .data import macro_f1


def calibrate_scales(feats: np.ndarray, percentile: float = 99.9,
                     floor: float = 1e-4) -> np.ndarray:
    """Per-feature activation scales from a calibration feature set.

    feats: (n, D) standardised features (e.g. train-split features).
    Returns (D,) float32 scales s_j = P99.9(|f_j|), floored so constant
    or near-zero features still quantise safely."""
    a = np.abs(np.asarray(feats, dtype=np.float32))
    s = np.percentile(a, percentile, axis=0).astype(np.float32)
    return np.maximum(s, np.float32(floor))


class QuantizedHead:
    """INT8 linear head with folded activation scales."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray,
                 feat_scale: np.ndarray):
        coef = np.asarray(coef, dtype=np.float64)
        self.feat_scale = np.asarray(feat_scale, dtype=np.float32)
        folded = coef * self.feat_scale[:, None].astype(np.float64)
        self.weight_scale = (np.max(np.abs(folded), axis=0)
                             / 127.0).astype(np.float32)      # (K,)
        q = np.round(folded / self.weight_scale[None, :])
        self.Wq = np.clip(q, -127, 127).astype(np.int8)        # (D, K)
        self.intercept = np.asarray(intercept, dtype=np.float32)

    # -- sizes ----------------------------------------------------------
    @property
    def weight_bytes(self) -> int:
        """Bytes of the quantised parameters (weights + scales)."""
        return int(self.Wq.nbytes + self.feat_scale.nbytes
                   + self.weight_scale.nbytes + self.intercept.nbytes)

    # -- inference --------------------------------------------------------
    def logits(self, feats: np.ndarray) -> np.ndarray:
        """feats: (D,) or (n, D) standardised float features."""
        F = np.asarray(feats, dtype=np.float32)
        single = F.ndim == 1
        F2 = F[None, :] if single else F
        q = np.clip(np.round(F2 / self.feat_scale), -127, 127)
        q = q.astype(np.int8)
        acc = q.astype(np.int32) @ self.Wq.astype(np.int32)    # exact
        out = acc * self.weight_scale[None, :] + self.intercept[None, :]
        return out[0] if single else out

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        lg = self.logits(feats)
        if lg.ndim == 1:
            lg = lg - lg.max()
            p = np.exp(lg)
            return p / p.sum()
        lg = lg - lg.max(axis=1, keepdims=True)
        p = np.exp(lg)
        return p / p.sum(axis=1, keepdims=True)


class QuantizedPrimaryModel:
    """Primary model whose head runs the INT8 path.

    Feature extraction is delegated to the wrapped float model; only
    the head matmul is quantised. Same predict_proba interface."""

    name = "multirocket-hydra-int8head"

    def __init__(self, base_model, calib_feats: np.ndarray):
        self._base = base_model
        self.classes = list(base_model.classes)
        scales = calibrate_scales(calib_feats)
        self.head = QuantizedHead(base_model._arrays["head_coef"],
                                  base_model._arrays["head_intercept"],
                                  scales)

    def features(self, window_raw):
        return self._base.features(window_raw)

    def predict_proba(self, window_raw):
        return self.head.predict_proba(self.features(window_raw))


def quantize_primary(base_model, calib_feats: np.ndarray):
    return QuantizedPrimaryModel(base_model, calib_feats)


def evaluate(bundle_dir, W, feats_calib, feats_eval, y_eval):
    """Accuracy/size report: float head vs INT8 head on the same
    standardised features. Returns a dict of measured numbers."""
    primary, _ = load_bundle(bundle_dir, W, verify=True)
    coef = primary._arrays["head_coef"]
    intercept = primary._arrays["head_intercept"]

    def float_pred(F):
        return np.argmax(F @ coef + intercept, axis=1)

    t0 = time.perf_counter()
    pred_f = float_pred(feats_eval)
    float_ms = (time.perf_counter() - t0) * 1e3 / len(feats_eval)

    qmodel = quantize_primary(primary, feats_calib)
    t0 = time.perf_counter()
    pred_q = np.argmax(qmodel.head.logits(feats_eval), axis=1)
    int8_ms = (time.perf_counter() - t0) * 1e3 / len(feats_eval)

    return {
        "W": int(W),
        "n_eval": int(len(y_eval)),
        "float_macro_F1": macro_f1(y_eval, pred_f),
        "int8_macro_F1": macro_f1(y_eval, pred_q),
        "delta_F1": macro_f1(y_eval, pred_q) - macro_f1(y_eval, pred_f),
        "agreement_float_vs_int8": float(np.mean(pred_f == pred_q)),
        "head_bytes_float32": int(coef.nbytes + intercept.nbytes),
        "head_bytes_int8_total": qmodel.head.weight_bytes,
        "head_ms_per_window_float_batchamortised": float_ms,
        "head_ms_per_window_int8_batchamortised": int8_ms,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nppad-edge-quantize")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--W", type=int, default=60)
    ap.add_argument("--cache", required=True,
                    help="study feature-cache pickle (train split is "
                         "used for calibration, test split to evaluate)")
    args = ap.parse_args(argv)

    import pickle

    from .features import SparseScaler

    primary, _ = load_bundle(args.bundle, args.W, verify=True)
    with open(args.cache, "rb") as fh:
        d = pickle.load(fh)
    # HYDRA sparse-scaler params are stored in the bundle alongside
    # the live arrays; recover them from a fresh npz read.
    from pathlib import Path

    z = np.load(Path(args.bundle) / f"primary_w{args.W}.npz")
    sp = SparseScaler.from_params(z["hy_mu"], z["hy_sigma"])
    mu, sd = primary._arrays["feat_mu"], primary._arrays["feat_sd"]

    def std(M, H):
        return np.concatenate([(M - mu) / sd, sp.transform(H)],
                              axis=1).astype(np.float32)

    calib = std(d["Mtr"], d["Htr"])
    feats_eval, y_eval = std(d["Mte"], d["Hte"]), d["yte"]
    res = evaluate(args.bundle, args.W, calib, feats_eval, y_eval)
    res["calibration"] = (f"train split of {args.cache} "
                          f"({len(calib)} windows), P99.9 |f| scales")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
