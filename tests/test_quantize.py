"""INT8 head (quantize.py): per-feature activation scales folded
into per-class int8 weights, exact int32 accumulation. Calibrated on
one slice of windows, evaluated on a disjoint slice; the accuracy
delta vs the float head is asserted, not assumed."""
import numpy as np

from nppad_edge.bundle import load_bundle
from nppad_edge.data import macro_f1
from nppad_edge.quantize import QuantizedHead, quantize_primary
from nppad_edge.runtime import load_test_windows

N_CALIB, N_EVAL = 40, 80


def _features(primary, data_root):
    X, y, _ = load_test_windows(data_root, 60)
    # the replay order is grouped by class, so draw a seeded spread
    # across the whole test set (a contiguous slice would cover only
    # a few classes and macro-F1 over 12 classes would be meaningless)
    idx = np.random.default_rng(0).permutation(len(y))[:N_CALIB + N_EVAL]
    F = np.stack([primary.features(X[i].astype(np.float32)) for i in idx])
    return F, y[idx]


def test_quantized_head_is_much_smaller(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    F, _ = _features(primary, data_root)
    q = quantize_primary(primary, F[:N_CALIB])
    float_bytes = (primary._arrays["head_coef"].nbytes
                   + primary._arrays["head_intercept"].nbytes)
    # int8 weights + scales must beat float32 by > 2.5x even after
    # paying for the per-feature scale vector
    assert q.head.weight_bytes < float_bytes / 2.5


def test_int8_accuracy_delta_is_small(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    F, y = _features(primary, data_root)
    calib, ev, y_ev = F[:N_CALIB], F[N_CALIB:], y[N_CALIB:]
    coef, icpt = primary._arrays["head_coef"], primary._arrays["head_intercept"]
    f1_float = macro_f1(y_ev, np.argmax(ev @ coef + icpt, axis=1))
    q = quantize_primary(primary, calib)
    f1_int8 = macro_f1(y_ev, np.argmax(q.head.logits(ev), axis=1))
    print(f"\nINT8 head: float F1 {f1_float:.4f} vs int8 F1 {f1_int8:.4f} "
          f"(delta {f1_int8 - f1_float:+.4f}) on {len(y_ev)} windows")
    assert f1_int8 >= 0.90
    assert f1_int8 >= f1_float - 0.03


def test_quantized_head_deterministic_small_case():
    rng = np.random.default_rng(0)
    coef = rng.normal(size=(50, 3)).astype(np.float32)
    icpt = rng.normal(size=3).astype(np.float32)
    feats = rng.normal(size=(7, 50)).astype(np.float32)
    scales = np.maximum(np.abs(feats).max(axis=0), 1e-4)
    qh = QuantizedHead(coef, icpt, scales)
    assert qh.Wq.dtype == np.int8
    lg = qh.logits(feats)
    assert lg.shape == (7, 3)
    # int8 logits track the float logits closely on well-scaled data
    ref = feats @ coef + icpt
    assert np.mean(np.abs(lg - ref)) < 0.25 * np.std(ref) + 0.05
