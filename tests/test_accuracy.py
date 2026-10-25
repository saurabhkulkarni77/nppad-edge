"""Edge-bundle accuracy on the held-out test windows, against the
full-precision study reference (MultiRocket+HYDRA, t0=100, W=60:
test macro-F1 0.9838). The delta is reported, not hidden."""
import json
from pathlib import Path

import numpy as np
import pytest

from nppad_edge.bundle import load_bundle
from nppad_edge.data import macro_f1
from nppad_edge.runtime import load_test_windows

REFERENCE_W60 = 0.9838


def _score(model, data_root, W):
    X, y, _ = load_test_windows(data_root, W)
    pred = np.array([int(np.argmax(model.predict_proba(X[i].astype(np.float32))))
                     for i in range(len(y))])
    return macro_f1(y, pred)


def test_primary_accuracy_w60(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    f1 = _score(primary, data_root, 60)
    print(f"\nedge bundle W=60 macro-F1 = {f1:.4f} "
          f"(study reference {REFERENCE_W60}, delta {f1 - REFERENCE_W60:+.4f})")
    assert f1 >= 0.95, f"edge bundle macro-F1 {f1:.4f} too far below reference"
    # consistency with the training run's own end-to-end measurement
    metrics_file = Path(__file__).parent.parent / "artifacts" / "train_metrics.json"
    if metrics_file.exists():
        recorded = json.loads(metrics_file.read_text())["W60"]["edge_bundle_test_macro_F1"]
        assert f1 == pytest.approx(recorded, abs=1e-6)


def test_fallback_accuracy_w60(bundle_dir, data_root):
    _, fallback = load_bundle(bundle_dir, 60, verify=True)
    f1 = _score(fallback, data_root, 60)
    print(f"\nfallback (safe mode) W=60 macro-F1 = {f1:.4f}")
    assert f1 >= 0.90, "safe-mode fallback should remain a useful classifier"


def test_primary_accuracy_w120(bundle_dir, data_root):
    if not (Path(bundle_dir) / "primary_w120.npz").exists():
        pytest.skip("W=120 model not in bundle")
    primary, _ = load_bundle(bundle_dir, 120, verify=True)
    f1 = _score(primary, data_root, 120)
    print(f"\nedge bundle W=120 macro-F1 = {f1:.4f} (study reference 0.9876)")
    assert f1 >= 0.95
