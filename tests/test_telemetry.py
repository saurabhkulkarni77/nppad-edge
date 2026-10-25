"""Downlink telemetry: only classification events leave the
processor — timestamp, class, confidence, model, mode. Never raw
sensor data."""
import io
import json

import numpy as np

from nppad_edge.data import CLASSES
from nppad_edge.runtime import EdgeRuntime, load_test_windows

ALLOWED_KEYS = {"seq", "timestamp", "class", "confidence", "model", "mode"}


def test_telemetry_format(bundle_dir, data_root):
    X, y, _ = load_test_windows(data_root, 60)
    windows = [X[i].astype(np.float32) for i in range(5)]
    rt = EdgeRuntime(bundle_dir, W=60, check_interval_s=60.0)
    tel = io.StringIO()
    with rt:
        rt.stream(windows, tel)
    lines = tel.getvalue().splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert set(rec) == ALLOWED_KEYS, f"unexpected telemetry keys: {set(rec)}"
        assert rec["seq"] == i
        assert rec["class"] in CLASSES
        assert 0.0 < rec["confidence"] <= 1.0
        assert rec["mode"] == "primary"
        assert "T" in rec["timestamp"]          # ISO-8601


def test_downlink_is_tiny_vs_raw(bundle_dir, data_root):
    """Process-at-capture: the downlink for N windows must be a small
    fraction of the raw sensor volume for the same windows."""
    X, y, _ = load_test_windows(data_root, 60)
    n = 20
    windows = [X[i].astype(np.float32) for i in range(n)]
    rt = EdgeRuntime(bundle_dir, W=60, check_interval_s=60.0)
    tel = io.StringIO()
    with rt:
        rt.stream(windows, tel)
    raw_bytes = sum(w.nbytes for w in windows)
    tel_bytes = len(tel.getvalue().encode())
    print(f"\ndownlink {tel_bytes} B vs raw {raw_bytes} B "
          f"({100.0 * tel_bytes / raw_bytes:.2f}%)")
    # ~150 B of JSON per event vs 2112 B per raw window: an order of
    # magnitude reduction even against the already-windowed data (and
    # far more against the full raw run records it was cut from).
    assert tel_bytes < 0.10 * raw_bytes
