"""Checkpoint/restart + golden restore (checkpoint.py): atomic state
snapshots, watchdog heartbeat staleness, and in-place repair of
corrupted weights from the verified bundle — the payload keeps
flying the primary model instead of degrading to safe mode."""
import io
import json
import time

import numpy as np

from nppad_edge.bundle import load_bundle
from nppad_edge.checkpoint import (CheckpointManager, GoldenRestorer,
                                   HeartbeatMonitor)
from nppad_edge.runtime import EdgeRuntime, load_test_windows


def test_checkpoint_roundtrip(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.json")
    assert cm.load() is None
    state = {"mode": "primary", "seq": 17, "events": []}
    cm.save(state)
    assert cm.load() == state
    # atomic write: no torn temp file left behind
    assert not (tmp_path / "ckpt.json.tmp").exists()


def test_heartbeat_staleness():
    mon = HeartbeatMonitor(stale_after_s=5.0)
    mon.beat("checker", when=100.0)
    mon.beat("watchdog", when=109.0)
    assert mon.stale_sources(now=110.0) == ["checker"]
    assert mon.last_beat("watchdog") == 109.0


def test_golden_restore_repairs_weights(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    X, y, _ = load_test_windows(data_root, 60)
    w = X[0].astype(np.float32)
    before = primary.predict_proba(w)
    primary.corrupt_for_test("head_coef", 999)
    assert primary.check_memory() is False
    restorer = GoldenRestorer(bundle_dir, 60)
    assert restorer.restore(primary) is True
    assert primary.check_memory() is True
    after = primary.predict_proba(w)
    assert np.array_equal(before, after)


def test_runtime_auto_restore_stays_on_primary(bundle_dir, data_root):
    X, y, _ = load_test_windows(data_root, 60)
    windows = [X[i].astype(np.float32) for i in range(4)]
    rt = EdgeRuntime(bundle_dir, W=60, check_interval_s=0.1,
                     auto_restore=True)
    tel = io.StringIO()
    with rt:
        rt.stream(windows[:2], tel)
        rt.primary.corrupt_for_test("head_coef", 999)
        time.sleep(0.6)                     # checker notices + scrub runs
        rt.stream(windows[2:], tel)
    types = [e["type"] for e in rt.events]
    assert "memory_corruption_detected" in types
    assert "model_restored_from_golden" in types
    assert rt.mode == "primary"             # repaired, NOT safe mode
    records = [json.loads(l) for l in tel.getvalue().splitlines()]
    assert all(r["model"] == rt.primary.name for r in records)
    assert [r["seq"] for r in records] == [0, 1, 2, 3]


def test_runtime_checkpoint_resume(bundle_dir, data_root, tmp_path):
    X, y, _ = load_test_windows(data_root, 60)
    windows = [X[i].astype(np.float32) for i in range(4)]
    ckpt = tmp_path / "state.json"
    rt = EdgeRuntime(bundle_dir, W=60, check_interval_s=60.0,
                     checkpoint_path=str(ckpt))
    with rt:
        rt.stream(windows[:2], io.StringIO())
    saved = CheckpointManager(ckpt).load()
    assert saved["seq"] == 2 and saved["mode"] == "primary"

    rt2 = EdgeRuntime(bundle_dir, W=60, check_interval_s=60.0)
    rt2.load_state_dict(saved)
    tel = io.StringIO()
    with rt2:
        results = rt2.stream(windows[2:], tel)
    assert [r[0] for r in results] == [2, 3]   # stream position resumed
