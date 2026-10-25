"""NPPAD-Edge runtime: the "spacecraft-side" application.

Behaviour, mirroring intelligent processing at the edge:
  * The uplink bundle is SHA-256-verified before anything is loaded;
    a corrupted bundle is refused, never flown.
  * Sensor windows are processed at the point of capture. Only
    classification events are "downlinked" (telemetry): timestamp,
    predicted class, confidence, and operating mode. Raw sensor data
    never leaves the processor.
  * A periodic in-memory checker re-hashes the live weight arrays
    (the software memory checker). If corruption is detected — e.g.
    a radiation-induced bit flip — the watchdog switches the payload
    to the small safe-mode fallback model and flags it in telemetry.
  * A compute budget is enforced: single-threaded execution, a memory
    ceiling, and per-inference latency accounting. Persistent budget
    breaches also trigger safe mode.

Run as:  python -m nppad_edge.runtime --bundle bundle --data-root DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# --- single-thread discipline: set BEFORE numpy is imported ---------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from .bundle import IntegrityError, load_bundle  # noqa: E402
from .checkpoint import (CheckpointManager, GoldenRestorer,  # noqa: E402
                         HeartbeatMonitor)
from .data import (CLASSES, assign_splits, build_windows, load_runs,  # noqa: E402
                   variable_sets)
from .harden import VotingClassifier  # noqa: E402


def _rss_mb() -> float:
    """Current resident set size (Linux /proc)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


class MemoryChecker(threading.Thread):
    """Periodic in-memory integrity checker (software memory checker).

    Recomputes the digest of the live model arrays every
    ``interval_s`` seconds. On mismatch it records a corruption event
    and notifies the watchdog callback."""

    def __init__(self, model, interval_s: float, on_corruption,
                 on_beat=None):
        super().__init__(daemon=True, name="memory-checker")
        self._model = model
        self._interval = interval_s
        self._on_corruption = on_corruption
        self._on_beat = on_beat
        self._stop = threading.Event()
        self.events = []

    def run(self):
        while not self._stop.wait(self._interval):
            if self._on_beat is not None:
                self._on_beat()
            if not self._model.check_memory():
                ev = {"type": "memory_corruption_detected",
                      "at": datetime.now(timezone.utc).isoformat()}
                self.events.append(ev)
                self._on_corruption(ev)
                return

    def stop(self):
        self._stop.set()


class EdgeRuntime:
    """Streaming classifier with watchdog / safe-mode supervision."""

    def __init__(self, bundle_dir, W=60, latency_budget_ms=250.0,
                 memory_budget_mb=512.0, check_interval_s=2.0,
                 breach_limit=3, tmr_head=False, auto_restore=False,
                 checkpoint_path=None):
        self.primary, self.fallback = load_bundle(bundle_dir, W, verify=True)
        self.W = int(W)
        self.model = self.primary
        self.mode = "primary"            # or "safe" (fallback model)
        self.latency_budget_ms = float(latency_budget_ms)
        self.memory_budget_mb = float(memory_budget_mb)
        self.breach_limit = int(breach_limit)
        self._lat_breaches = 0
        self.events = []                 # watchdog / mission events
        self.latencies_ms = []
        self._seq = 0                    # mission stream position
        self._check_interval = float(check_interval_s)
        # NASA-technique options (see NASA_TECHNIQUES.md); all default
        # off so the baseline behaviour above is unchanged:
        #   tmr_head      triple-modular-redundant head voting
        #   auto_restore  scrub weights from the golden bundle and keep
        #                 flying the primary instead of safe mode
        #   checkpoint_path  persist runtime state after every window
        self._voter = VotingClassifier(self.primary) if tmr_head else None
        self._restorer = (GoldenRestorer(bundle_dir, W)
                          if auto_restore else None)
        self._checkpoint = (CheckpointManager(checkpoint_path)
                            if checkpoint_path else None)
        self.heartbeats = HeartbeatMonitor(
            stale_after_s=3.0 * self._check_interval)
        self.checker = MemoryChecker(
            self.primary, check_interval_s, self._on_corruption,
            on_beat=lambda: self.heartbeats.beat("memory-checker"))
        self._lock = threading.Lock()
        self._last_disagreement = None

    # -- watchdog -----------------------------------------------------
    def _on_corruption(self, ev):
        with self._lock:
            self.events.append(ev)
            if self._restorer is not None:
                # Configuration scrub: repair from the golden bundle.
                # Only an unrecoverable model degrades to safe mode.
                if self._restorer.restore(self.primary):
                    self.events.append({
                        "type": "model_restored_from_golden",
                        "at": datetime.now(timezone.utc).isoformat()})
                    self.checker = MemoryChecker(
                        self.primary, self._check_interval,
                        self._on_corruption,
                        on_beat=lambda: self.heartbeats.beat(
                            "memory-checker"))
                    self.checker.start()
                    return
            self._enter_safe_mode("memory corruption detected by checker")

    def _enter_safe_mode(self, reason):
        if self.mode != "safe":
            self.mode = "safe"
            self.model = self.fallback
            self.events.append({
                "type": "safe_mode_entered", "reason": reason,
                "at": datetime.now(timezone.utc).isoformat()})

    def _budget_watch(self, latency_ms):
        """Persistent latency or memory breach -> safe mode."""
        if latency_ms > self.latency_budget_ms:
            self._lat_breaches += 1
        else:
            self._lat_breaches = 0
        rss = _rss_mb()
        if rss == rss and rss > self.memory_budget_mb:   # rss not NaN
            self._enter_safe_mode(
                f"memory budget breached: RSS {rss:.0f} MB > "
                f"{self.memory_budget_mb:.0f} MB")
        elif self._lat_breaches >= self.breach_limit:
            self._enter_safe_mode(
                f"latency budget breached {self._lat_breaches}x in a row "
                f"(> {self.latency_budget_ms:.0f} ms/window)")

    # -- inference ----------------------------------------------------
    def classify_window(self, window_raw):
        """Classify one raw (T, C) window. Returns (class, confidence)."""
        t0 = time.perf_counter()
        if self.mode == "primary" and self._voter is not None:
            proba = self._voter.predict_proba(window_raw)
            self._last_disagreement = bool(
                self._voter.last_vote["disagreement"])
        else:
            proba = self.model.predict_proba(window_raw)
            self._last_disagreement = None
        dt_ms = (time.perf_counter() - t0) * 1e3
        self.latencies_ms.append(dt_ms)
        self._budget_watch(dt_ms)
        self._seq += 1
        k = int(np.argmax(proba))
        return self.model.classes[k], float(proba[k])

    # -- checkpoint / restart ------------------------------------------
    def state_dict(self) -> dict:
        """Serializable runtime state for checkpoint/restart."""
        return {
            "W": self.W,
            "mode": self.mode,
            "seq": self._seq,
            "latencies_ms_count": len(self.latencies_ms),
            "latency_ms_mean": (float(np.mean(self.latencies_ms))
                                if self.latencies_ms else None),
            "events": self.events,
        }

    def load_state_dict(self, state: dict):
        """Resume stream position / mode from a checkpoint state."""
        if int(state.get("W", self.W)) != self.W:
            raise ValueError("checkpoint W does not match runtime W")
        self._seq = int(state.get("seq", 0))
        self.events = list(state.get("events", []))
        if state.get("mode") == "safe":
            self._enter_safe_mode("restored from checkpoint in safe mode")

    def stream(self, windows, telemetry_out, max_windows=None):
        """Process windows in order, writing downlink telemetry (JSONL).

        Only classification events are written — never raw data.
        Returns the list of (seq, class, confidence) for inspection."""
        results = []
        n = len(windows) if max_windows is None else min(len(windows), max_windows)
        for i in range(n):
            seq = self._seq
            cls, conf = self.classify_window(windows[i])
            rec = {
                "seq": seq,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "class": cls,
                "confidence": round(conf, 6),
                "model": self.model.name,
                "mode": self.mode,
            }
            if self._last_disagreement is not None:
                rec["tmr_disagreement"] = self._last_disagreement
            telemetry_out.write(json.dumps(rec) + "\n")
            telemetry_out.flush()
            if self._checkpoint is not None:
                self._checkpoint.save(self.state_dict())
            results.append((seq, cls, conf))
        return results

    def __enter__(self):
        self.checker.start()
        return self

    def __exit__(self, *exc):
        self.checker.stop()
        return False


def load_test_windows(data_root, W, t0=100.0):
    """Replay source: the study's held-out test windows (raw values).

    Packaged-data path (this GitHub repo): if ``data_root`` contains a
    ``test_windows_w{W}.npz`` file — as ``demo_data/`` does — the windows
    are loaded from it. Those files were written by this exact pipeline
    from the full NPPAD CSV tree (far too large to redistribute in git)
    and are bit-identical to the CSV-derived windows; labels included.
    With a real ``Operation_csv_data`` directory the CSV path below is
    used unchanged.
    """
    pkg = Path(data_root) / f"test_windows_w{int(W)}.npz"
    if pkg.exists():
        z = np.load(pkg)
        return z["X"], z["y"], []
    runs = assign_splits(load_runs(data_root))
    _, cols = variable_sets(runs)
    X, y, meta = build_windows(runs, "test", W, cols, t0=t0)
    return X, y, meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nppad-edge",
                                 description="NPPAD-Edge runtime")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--data-root", required=True,
                    help="NPPAD Operation_csv_data directory (replay source)")
    ap.add_argument("--W", type=int, default=60)
    ap.add_argument("--t0", type=float, default=100.0)
    ap.add_argument("--telemetry", default="telemetry.jsonl")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--latency-budget-ms", type=float, default=250.0)
    ap.add_argument("--memory-budget-mb", type=float, default=512.0)
    ap.add_argument("--check-interval", type=float, default=2.0)
    ap.add_argument("--tmr-head", action="store_true",
                    help="triple-modular-redundant head voting (harden.py)")
    ap.add_argument("--auto-restore", action="store_true",
                    help="scrub corrupted weights from the golden bundle "
                         "instead of dropping to safe mode (checkpoint.py)")
    ap.add_argument("--checkpoint", default=None,
                    help="path to write runtime-state checkpoints")
    args = ap.parse_args(argv)

    try:
        rt = EdgeRuntime(args.bundle, W=args.W,
                         latency_budget_ms=args.latency_budget_ms,
                         memory_budget_mb=args.memory_budget_mb,
                         check_interval_s=args.check_interval,
                         tmr_head=args.tmr_head,
                         auto_restore=args.auto_restore,
                         checkpoint_path=args.checkpoint)
    except IntegrityError as e:
        print(f"BUNDLE REJECTED: {e}", file=sys.stderr)
        return 2

    X, y, meta = load_test_windows(args.data_root, args.W, t0=args.t0)
    windows = [X[i].astype(np.float32) for i in range(len(X))]
    with rt, open(args.telemetry, "w") as fh:
        results = rt.stream(windows, fh, max_windows=args.max_windows)

    lat = np.array(rt.latencies_ms)
    n = len(results)
    correct = sum(1 for (seq, cls, _) in results
                  if CLASSES.index(cls) == int(y[seq]))
    summary = {
        "windows_processed": n,
        "mode_final": rt.mode,
        "events": rt.events,
        "latency_ms_mean": float(lat.mean()) if n else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if n else None,
        "rss_mb_final": _rss_mb(),
        "replay_accuracy_on_test_windows": (correct / n) if n else None,
        "telemetry_file": str(args.telemetry),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
