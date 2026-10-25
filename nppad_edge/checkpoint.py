"""Watchdog heartbeat, runtime checkpoint/restart, and golden restore.

SpaceCube pairs its high-performance commercial processors with a
small "critical function" manager/watchdog (GSFC "SpaceCube Overview",
Code 587, January 2020, NTRS 20200000731): the unreliable-but-fast
compute is supervised by something simple and trustworthy, and the
configuration is periodically scrubbed back to a known-good state.
Ubotica's CogniSAT flew the same pattern in software — the NASA
Spinoff account notes the payload was protected by "a memory checker
to detect data corruption caused by radiation". JPL's Autonomous
Sciencecraft Experiment (Tran et al., AAAI 2004) added the operations
side: robust execution and replanning so the mission continues after a
fault instead of waiting for the ground.

NPPAD-Edge already *detects* corruption (SHA-256 bundle verification +
the in-memory digest checker) and *degrades* (safe-mode fallback).
This module adds the missing middle: **repair and resume**.

* ``CheckpointManager`` — atomic JSON snapshots of runtime state
  (mode, stream position, latency accounting, event log). Written via
  a temp file + ``os.replace`` so a crash mid-write can never leave a
  torn checkpoint: the reader sees the old state or the new one.
* ``HeartbeatMonitor`` — tracks the last-seen beat of each supervised
  component (e.g. the memory-checker thread) and reports sources that
  have gone stale, i.e. a watchdog that is itself watched.
* ``GoldenRestorer`` — the scrub: on a digest mismatch, reload the
  golden weight arrays from the (re-verified) uplink bundle and copy
  them into the live model *in place*, so the transforms that were
  bound to the live arrays at load time see the repaired values. If
  the restored model re-verifies against its load digest, the payload
  stays on the primary model; only if restore fails does the watchdog
  fall back to safe mode.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .bundle import (FallbackModel, IntegrityError, PrimaryModel,
                     verify_bundle)

CHECKPOINT_FORMAT = "nppad-edge-checkpoint/1"


class CheckpointError(RuntimeError):
    pass


class CheckpointManager:
    """Atomic JSON checkpoints of runtime state."""

    def __init__(self, path):
        self.path = Path(path)

    def save(self, state: dict) -> dict:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)          # atomic on POSIX
        return payload

    def load(self):
        """Return the saved state dict, or None if no checkpoint
        exists. Raises CheckpointError on a malformed file."""
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise CheckpointError(f"unreadable checkpoint: {e}") from e
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError(
                f"checkpoint format mismatch: {payload.get('format')!r}")
        return payload["state"]


class HeartbeatMonitor:
    """Staleness detection for supervised components.

    ``beat(source)`` is called by each watched component;
    ``stale_sources()`` lists those whose last beat is older than the
    configured threshold — the watchdog's own health check."""

    def __init__(self, stale_after_s: float):
        self.stale_after_s = float(stale_after_s)
        self._beats: dict[str, float] = {}

    def beat(self, source: str, when: float | None = None):
        self._beats[source] = time.monotonic() if when is None else when

    def last_beat(self, source: str):
        return self._beats.get(source)

    def stale_sources(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        return sorted(s for s, t in self._beats.items()
                      if now - t > self.stale_after_s)


class GoldenRestorer:
    """Repair live model weights from the verified uplink bundle.

    Construction re-verifies the bundle (SHA-256 manifest); each
    ``restore()`` re-verifies again before trusting the golden copy —
    a corrupted golden source must never be scrubbed *in*."""

    def __init__(self, bundle_dir, W: int):
        self.bundle_dir = Path(bundle_dir)
        self.W = int(W)
        verify_bundle(self.bundle_dir)

    def restore(self, model) -> bool:
        """Copy golden arrays into ``model`` in place. Returns True
        iff the model afterwards matches its load-time digest."""
        if isinstance(model, PrimaryModel):
            npz = self.bundle_dir / f"primary_w{self.W}.npz"
            fresh = PrimaryModel(npz)
        elif isinstance(model, FallbackModel):
            npz = self.bundle_dir / f"fallback_w{self.W}.npz"
            fresh = FallbackModel(npz)
        else:
            raise TypeError(f"cannot restore {type(model).__name__}")
        try:
            verify_bundle(self.bundle_dir)
        except IntegrityError:
            return False
        for name, arr in fresh._arrays.items():
            model._arrays[name][...] = arr      # in place: live
            # references held by the transforms stay valid
        if isinstance(model, PrimaryModel):
            # HYDRA weights were bound as a float64 copy at load time
            model._hy.W = model._arrays["hy_W"].astype("float64")
        return model.check_memory()
