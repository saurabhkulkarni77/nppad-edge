"""Fault-injection demonstration for NPPAD-Edge.

Simulates the two radiation-corruption scenarios the design protects
against:

  A. Corrupted uplink: a bit is flipped in a copy of the bundle file
     on disk. The loader's SHA-256 verification must REJECT it — a
     corrupted model is never flown.

  B. In-flight single-event upset: a bit is flipped in the live,
     in-memory head weights after loading. The periodic memory
     checker must catch the digest mismatch, and the watchdog must
     switch the payload to the safe-mode fallback model, flagging
     safe mode in the downlink telemetry.

Run:  python -m nppad_edge.faulttest --bundle bundle --data-root DIR
Exit code 0 = both protections behaved as designed.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .bundle import IntegrityError, verify_bundle
from .runtime import EdgeRuntime, load_test_windows


def scenario_a(bundle_dir) -> bool:
    print("== Scenario A: corrupted uplink bundle ==")
    tmp = Path(tempfile.mkdtemp(prefix="nppad_edge_fault_"))
    bad = tmp / "bundle_bad"
    shutil.copytree(bundle_dir, bad)
    target = bad / "primary_w60.npz"
    blob = bytearray(target.read_bytes())
    blob[len(blob) // 2] ^= 0x01                      # flip one bit
    target.write_bytes(bytes(blob))
    try:
        verify_bundle(bad)
        print("  FAIL: corrupted bundle was ACCEPTED")
        return False
    except IntegrityError as e:
        print(f"  loader rejected the corrupted bundle:\n    {e}")
        print("  PASS: corrupted uplink is never flown")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_b(bundle_dir, data_root) -> bool:
    print("== Scenario B: in-flight bit flip (single-event upset) ==")
    X, y, _ = load_test_windows(data_root, 60)
    windows = [X[i].astype(np.float32) for i in range(6)]
    rt = EdgeRuntime(bundle_dir, W=60, check_interval_s=0.2)
    tel = io.StringIO()
    with rt:
        rt.stream(windows[:3], tel)                   # healthy operation
        print(f"  before fault: mode={rt.mode}, model={rt.model.name}")
        rt.primary.corrupt_for_test("head_coef", 12345)
        print("  injected: flipped bit 7 of one head weight in memory")
        time.sleep(1.0)                               # checker interval elapses
        rt.stream(windows[3:], tel)                   # supervised operation
    records = [json.loads(l) for l in tel.getvalue().splitlines()]
    modes = [(r["seq"], r["model"], r["mode"]) for r in records]
    for m in modes:
        print(f"    telemetry seq {m[0]}: model={m[1]} mode={m[2]}")
    event_types = [e["type"] for e in rt.events]
    print(f"  watchdog events: {event_types}")
    ok = (rt.mode == "safe"
          and "memory_corruption_detected" in event_types
          and "safe_mode_entered" in event_types
          and all(r["mode"] == "safe" for r in records[3:])
          and all(r["model"] == rt.fallback.name for r in records[3:]))
    print("  PASS: checker caught the upset; payload on safe-mode model"
          if ok else "  FAIL: safe mode did not engage as designed")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nppad-edge-faulttest")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    ok_a = scenario_a(args.bundle)
    ok_b = scenario_b(args.bundle, args.data_root)
    print("== fault-injection demo:", "ALL PASS ==" if (ok_a and ok_b)
          else "FAILURES PRESENT ==")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
