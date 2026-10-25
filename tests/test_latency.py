"""Single-core latency: the runtime must classify a window well
inside the mission latency budget when pinned to one CPU core
(``taskset -c 0``) with BLAS thread caps at 1."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_BUDGET_MS = 250.0     # mission budget used by the runtime watchdog


def test_single_core_latency(bundle_dir, data_root):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = "1"
    cmd = ["taskset", "-c", "0", sys.executable, "-m", "nppad_edge.bench",
           "--bundle", str(bundle_dir), "--data-root", str(data_root),
           "--W", "60", "--limit", "40"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                          capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(proc.stdout)
    lat = res["primary"]
    print(f"\nsingle-core latency: mean {lat['latency_ms_mean']:.1f} ms, "
          f"p95 {lat['latency_ms_p95']:.1f} ms over {lat['n_windows']} windows")
    assert lat["latency_ms_mean"] < MEAN_BUDGET_MS
