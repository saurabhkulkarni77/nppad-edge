"""Single-process benchmark harness for NPPAD-Edge.

Measures, on the replayed held-out test windows: per-window latency
(mean / p95), peak RSS, and macro-F1 for the primary and fallback
models. Run under ``taskset -c 0`` with the thread caps set for the
single-core numbers quoted in BENCHMARKS.md.

    python -m nppad_edge.bench --bundle bundle --data-root DIR --W 60
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from .bundle import load_bundle
from .data import macro_f1
from .runtime import load_test_windows


def bench_W(bundle_dir, data_root, W, limit=None):
    primary, fallback = load_bundle(bundle_dir, W, verify=True)
    X, y, _ = load_test_windows(data_root, W)
    n = len(y) if limit is None else min(limit, len(y))
    out = {}
    for name, model in (("primary", primary), ("fallback", fallback)):
        lat = []
        pred = np.empty(n, dtype=int)
        for i in range(n):
            t0 = time.perf_counter()
            p = model.predict_proba(X[i].astype(np.float32))
            lat.append((time.perf_counter() - t0) * 1e3)
            pred[i] = int(np.argmax(p))
        lat = np.array(lat)
        out[name] = {
            "macro_F1": macro_f1(y[:n], pred),
            "accuracy": float(np.mean(pred == y[:n])),
            "latency_ms_mean": float(lat.mean()),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "latency_ms_max": float(lat.max()),
            "n_windows": n,
        }
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nppad-edge-bench")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--W", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    res = bench_W(args.bundle, args.data_root, args.W, args.limit)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
