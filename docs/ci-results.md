# NPPAD-Edge — results measured in GitHub Actions CI

Measured 2026-10-25T21:55:47.590133+00:00 on GitHub Actions ubuntu-latest, 1 core (taskset -c 0).
Run: https://github.com/saurabhkulkarni77/nppad-edge/actions/runs/40237680116

- Test suite: **26 passed**
- Fault-injection demo: **ALL PASS**
- Benchmark: held-out test windows, single core, bundle verified
  (SHA-256) before loading.

| Window | Primary macro-F1 | Fallback macro-F1 | Primary latency mean | Primary latency p95 | Windows |
|---|---|---|---|---|---|
| W = 60 s | 0.9838 | 0.9570 | 29.5 ms | 30.8 ms | 240 |
| W = 120 s | 0.9876 | 0.9528 | 29.3 ms | 29.5 ms | 240 |

Benchmark-process peak RSS (includes the replay harness): W = 60: 74 MB · W = 120: 75 MB (budget 512 MB).

Reference (full-precision study): W = 60 macro-F1 0.9838, W = 120
macro-F1 0.9876. Local-machine figures (contended shared VM) are in
`BENCHMARKS.md`; CI numbers are measured on different hardware and may
differ slightly.
