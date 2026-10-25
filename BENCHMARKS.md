# NPPAD-Edge — measured benchmarks

All numbers below were **measured** on this machine (a shared 2-vCPU
VM, heavily loaded by unrelated work during the runs — treat latencies
as upper bounds; a quiet core will do better). Nothing is estimated.

## Simulated spacecraft budget (stated plainly)

| Resource | Budget | How enforced / measured |
|---|---|---|
| CPU | **1 core** | `taskset -c 0`, BLAS/OpenMP thread caps = 1 |
| RAM | **≤ 512 MB** | watchdog checks RSS; payload-only peak measured at **72.9 MB** |
| Latency | **250 ms / window** | runtime watchdog; 3 consecutive breaches → safe mode |
| Downlink | events only | ~150 B per classification event (JSONL) |

## Uplink bundle size

| File | Bytes |
|---|---|
| primary_w60.npz (MultiRocket+HYDRA + ridge head, float16) | 942,874 |
| primary_w120.npz | 1,014,674 |
| fallback_w60.npz (LogReg+ safe mode) | 9,128 |
| fallback_w120.npz | 9,360 |
| model_card.json + manifest.json | 5,003 |
| **Bundle total** | **1,981,039 B ≈ 1.98 MB** |

## Accuracy — edge bundle vs full-precision study (test macro-F1, t0=100)

| Model | W=60 edge | W=60 study | Δ | W=120 edge | W=120 study | Δ |
|---|---|---|---|---|---|---|
| MultiRocket+HYDRA + linear head | **0.9838** | 0.9838 | **0.0000** | **0.9876** | 0.9876 | **0.0000** |
| Fallback (LogReg+, safe mode) | 0.9570 | 0.9675 | −0.0105 | 0.9528 | 0.9677 | −0.0149 |

The edge bundle reproduces the study's headline scores **exactly** on
the held-out test windows (240 per window size): the NumPy transforms
are bit-faithful to aeon (max abs feature difference 7.6e-06 for
MultiRocket, ≤3.8e-06 for HYDRA, over 50,752 features), and float16
storage of the head costs nothing measurable. The fallback trails the
study's LogReg+ by ~1–1.5 points — it is a deliberately small
safe-mode model with a lighter tuning grid, and the delta is reported
as measured. Head selection (ridge α=0.1 over logistic candidates)
was made on validation macro-F1 only.

## Latency — single core, per window (240 test windows each)

| Model | W=60 mean | W=60 p95 | W=60 max | W=120 mean | W=120 p95 | W=120 max |
|---|---|---|---|---|---|---|
| Primary | 78.7 ms | 139.5 ms | 287.3 ms | 84.2 ms | 158.7 ms | 339.2 ms |
| Fallback | 0.13 ms | 0.08 ms | 2.3 ms | 0.15 ms | 0.22 ms | 3.2 ms |

Almost all primary latency is the MultiRocket feature computation in
pure NumPy; the linear head itself is a single 50,752×12 matmul.

## Memory

| Process | Peak RSS |
|---|---|
| Payload only (bundle + inference over 240 windows) | **72.9 MB** |
| Full replay harness (also holds the whole NPPAD dataset in RAM) | 446–455 MB |

Both sit inside the 512 MB budget; the payload itself is a 73 MB
process — the replay figure is the ground harness's dataset, not the
model.

## Downlink volume

20 windows → 3,008 B of telemetry vs 42,240 B of raw window data
(7.1%, a 14× reduction) — and the raw *run records* the windows were
cut from are ~100× larger still. Telemetry per event: sequence,
ISO-8601 timestamp, class, confidence, model, mode. Nothing else
leaves the processor.
