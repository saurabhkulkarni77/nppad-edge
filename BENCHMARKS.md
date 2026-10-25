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

## NASA techniques (2026-10-25; `nppad_edge.nasa_bench`, cache-calibrated)

Measured with `nasa_bench` using the study feature caches: INT8
scales and sensitivity scores calibrated on the TRAIN split (715
windows), evaluated on the TEST split (240 windows). Full mapping to
sources in `NASA_TECHNIQUES.md`.

| Technique | W=60 | W=120 |
|---|---|---|
| TMR head voting — latency mean, float → TMR | 45.9 → 46.4 ms | 44.2 → 43.8 ms (≈ noise) |
| TMR — clean argmax agreement / max Δproba after 1 replica bit flip | 1.000 / **0.0** | 1.000 / 1.5e-08 |
| TMR — extra memory (2 extra head copies) | +4,872,288 B | +4,872,288 B |
| INT8 head — macro-F1 float → int8 (Δ) | 0.9838 → 0.9792 (**−0.0046**) | 0.9876 → 0.9621 (**−0.0255**) |
| INT8 head — prediction agreement with float | 98.8% | 97.1% |
| INT8 head — storage float32 → int8 total | 2,436,144 → 812,128 B (3.0x) | same |
| INT8 head — head time/window (batch-amortised) | 0.23 → 0.99 ms (int8 *slower* in NumPy: no int32 BLAS) | 0.04 → 0.89 ms |
| Golden restore after injected corruption | ~32 ms, predictions bit-identical | ~30 ms, predictions bit-identical |
| Selective protection — golden bytes (top 5% of head entries) | 365,460 B = 15.0% of a full golden head | same |
| Bit-flip campaign — mean macro-F1 drop, 1 flip (top / random / bottom strata) | ~1e-05 / 0.0 / 0.0 | ~2e-05 / 0.0 / 0.0 |
| Bit-flip campaign — mean drop, 32 flips (top / random / bottom) | 0.34 / 0.36 / 0.39 — strata do **not** separate (exponent flips dominate; negative result, reported as measured) | 0.35 / 0.39 / 0.34 — same |

The single-flip result is itself a finding: a logit summed over
50,752 features gives no single weight leverage, so this head is
inherently single-flip tolerant; TMR and golden restore are the
mitigations that measurably matter here, and selective protection is
cheap insurance against accumulated upsets (its scrub demonstrably
repairs injected corruption).
