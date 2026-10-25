# NPPAD-Edge

**A nuclear-plant accident classifier packaged to run under
outer-space-like, limited-compute conditions** — one CPU core, a
≤ 512 MB memory budget, a checksummed model bundle, and a runtime that
watches its own memory for corruption and falls back to a safe-mode
model if anything goes wrong.

[![CI](https://github.com/saurabhkulkarni77/nppad-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/saurabhkulkarni77/nppad-edge/actions/workflows/ci.yml)
· **[Live demo page](https://saurabhkulkarni77.github.io/nppad-edge/)**
· **[CI-measured results](docs/ci-results.md)**

NPPAD-Edge takes the winning model from a Homework 2 SOTA study of the
NPPAD dataset — **MultiRocket + HYDRA** random-kernel features with a
linear head — and rebuilds it as a self-contained edge payload: a
compact, SHA-256-checksummed "uplink bundle" plus a pure-NumPy runtime
that verifies its own integrity, enforces a compute budget, and
downlinks only classification events instead of raw data.

## The concept it mirrors

NASA's Spinoff feature *Intelligent Processing at the Edge* describes
Ubotica's CogniSAT: a small companion processor flown alongside a
satellite's main computer, running ML models adapted to run at the
point of capture in a power- and compute-starved environment. Instead
of downlinking raw imagery, the satellite processes data on board and
sends down only the useful results, and the software carries
protection against the radiation-induced data corruption that is a
fact of life in orbit.

NPPAD-Edge maps that architecture onto a nuclear-plant monitoring
payload:

| CogniSAT / space edge concept        | NPPAD-Edge equivalent                              |
|--------------------------------------|----------------------------------------------------|
| ML model adapted to a tiny processor | MultiRocket+HYDRA features + linear head, exported to a ~MB-scale NumPy bundle (float16 weights) |
| Process at the point of capture      | Streaming classifier on the sensor feed (replayed plant transients) |
| Downlink only useful results         | Telemetry = timestamp, class, confidence, mode. Raw windows never leave the processor |
| Radiation corrupts data/software     | SHA-256-verified uplink bundle + periodic in-memory checker (digest of live weights) |
| Fault protection / safe mode         | Watchdog falls back to a small backup model (LogReg+ on summary features) and flags safe mode in telemetry |
| Strict power/compute envelope        | Single-threaded runtime, ≤512 MB RAM budget, per-window latency budget with breach accounting |

## Measured results

Held-out test windows (240 per window size), edge bundle vs the
full-precision study model it was exported from:

| Model | W = 60 s macro-F1 | W = 120 s macro-F1 |
|---|---|---|
| MultiRocket+HYDRA + linear head (edge bundle) | **0.9838** | **0.9876** |
| Full-precision study reference | 0.9838 | 0.9876 |
| Delta | **0.0000** | **0.0000** |
| Safe-mode fallback (LogReg+) | 0.9570 | 0.9528 |

- **Bundle size: 1.98 MB total** (primary models ≈ 0.94 MB / 1.01 MB
  in float16, fallbacks ≈ 9 KB each, plus manifest + model card).
- **Latency: ≈ 79 ms per window on a single core** (W = 60; mean over
  240 windows, `taskset -c 0` on a contended shared VM — an upper
  bound; see `BENCHMARKS.md`).
- **Payload memory: 72.9 MB peak RSS** against the 512 MB budget.
- **Tests: 12/12 pass** (integrity, accuracy, latency, telemetry).
- **Fault-injection demo: ALL PASS** — a corrupted uplink bundle is
  rejected by SHA-256 verification; an in-flight bit flip in the live
  weights is caught by the memory checker and the payload continues on
  the safe-mode model, flagged in telemetry.

Every push re-runs the full suite, the single-core benchmark and the
fault-injection demo in GitHub Actions; the numbers measured in CI are
committed to [`docs/ci-results.md`](docs/ci-results.md) and may differ
slightly from the local figures above (different hardware).

## The model

- **Task**: classify a window of plant sensor data (88 realistic
  channels — the 8 oracle/leak variables excluded — sampled every
  10 s, onset-aligned at t0 = 100 s) into 12 accident classes.
- **Primary**: MultiRocket + HYDRA features (the study's exact aeon
  implementations, kernel parameters extracted at export; the forward
  transforms re-implemented in NumPy and verified bit-faithful against
  aeon — max abs feature difference ≤ 7.6e-06 over 50,752 features),
  standardised, with a linear head (ridge, α = 0.1, selected on
  validation macro-F1). Windows: W = 60 s (primary) and W = 120 s.
- **Fallback (safe mode)**: LogReg+ — logistic regression on
  per-channel [last, mean, slope] summary features.
- **Protocol**: identical to the study — run-level split
  725 train / 242 val / 244 test, LogHeavy → ChannelScaler → clip ±10
  fitted on train only, scaled windows zero-padded to 12 steps,
  hyperparameters chosen on validation macro-F1 only.

## NASA techniques added (2026-10-25)

Four techniques from the NASA onboard-computing literature, each a
pure-NumPy module with its own tests and measured effect — full
mapping, sources and honest deltas in
[`NASA_TECHNIQUES.md`](NASA_TECHNIQUES.md):

| Technique | Module | NASA source | Measured effect |
|---|---|---|---|
| TMR voting over 3 head replicas | `harden.py` | SpaceCube "Full TMR/NMR" | single-replica bit flip fully masked (Δ proba 0.0); +0.5 ms/window, +4.9 MB |
| INT8-quantized head | `quantize.py` | CogniSAT model adaptation to an integer VPU | macro-F1 −0.0046 (W=60) / −0.0255 (W=120); head storage 3.0x smaller; slower in NumPy (no int32 BLAS) |
| Watchdog heartbeat, checkpoint/restart, golden restore | `checkpoint.py` | SpaceCube watchdog + configuration scrubbing; ASE continue-after-fault | corrupted weights repaired in ~30 ms, predictions bit-identical; safe mode becomes the last resort |
| Sensitivity-based selective protection | `sensitivity.py` | SpaceCube "Selective mitigation"; RedNet (supporting) | golden copy of top-5% entries at 15% of full-head cost; campaign shows this head is single-flip tolerant everywhere and the ranking does not predict multi-flip damage (reported as measured) |

Enable in the runtime: `--tmr-head`, `--auto-restore`,
`--checkpoint PATH`. Measure everything:
`python -m nppad_edge.nasa_bench --bundle bundle --data-root demo_data --W 60`.

## Layout

```
nppad_edge/
  data.py       dataset + study preprocessing (self-contained)
  features.py   MultiRocket / HYDRA forward transforms in pure NumPy
  train.py      ground-side training + bundle export (needs aeon+sklearn)
  bundle.py     bundle format, SHA-256 manifest, verified loading,
                in-memory digest, fault-injection hook
  runtime.py    the edge app: streaming, memory checker, watchdog,
                compute budgets, telemetry downlink
  faulttest.py  scripted radiation-fault demonstration
  bench.py      latency / memory / accuracy benchmark harness
  harden.py     TMR voting over head replicas (NASA techniques)
  quantize.py   INT8-quantized head with calibration + accuracy report
  checkpoint.py watchdog heartbeat, checkpoint/restart, golden restore
  sensitivity.py sensitivity campaign + selective-protection scrubbing
  nasa_bench.py measured evaluation of the four NASA techniques
bundle/         the exported uplink bundle (weights + manifest + card)
demo_data/      the 240 held-out test windows per W, packaged as .npz
                (bit-identical to the full-dataset pipeline output —
                the raw NPPAD CSV tree is far too large for git)
tests/          pytest suite (integrity, accuracy, latency, telemetry,
                + one file per NASA-technique module)
docs/           GitHub Pages demo page + CI-measured results
BENCHMARKS.md   local measured bundle size, RAM, latency, accuracy
NASA_TECHNIQUES.md  technique -> NASA source -> implementation -> effect
```

## Quick start

```bash
pip install -r requirements.txt        # numpy + pandas (+ pytest)

# Tests (run against the packaged test windows in demo_data/)
python -m pytest tests/

# Single-core benchmark, as quoted in BENCHMARKS.md
taskset -c 0 python -m nppad_edge.bench \
  --bundle bundle --data-root demo_data --W 60

# Fault-injection demonstration (corrupted uplink + in-flight
# bit flip -> safe mode)
python -m nppad_edge.faulttest \
  --bundle bundle --data-root demo_data

# Run the payload on the replayed sensor stream
OMP_NUM_THREADS=1 python -m nppad_edge.runtime \
  --bundle bundle --data-root demo_data --W 60 --telemetry telemetry.jsonl
```

To run against the full NPPAD dataset instead of the packaged
windows, point `--data-root` / the `NPPAD_DATA_ROOT` environment
variable at an `Operation_csv_data` directory — the pipeline path is
unchanged. Ground-side training/export (one-time, needs
`requirements-train.txt`):

```bash
python -m nppad_edge.train \
  --data-root /path/to/Operation_csv_data --bundle bundle
```

## Honest limits

- This is a **simulation** of space conditions on ordinary Earth
  hardware: one CPU core, a memory ceiling, and scripted bit flips
  stand in for a radiation environment. No radiation testing, no
  real-time OS, no flight hardware is involved.
- The "sensor stream" is a replay of recorded NPPAD test transients,
  and "downlink" is a local JSONL file (~150 B per event — telemetry
  for 20 windows is 7.1% of the raw window volume).
- Bit flips are injected in the live weight arrays and in bundle
  files; the checker hashes the working weight arrays — it is a
  software watchdog, not ECC memory.
- The fallback trails the study's LogReg+ by ~1–1.5 macro-F1 points —
  it is a deliberately small safe-mode model, and the delta is
  reported as measured in `BENCHMARKS.md`, not rounded away.

## Provenance

Dataset: NPPAD (Qi et al., Tsinghua, 2022; *Scientific Data* 9,
s41597-022-01879-1), PCTRAN-simulated three-loop PWR transients.
Models: MultiRocket (Tan et al., *Data Min. Knowl. Discov.* 2022) and
HYDRA (Dempster, Schmidt & Webb, *Data Min. Knowl. Discov.* 2023),
as implemented in aeon 1.6.0 — the same implementations that produced
the Homework 2 SOTA results this project deploys. Built by Saurabh
Kulkarni as the edge-deployment companion to that study.
