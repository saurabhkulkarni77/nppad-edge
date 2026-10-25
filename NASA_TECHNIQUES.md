# NASA techniques implemented in NPPAD-Edge

Four techniques from the NASA onboard-computing literature, implemented
as pure-NumPy modules in `nppad_edge/` and measured on the held-out
test windows (240 per window size). Every number below comes from
`python -m nppad_edge.nasa_bench` (artifacts `nasa_bench_w60.json` /
`nasa_bench_w120.json`); deltas are reported as measured, including
the negative ones. Latencies are from the project's contended shared
VM and are upper bounds, as in `BENCHMARKS.md`.

## Sources

1. **NASA Spinoff, "Intelligent Processing at the Edge"** — Ubotica
   Technologies' CogniSAT edge processors, validated with NASA JPL;
   flown on the ISS in 2022 integrated with HPE Spaceborne Computer-2;
   protected by hardware surge protection plus "a memory checker to
   detect data corruption caused by radiation"; JPL models adapted by
   Ubotica to the target processor (Intel Myriad X / Snapdragon 855).
   https://www.techbriefs.com/component/content/article/53164-intelligent-processing-at-the-edge?m=1409
2. **Christopher Wilson, "SpaceCube Overview,"** NASA Goddard Space
   Flight Center, Science Data Processing Branch (Code 587), FISO
   seminar, January 2020. NTRS 20200000731.
   https://ntrs.nasa.gov/api/citations/20200000731/downloads/20200000731.pdf
   — Source of the reliability spectrum used below: box redundancy,
   **Full TMR/NMR**, **Selective mitigation**, DDR ECC, **Configuration
   Scrubbing**, Flash ECC; a small "critical function" manager/watchdog;
   commercial devices quoted at ~148 GOPS/W (int8) vs ~60 GOPS/W
   (float32).
3. **Geist et al., "SpaceCube v3.0 NASA Next-Generation
   High-Performance Processor for Science Applications,"** 33rd Annual
   AIAA/USU Conference on Small Satellites, SSC19-XII-02, 2019.
   https://docslib.org/doc/3655705/spacecube-v3-0-nasa-next-generation-high-performance-processor-for-science-applications
4. **NASA JPL, "How NASA Is Testing AI to Make Earth-Observing
   Satellites Smarter,"** 2025 — Dynamic Targeting's first flight test
   (mid-July 2025) on CogniSAT-6 (Open Cosmos satellite, Ubotica AI
   payload): look ahead ~500 km, decide cloud-vs-clear and repoint in
   under 90 seconds, no human in the loop.
   https://www.jpl.nasa.gov/news/how-nasa-is-testing-ai-to-make-earth-observing-satellites-smarter/
5. **Estlin et al., "AEGIS Automated Targeting for the MER Opportunity
   Rover,"** ISAIRAS 2010 (journal version: ACM Trans. Intell. Syst.
   Technol. 3(3), 2012, DOI 10.1145/2168752.2168764). Onboard science
   targeting on a 20 MHz RAD6000 with a < 4 MB RAM allocation — the
   original "process at capture, decide onboard" system.
   https://ai.jpl.nasa.gov/public/papers/estlin-isairas2010-aegis.pdf
6. **Tran, Chien et al., "The Autonomous Sciencecraft Experiment
   Onboard the EO-1 Spacecraft,"** AAAI 2004 — CASPER planning /
   replanning and robust execution: the mission continues after a
   fault instead of waiting for the ground.
   https://aaai.org/papers/01040-AAAI04-168-the-autonomous-sciencecraft-experiment-onboard-the-eo-1-spacecraft/
7. Supporting (not NASA): **Wang et al., "A Case for
   Application-Aware Space Radiation Tolerance in Orbital Computing"
   (RedNet),** arXiv:2407.11853, 2024 — measured that SEU/MCU
   sensitivity is uneven across a network's parameters, so protecting
   the sensitive fraction suppresses most error at a fraction of the
   redundancy cost.
   https://arxiv.org/abs/2407.11853

## (a) Redundant inference with voting — `harden.py`

**From:** SpaceCube "Full TMR/NMR" (sources 2, 3).
**Implemented:** `VotingClassifier`. Default head-level TMR: features
are computed once; the linear head is applied by three independent
in-memory replicas of the head weights; the output probability is the
per-class median and the replicas' argmax votes are reported
(`EdgeRuntime(tmr_head=True)` adds a `tmr_disagreement` field to
telemetry). A `from_models` full mode votes three complete pipelines
(documented at ~3x feature cost, not the default).
**Measured:** clean-run argmax agreement with the single model
**1.000** at both W; latency W=60 45.9 → 46.4 ms mean (+0.5 ms), W=120
44.2 → 43.8 ms (noise) — head TMR is nearly free because features
dominate the runtime. After a violent bit flip (bit 30) in one
replica's head, the voted probability changed by **0.0** (W=60) /
**1.5e-08** (W=120, one float ulp from renormalisation): the fault is
completely masked while the replica digest check still reports it.
Cost: +4,872,288 B (≈ 4.9 MB) for the two extra head copies.

## (b) INT8-quantized head — `quantize.py`

**From:** CogniSAT / Dynamic Targeting practice of adapting the float
model to an integer VPU (sources 1, 4) and SpaceCube's int8-vs-float
efficiency figures (source 2).
**Implemented:** `QuantizedHead`. Per-feature activation scales
(P99.9 of |f| over the calibration set — the **train** split locally)
are folded into the weights; weights are quantized per output class
to int8; inference quantizes features to int8, accumulates exactly in
int32, and rescales. Feature extraction stays float (documented
scope).
**Measured (train-calibrated, test-evaluated):**

| W | Float macro-F1 | INT8 macro-F1 | Δ | Agreement | Head bytes float32 → int8 total |
|---|---|---|---|---|---|
| 60 | 0.9838 | 0.9792 | **−0.0046** | 98.8% | 2,436,144 → 812,128 (3.0x smaller) |
| 120 | 0.9876 | 0.9621 | **−0.0255** | 97.1% | 2,436,144 → 812,128 (3.0x smaller) |

Honest costs: the W=120 delta (−2.6 points) is real, not noise. And
on this CPU, NumPy's int32 matmul has no BLAS acceleration, so the
int8 head is *slower* than the float head (0.99 vs 0.23 ms/window,
batch-amortised, W=60) — still three orders of magnitude inside the
250 ms budget. The wins here are storage (3.0x) and exact integer
accumulation; on a flight VPU int8 is the accelerated path.

## (c) Watchdog heartbeat + checkpoint/restart + golden restore — `checkpoint.py`

**From:** SpaceCube's small "critical function" manager/watchdog and
configuration scrubbing (source 2), CogniSAT's software memory
checker (source 1), ASE's continue-after-fault operations pattern
(source 6).
**Implemented:** `CheckpointManager` (atomic JSON runtime-state
snapshots — temp file + `os.replace`, so a crash mid-write can never
tear a checkpoint), `HeartbeatMonitor` (staleness detection for the
checker thread — the watchdog is itself watched), and
`GoldenRestorer` (re-verifies the bundle, then copies golden weight
arrays into the live model in place and re-checks the load digest).
`EdgeRuntime(auto_restore=True, checkpoint_path=...)` wires it in: on
a digest mismatch the runtime first scrubs from the golden bundle and
keeps flying the primary; safe mode is now the *last* resort, and the
stream position (seq) resumes from the checkpoint after a restart.
**Measured:** restore completes in **~30–32 ms** (including bundle
re-verification and model rebuild) and post-restore predictions are
bit-identical to pre-corruption. A minimal state checkpoint is
126 B on disk.

## (d) Sensitivity-based selective protection — `sensitivity.py`

**From:** SpaceCube "Selective mitigation" (source 2) and RedNet's
uneven-sensitivity finding (source 7).
**Implemented:** `entry_scores` ranks head weights by E|f_j| x |W_jk|
(clean-leverage proxy); `validate_by_campaign` checks the ranking
with seeded bit-flip campaigns; `SelectiveProtection` keeps a golden
copy of the top 5% of entries (30,451 of 609,024) and scrubs them
back in place on demand. Golden storage **365,460 B = 15.0%** of a
full golden head (2,436,144 B); a scrub repaired an injected
corruption and restored the exact value.
**Measured — including the negative result:** single-bit flips are
nearly harmless *everywhere* in this head (top-stratum mean macro-F1
drop ~1e-05 over 16 trials; random and bottom strata 0.0), because a
logit is a sum over ~50k features and no single weight has leverage.
At 32 simultaneous flips the damage is severe (mean drop ~0.34–0.39)
but the strata do **not** separate — multi-flip damage is dominated by
exponent-bit flips, which can make *any* weight enormous regardless
of its clean magnitude, so the clean-leverage proxy does not predict
it. Conclusion, stated plainly: for this wide linear head, selective
protection is cheap insurance against slow upset accumulation (its
scrub demonstrably repairs), but the mitigations that measurably work
here are (a) TMR voting and (c) golden restore. The campaign code is
kept so the claim stays falsifiable.

## Budgets re-check

All four techniques stay inside the mission budgets: TMR adds
~4.9 MB and ~0.5 ms/window; the int8 head adds ≤ ~1.6 ms/window and
*removes* 1.6 MB of head storage if flown; restore is a one-off
~30 ms on fault; selective-protection golden data is 0.37 MB. The
payload's ~73 MB process and 250 ms/window budget are unaffected in
any material way.
