"""Sensitivity-based selective protection of the classifier head.

Full triple-modular redundancy protects everything at 3x cost; flight
designers rarely pay that everywhere. SpaceCube's reliability spectrum
(GSFC "SpaceCube Overview", Code 587, January 2020) names the middle
option explicitly: **selective mitigation** — protect what matters,
accept the risk elsewhere. Wang et al., "A Case for Application-Aware
Space Radiation Tolerance in Orbital Computing" (arXiv:2407.11853,
2024) measured why this works for neural inference: SEU/MCU
sensitivity is *uneven* across a network's parameters, so protecting
the sensitive fraction suppresses almost all of the error at a
fraction of the redundancy cost.

This module measures the unevenness for NPPAD-Edge's linear head and
builds the protection set:

* ``entry_scores`` — a cheap analytic sensitivity proxy per head
  weight: E|f_j| (how large the feature driving weight (j, k) runs)
  times |W_jk| (how large the weight is). A bit flip scales a weight
  multiplicatively, so large-weight x large-activation entries have
  the most leverage over the logits.
* ``validate_by_campaign`` — the empirical check: seeded single-bit
  flips sampled from the top-scored entries vs from all entries vs
  from the bottom half, scoring the macro-F1 drop of each group on
  held-out features. The ranking is only trusted because this
  campaign measures it.
* ``SelectiveProtection`` — keeps a golden copy of just the top
  ``frac`` entries of ``head_coef`` (plus the intercept) and *scrubs*
  them: compare live vs golden, repair mismatches in place, report
  how many were repaired. Storage is ~``frac`` of the head plus an
  int32 index list, versus a full golden copy of every array.
"""
from __future__ import annotations

import numpy as np

from .data import macro_f1


def head_predict(coef, intercept, feats):
    return np.argmax(feats @ coef + intercept, axis=1)


def entry_scores(model, feats_calib) -> np.ndarray:
    """(D, K) sensitivity proxy: mean|f_j| over calibration features,
    times |W_jk|. Higher = a flip here moves the logits more."""
    mean_abs = np.mean(np.abs(np.asarray(feats_calib, dtype=np.float32)),
                       axis=0)                                   # (D,)
    return mean_abs[:, None] * np.abs(model._arrays["head_coef"])


def select_top_entries(scores: np.ndarray, frac: float):
    """Flat indices of the top ``frac`` entries by score."""
    n = scores.size
    k = max(1, int(round(frac * n)))
    flat = np.argpartition(scores.ravel(), n - k)[n - k:]
    return np.sort(flat).astype(np.int64)


def _flip_one(arr: np.ndarray, flat_index: int, bit: int, rng) -> None:
    u = arr.reshape(-1).view(np.uint32)
    u[flat_index] = u[flat_index] ^ np.uint32(1 << bit)


def validate_by_campaign(model, feats_eval, y_eval, seed: int = 0,
                         trials: int = 16, flips_per_trial: int = 1,
                         top_frac: float = 0.01):
    """Seeded bit-flip campaign over three strata of head entries.

    For each stratum, run ``trials`` trials; each trial flips
    ``flips_per_trial`` entries drawn from that stratum (one random
    bit each, uniform over the 32 bits, as radiation strikes would)
    in a scratch copy of the head and measures the macro-F1 drop.
    Returns per-stratum mean drops."""
    coef0 = model._arrays["head_coef"]
    intercept = model._arrays["head_intercept"]
    base = macro_f1(y_eval, head_predict(coef0, intercept, feats_eval))
    scores = entry_scores(model, feats_eval)
    flat_scores = scores.ravel()
    order = np.argsort(flat_scores)
    n = flat_scores.size
    k_top = max(1, int(round(top_frac * n)))
    strata = {
        "top": order[n - k_top:],
        "random": np.arange(n),
        "bottom": order[: n // 2],
    }
    rng = np.random.default_rng(seed)
    out = {"base_macro_F1": base, "trials": trials,
           "flips_per_trial": flips_per_trial,
           "top_frac": top_frac, "strata": {}}
    for name, pool in strata.items():
        drops = []
        for _ in range(trials):
            coef = coef0.copy()
            for idx in pool[rng.integers(len(pool),
                                         size=flips_per_trial)]:
                _flip_one(coef, int(idx), int(rng.integers(0, 32)), rng)
            f1 = macro_f1(y_eval, head_predict(coef, intercept, feats_eval))
            drops.append(base - f1)
        out["strata"][name] = {
            "mean_F1_drop": float(np.mean(drops)),
            "max_F1_drop": float(np.max(drops)),
            "frac_trials_harmful": float(np.mean(np.array(drops) > 0)),
        }
    return out


class SelectiveProtection:
    """Golden-copy scrubbing of the most sensitive head entries.

    Keeps golden values for the flat ``indices`` of ``head_coef``
    (chosen by ``entry_scores``) and always for the intercept.
    ``scrub()`` repairs any drifted protected entries in place and
    returns how many values were repaired."""

    def __init__(self, model, indices):
        self._model = model
        self.indices = np.asarray(indices, dtype=np.int64)
        coef = model._arrays["head_coef"]
        self._golden = coef.ravel()[self.indices].copy()
        self._golden_intercept = model._arrays["head_intercept"].copy()
        self.protected_entries = int(len(self.indices))
        self.total_entries = int(coef.size)

    @classmethod
    def from_calibration(cls, model, feats_calib, frac: float = 0.05):
        scores = entry_scores(model, feats_calib)
        return cls(model, select_top_entries(scores, frac))

    @property
    def golden_bytes(self) -> int:
        """Storage of the protection set (golden values + indices +
        intercept), vs ``total_entries * 4`` for a full golden head."""
        return int(self._golden.nbytes + self.indices.nbytes
                   + self._golden_intercept.nbytes)

    def scrub(self) -> int:
        """Repair protected entries that differ from golden.
        Returns the number of values repaired."""
        coef = self._model._arrays["head_coef"]
        live = coef.ravel()
        bad = live[self.indices] != self._golden
        n = int(bad.sum())
        if n:
            live[self.indices[bad]] = self._golden[bad]
        icpt = self._model._arrays["head_intercept"]
        if not np.array_equal(icpt, self._golden_intercept):
            icpt[...] = self._golden_intercept
            n += 1
        return n

    def check(self) -> bool:
        """True iff all protected entries currently match golden."""
        live = self._model._arrays["head_coef"].ravel()
        return bool(np.array_equal(live[self.indices], self._golden)
                    and np.array_equal(
                        self._model._arrays["head_intercept"],
                        self._golden_intercept))
