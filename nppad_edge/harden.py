"""Triple-modular-redundant (TMR) inference for the primary model.

NASA's SpaceCube family flies reconfigurable processors whose
reliability is a *spectrum* chosen per mission: the GSFC "SpaceCube
Overview" (Science Data Processing Branch, Code 587, January 2020,
NTRS 20200000731) lists "Full TMR/NMR" alongside selective mitigation
and box redundancy, and the SpaceCube v3.0 paper (Geist et al., AIAA/USU
Small Satellite Conference 2019) describes the same hybrid approach of
accepting that radiation upsets happen and designing the system so a
single upset cannot silently change the answer.

This module applies that idea in software to the classifier head:

* ``mode="head"`` (default): the (expensive) MultiRocket+HYDRA feature
  extraction runs once; the linear head — a single 50,752 x 12 matmul
  and, per ``sensitivity.py``, the array whose corruption hurts most —
  is applied by three independent in-memory replicas of the head
  weights. The returned probability is the per-class **median** of the
  three replicas, and the per-replica argmax votes are reported. A bit
  flip in any one replica's head (or in the primary model's own live
  head, which voting does not use) cannot change the median: with two
  clean replicas the median *is* the clean probability.
* ``mode="full"``: three complete model replicas each run the full
  pipeline and vote the same way. This also covers faults in the
  feature transforms, at ~3x the feature cost (~240 ms/window locally —
  at the edge of the 250 ms latency budget — and ~3x model memory), so
  it is offered but not the default.

Voting is a complement to, not a replacement for, the bundle SHA-256
verification and the in-memory digest checker: those *detect*
corruption; voting *masks* a single fault so the payload keeps flying
the primary model instead of degrading to safe mode.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    p = np.exp(logits)
    return p / p.sum()


def _head_digest(coef: np.ndarray, intercept: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (coef, intercept):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(np.array(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


class VotingClassifier:
    """Median/vote combiner over three replicas of a primary model.

    Head mode::

        voter = VotingClassifier(primary_model)          # head TMR
        proba = voter.predict_proba(window)              # median proba
        voter.last_vote  # {"votes": [...], "majority": str,
                         #  "disagreement": bool}

    Full mode::

        voter = VotingClassifier.from_models([m1, m2, m3])
    """

    name_suffix = "+tmr"

    def __init__(self, model, replicas: int = 3, mode: str = "head"):
        if mode != "head":
            raise ValueError("use VotingClassifier.from_models for full mode")
        if replicas < 3 or replicas % 2 == 0:
            raise ValueError("voting needs an odd replica count >= 3")
        self._model = model
        self.mode = "head"
        self.classes = list(model.classes)
        self.name = model.name + self.name_suffix
        coef = model._arrays["head_coef"]
        intercept = model._arrays["head_intercept"]
        # independent copies: a fault in one replica's memory cannot
        # alias into another replica (or into the model's live arrays)
        self._heads = [(coef.copy(), intercept.copy())
                       for _ in range(replicas)]
        self._digests = [_head_digest(c, b) for c, b in self._heads]
        self.last_vote = None

    @classmethod
    def from_models(cls, models):
        """Full-pipeline TMR over complete model replicas."""
        if len(models) < 3 or len(models) % 2 == 0:
            raise ValueError("voting needs an odd replica count >= 3")
        obj = cls.__new__(cls)
        obj._model = models[0]
        obj._models = list(models)
        obj.mode = "full"
        obj.classes = list(models[0].classes)
        obj.name = models[0].name + cls.name_suffix
        obj._heads = None
        obj._digests = None
        obj.last_vote = None
        return obj

    # -- inference ----------------------------------------------------
    def _combine(self, probs: np.ndarray) -> np.ndarray:
        """probs: (n_replicas, n_classes) -> median proba + vote record."""
        votes_idx = [int(np.argmax(p)) for p in probs]
        if all(np.array_equal(probs[0], p) for p in probs[1:]):
            med = probs[0]                      # unanimous: bit-exact
        else:
            med = np.median(probs, axis=0)
            med = med / med.sum()               # renormalise (float
            # rounding may move the result by ~1 ulp vs a single model)
        majority_idx = int(np.argmax(med))
        self.last_vote = {
            "votes": [self.classes[i] for i in votes_idx],
            "majority": self.classes[majority_idx],
            "disagreement": len(set(votes_idx)) > 1,
        }
        return med

    def predict_proba(self, window_raw) -> np.ndarray:
        if self.mode == "full":
            probs = np.stack([m.predict_proba(window_raw)
                                for m in self._models])
            return self._combine(probs)
        f = self._model.features(window_raw)
        probs = np.stack([_softmax(f @ coef + intercept)
                          for coef, intercept in self._heads])
        return self._combine(probs)

    # -- integrity ------------------------------------------------------
    def check_replicas(self) -> bool:
        """True iff every replica head still matches its load digest."""
        if self.mode == "full":
            return all(m.check_memory() for m in self._models)
        return all(_head_digest(c, b) == d
                   for (c, b), d in zip(self._heads, self._digests))

    def corrupt_replica_for_test(self, replica: int, flat_index: int,
                                 bit: int = 23):
        """Fault-injection hook (tests only): flip one bit in one
        replica's head weights, simulating a single-event upset that
        struck exactly one of the three redundant copies."""
        coef = self._heads[replica][0]
        u = coef.reshape(-1).view(np.uint32)
        u[flat_index] = u[flat_index] ^ np.uint32(1 << bit)
