"""Uplink bundle: packaging, integrity verification, and the NumPy
inference models that run on the edge.

A bundle is a directory containing:
  primary_w<W>.npz   MultiRocket+HYDRA parameters + linear head
  fallback_w<W>.npz  LogReg+ safe-mode model (summary features)
  manifest.json      SHA-256 of every file + bundle configuration
  model_card.json    accuracy / size / provenance record

Integrity model (mirrors the memory-checker protection described for
spaceborne ML payloads): the loader refuses a bundle whose files do
not match the manifest, and after loading, a digest of the in-memory
weight arrays is recorded so a periodic checker can detect post-load
corruption (e.g. a radiation-induced bit flip) in the live arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import CLASSES, ChannelScaler, LogHeavy, apply_preprocessing, summary_features
from .features import HydraFeatures, MultiRocketFeatures, SparseScaler, pad_windows

MANIFEST_NAME = "manifest.json"


class IntegrityError(RuntimeError):
    """Raised when a bundle fails SHA-256 verification."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(bundle_dir, config: dict) -> dict:
    bundle_dir = Path(bundle_dir)
    files = {}
    for p in sorted(bundle_dir.iterdir()):
        if p.is_file() and p.name != MANIFEST_NAME:
            files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"format": "nppad-edge-bundle/1", "config": config, "files": files}
    (bundle_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(bundle_dir) -> dict:
    p = Path(bundle_dir) / MANIFEST_NAME
    if not p.exists():
        raise IntegrityError(f"manifest missing in bundle {bundle_dir}")
    return json.loads(p.read_text())


def verify_bundle(bundle_dir) -> dict:
    """Verify every file in the bundle against the manifest.

    Returns the manifest on success; raises IntegrityError otherwise.
    Any file present but not listed, listed but missing, or with a
    mismatched hash is a hard failure — a corrupted uplink is never
    flown."""
    bundle_dir = Path(bundle_dir)
    manifest = read_manifest(bundle_dir)
    listed = manifest.get("files", {})
    for name, rec in listed.items():
        p = bundle_dir / name
        if not p.exists():
            raise IntegrityError(f"bundle file missing: {name}")
        actual = sha256_file(p)
        if actual != rec["sha256"]:
            raise IntegrityError(
                f"SHA-256 mismatch for {name}: manifest {rec['sha256'][:16]}… "
                f"actual {actual[:16]}… — bundle rejected")
    extra = {p.name for p in bundle_dir.iterdir()
             if p.is_file() and p.name != MANIFEST_NAME} - set(listed)
    if extra:
        raise IntegrityError(f"unlisted files in bundle: {sorted(extra)}")
    return manifest


def _arrays_digest(arrays: dict) -> str:
    """Digest of the live in-memory weight arrays (fixed order)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(np.array(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


class _BaseModel:
    """Shared preprocessing + in-memory integrity digest."""

    def _init_pre(self, z):
        self._lg = LogHeavy.from_params(z["pre_log_idx"], z["pre_log_s"])
        self._sc = ChannelScaler.from_params(z["pre_mu"], z["pre_sd"])

    def _preprocess(self, window_raw):
        """window_raw: (T, C) raw plant values -> scaled (T, C)."""
        X = np.asarray(window_raw, dtype=np.float32)[None, :, :]
        return apply_preprocessing(X, self._lg, self._sc)[0]

    def memory_digest(self) -> str:
        return _arrays_digest(self._arrays)

    def check_memory(self) -> bool:
        return self.memory_digest() == self._digest0

    def corrupt_for_test(self, array_name: str, flat_index: int):
        """Fault-injection hook (used only by the fault test): flip one
        bit in a live float32 weight array, simulating a radiation
        strike (single-event upset)."""
        a = self._arrays[array_name]
        assert a.dtype == np.float32, "fault hook expects float32 arrays"
        u = a.reshape(-1).view(np.uint32)
        u[flat_index] = u[flat_index] ^ np.uint32(1 << 7)


class PrimaryModel(_BaseModel):
    """MultiRocket+HYDRA features + standardised linear head."""

    name = "multirocket-hydra"

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self._init_pre(z)
        mr_params = []
        for half in ("mr0", "mr1"):
            mr_params.append(tuple(z[f"{half}_{k}"] for k in
                                   ("nchan", "chan", "dil", "nfpd", "bias")))
        self._mr = MultiRocketFeatures(z["mr_indices"], mr_params[0], mr_params[1])
        self._hy = HydraFeatures(
            z["hy_W"].astype(np.float64), z["hy_idx"], z["hy_dil"], z["hy_pad"],
            int(z["hy_g"]), int(z["hy_k"]))
        self._sparse = SparseScaler.from_params(z["hy_mu"], z["hy_sigma"])
        # live (working) copies — these are what the checker watches
        self._arrays = {
            "head_coef": z["head_coef"].astype(np.float32),
            "head_intercept": z["head_intercept"].astype(np.float32),
            "feat_mu": z["feat_mu"].astype(np.float32),
            "feat_sd": z["feat_sd"].astype(np.float32),
            "mr0_bias": mr_params[0][4].astype(np.float32),
            "mr1_bias": mr_params[1][4].astype(np.float32),
            "hy_W": z["hy_W"].astype(np.float32),
        }
        self._intercept = self._arrays["head_intercept"]
        self.classes = [str(c) for c in z["classes"]]
        self._digest0 = self.memory_digest()
        # rebind transforms to the live copies so corruption is visible
        # (diff half keeps the raw half's channel subsets, per aeon)
        self._mr.parameter = (mr_params[0][0], mr_params[0][1], mr_params[0][2],
                              mr_params[0][3], self._arrays["mr0_bias"])
        self._mr.parameter1 = (mr_params[0][0], mr_params[0][1],
                               mr_params[1][2], mr_params[1][3],
                               self._arrays["mr1_bias"])
        self._hy.W = self._arrays["hy_W"].astype(np.float64)

    def features(self, window_raw):
        Xs = self._preprocess(window_raw)                # (T, C) scaled
        x = pad_windows(Xs.T[None, :, :])[0]             # (C, Tpad)
        f_mr = self._mr.transform_one(x)
        f_hy = self._hy.transform_one(x)
        mr_s = (f_mr - self._arrays["feat_mu"]) / self._arrays["feat_sd"]
        hy_s = self._sparse.transform(f_hy[None, :])[0]
        return np.concatenate([mr_s, hy_s]).astype(np.float32)

    def predict_proba(self, window_raw):
        f = self.features(window_raw)
        logits = f @ self._arrays["head_coef"] + self._intercept
        logits = logits - logits.max()
        p = np.exp(logits)
        return p / p.sum()


class FallbackModel(_BaseModel):
    """Safe-mode model: LogReg+ on per-channel summary features."""

    name = "logreg-plus-fallback"

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self._init_pre(z)
        self._arrays = {
            "head_coef": z["head_coef"].astype(np.float32),
            "head_intercept": z["head_intercept"].astype(np.float32),
            "sum_mu": z["sum_mu"].astype(np.float32),
            "sum_sd": z["sum_sd"].astype(np.float32),
        }
        self._intercept = self._arrays["head_intercept"]
        self.classes = [str(c) for c in z["classes"]]
        self._digest0 = self.memory_digest()

    def predict_proba(self, window_raw):
        Xs = self._preprocess(window_raw)                # (T, C)
        f = summary_features(Xs[None, :, :])[0]          # (3C,)
        fs = (f - self._arrays["sum_mu"]) / self._arrays["sum_sd"]
        logits = fs @ self._arrays["head_coef"] + self._intercept
        logits = logits - logits.max()
        p = np.exp(logits)
        return p / p.sum()


def load_bundle(bundle_dir, W: int, verify: bool = True):
    """Verify (by default) and load (primary, fallback) for window W."""
    bundle_dir = Path(bundle_dir)
    if verify:
        verify_bundle(bundle_dir)
    primary = PrimaryModel(bundle_dir / f"primary_w{int(W)}.npz")
    fallback = FallbackModel(bundle_dir / f"fallback_w{int(W)}.npz")
    return primary, fallback
