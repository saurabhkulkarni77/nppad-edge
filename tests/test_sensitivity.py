"""Selective protection (sensitivity.py): a golden copy of only the
most sensitive head entries, scrubbed back in place — SpaceCube's
"selective mitigation" instead of full redundancy."""
import numpy as np

from nppad_edge.bundle import load_bundle
from nppad_edge.runtime import load_test_windows
from nppad_edge.sensitivity import (SelectiveProtection, entry_scores,
                                    select_top_entries,
                                    validate_by_campaign)

N = 60


def _feats(primary, data_root):
    X, y, _ = load_test_windows(data_root, 60)
    # seeded spread across the whole test set (replay order is grouped
    # by class; a contiguous slice would skew macro-F1)
    idx = np.random.default_rng(0).permutation(len(y))[:N]
    F = np.stack([primary.features(X[i].astype(np.float32)) for i in idx])
    return F, y[idx]


def test_scores_rank_and_select(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    F, _ = _feats(primary, data_root)
    scores = entry_scores(primary, F)
    assert scores.shape == primary._arrays["head_coef"].shape
    idx = select_top_entries(scores, 0.05)
    assert len(idx) == int(round(0.05 * scores.size))
    # the selected entries really are the high-score ones
    assert scores.ravel()[idx].min() >= np.percentile(scores, 94.0)


def test_campaign_runs_and_reports(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    F, y = _feats(primary, data_root)
    out = validate_by_campaign(primary, F, y, trials=3)
    assert out["base_macro_F1"] > 0.7
    for name in ("top", "random", "bottom"):
        s = out["strata"][name]
        assert 0.0 <= s["mean_F1_drop"] <= 1.0
        assert s["max_F1_drop"] >= s["mean_F1_drop"]


def test_scrub_repairs_only_protected(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    F, _ = _feats(primary, data_root)
    prot = SelectiveProtection.from_calibration(primary, F, frac=0.05)
    coef = primary._arrays["head_coef"]
    # protection set is a small fraction of a full golden head
    assert prot.golden_bytes < 0.25 * (coef.nbytes
                                       + primary._arrays["head_intercept"].nbytes)
    assert prot.check() is True

    protected_idx = int(prot.indices[0])
    unprotected_idx = int(np.setdiff1d(np.arange(coef.size),
                                       prot.indices)[0])
    orig_p = coef.ravel()[protected_idx]
    orig_u = coef.ravel()[unprotected_idx]
    u = coef.reshape(-1).view(np.uint32)
    u[protected_idx] = u[protected_idx] ^ np.uint32(1 << 30)
    u[unprotected_idx] = u[unprotected_idx] ^ np.uint32(1 << 30)
    assert prot.check() is False
    repaired = prot.scrub()
    assert repaired >= 1
    assert coef.ravel()[protected_idx] == orig_p      # scrubbed back
    assert coef.ravel()[unprotected_idx] != orig_u    # left as-is by design
    assert prot.check() is True
