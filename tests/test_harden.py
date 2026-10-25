"""TMR head voting (harden.py): three independent replicas of the
linear head, median probability + majority vote — SpaceCube's
"Full TMR/NMR" applied in software to the classifier head."""
import numpy as np

from nppad_edge.bundle import load_bundle
from nppad_edge.harden import VotingClassifier
from nppad_edge.runtime import load_test_windows


def _windows(data_root, n=4):
    X, y, _ = load_test_windows(data_root, 60)
    return [X[i].astype(np.float32) for i in range(n)]


def test_voting_matches_clean_model(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    voter = VotingClassifier(primary)
    for w in _windows(data_root):
        pv = voter.predict_proba(w)
        pf = primary.predict_proba(w)
        # median of three identical replicas IS the clean probability
        assert np.array_equal(pv, pf)
        assert voter.last_vote["disagreement"] is False
        assert voter.last_vote["majority"] == primary.classes[int(np.argmax(pf))]


def test_single_replica_bitflip_is_masked(bundle_dir, data_root):
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    voter = VotingClassifier(primary)
    w = _windows(data_root, 1)[0]
    clean = voter.predict_proba(w)
    voter.corrupt_replica_for_test(1, int(np.argmax(
        np.abs(voter._heads[1][0]))), bit=30)          # violent flip
    assert voter.check_replicas() is False             # fault is visible
    masked = voter.predict_proba(w)
    # two clean replicas outvote the corrupted one: output unchanged
    # (to float rounding — the median is renormalised)
    assert np.allclose(masked, clean, atol=1e-6)
    # ... while the corrupted replica alone would have answered with
    # a visibly different probability vector
    coef, intercept = voter._heads[1]
    f = primary.features(w)
    logits = f @ coef + intercept
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert not np.allclose(p, clean)


def test_full_mode_voting(bundle_dir, data_root):
    models = [load_bundle(bundle_dir, 60, verify=True)[0] for _ in range(3)]
    voter = VotingClassifier.from_models(models)
    w = _windows(data_root, 1)[0]
    pv = voter.predict_proba(w)
    pf = models[0].predict_proba(w)
    assert np.array_equal(pv, pf)
    assert voter.check_replicas() is True
