"""Bundle integrity: the loader must accept a clean uplink bundle and
refuse any corrupted one (SHA-256 manifest verification)."""
import shutil
from pathlib import Path

import pytest

from nppad_edge.bundle import IntegrityError, load_bundle, verify_bundle


def test_good_bundle_verifies(bundle_dir):
    manifest = verify_bundle(bundle_dir)
    names = set(manifest["files"])
    assert "primary_w60.npz" in names
    assert "fallback_w60.npz" in names
    assert "model_card.json" in names


def test_good_bundle_loads(bundle_dir):
    primary, fallback = load_bundle(bundle_dir, 60, verify=True)
    assert primary.check_memory()
    assert fallback.check_memory()


def _corrupted_copy(bundle_dir, tmp_path, target_name):
    bad = tmp_path / "bundle_bad"
    shutil.copytree(bundle_dir, bad)
    target = bad / target_name
    blob = bytearray(target.read_bytes())
    blob[len(blob) // 3] ^= 0x01
    target.write_bytes(bytes(blob))
    return bad


def test_bitflip_in_primary_rejected(bundle_dir, tmp_path):
    bad = _corrupted_copy(bundle_dir, tmp_path, "primary_w60.npz")
    with pytest.raises(IntegrityError):
        verify_bundle(bad)
    with pytest.raises(IntegrityError):
        load_bundle(bad, 60, verify=True)


def test_bitflip_in_model_card_rejected(bundle_dir, tmp_path):
    bad = _corrupted_copy(bundle_dir, tmp_path, "model_card.json")
    with pytest.raises(IntegrityError):
        verify_bundle(bad)


def test_missing_file_rejected(bundle_dir, tmp_path):
    bad = tmp_path / "bundle_missing"
    shutil.copytree(bundle_dir, bad)
    (bad / "fallback_w60.npz").unlink()
    with pytest.raises(IntegrityError):
        verify_bundle(bad)


def test_in_memory_corruption_detected(bundle_dir):
    """The memory checker's digest must notice a single flipped bit
    in the live weights (radiation upset simulation)."""
    primary, _ = load_bundle(bundle_dir, 60, verify=True)
    assert primary.check_memory()
    primary.corrupt_for_test("head_coef", 999)
    assert not primary.check_memory()
