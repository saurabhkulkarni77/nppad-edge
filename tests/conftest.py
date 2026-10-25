import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "bundle"
# GitHub repo: default to the packaged held-out test windows in
# demo_data/ (bit-identical to the full-dataset pipeline output).
# Set NPPAD_DATA_ROOT to a real Operation_csv_data directory to run
# against the full NPPAD dataset instead.
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "demo_data")


@pytest.fixture(scope="session")
def bundle_dir():
    if not (BUNDLE_DIR / "manifest.json").exists():
        pytest.skip("bundle not built yet (run nppad_edge.train first)")
    return BUNDLE_DIR


@pytest.fixture(scope="session")
def data_root():
    root = os.environ.get("NPPAD_DATA_ROOT", DEFAULT_DATA_ROOT)
    if not Path(root).exists():
        pytest.skip(f"NPPAD data root not available: {root}")
    return root
