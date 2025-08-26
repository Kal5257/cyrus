import os
import sys
from pathlib import Path
import pytest

# Resolve repo root from this file: <repo>/tests/conftest.py -> parents[1] = <repo>
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path, monkeypatch):
    """Redirect memory persistence to a temp folder for every test."""
    monkeypatch.setenv("CYRUS_DATA_DIR", str(tmp_path))