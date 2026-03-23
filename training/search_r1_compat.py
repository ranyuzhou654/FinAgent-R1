"""Helpers for importing the vendored Search-R1 veRL codebase."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SEARCH_R1_DIR = ROOT_DIR / "vendor" / "Search-R1"


def ensure_search_r1_on_path() -> Path:
    if not SEARCH_R1_DIR.exists():
        raise FileNotFoundError(f"Search-R1 directory not found: {SEARCH_R1_DIR}")
    search_r1_path = str(SEARCH_R1_DIR)
    if search_r1_path not in sys.path:
        sys.path.insert(0, search_r1_path)
    return SEARCH_R1_DIR
