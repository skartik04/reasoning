from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("REASONING_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DATA_DIR = Path(os.getenv("REASONING_DATA_DIR", ARTIFACTS_DIR / "data"))
RESPONSES_DIR = Path(os.getenv("REASONING_RESPONSES_DIR", ARTIFACTS_DIR / "responses"))
RESULTS_DIR = Path(os.getenv("REASONING_RESULTS_DIR", ARTIFACTS_DIR / "results"))
RESIDUALS_DIR = Path(os.getenv("REASONING_RESIDUALS_DIR", ARTIFACTS_DIR / "residuals"))
HF_CACHE_DIR = Path(os.getenv("REASONING_HF_CACHE", ARTIFACTS_DIR / "hf_cache"))


def ensure_artifact_dirs() -> None:
    """Create the standard local artifact folders used by scripts."""
    for path in (ARTIFACTS_DIR, DATA_DIR, RESPONSES_DIR, RESULTS_DIR, RESIDUALS_DIR, HF_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
