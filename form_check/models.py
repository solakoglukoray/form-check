"""Model file management — bundled .task file path with download fallback."""

from __future__ import annotations

import urllib.request
from pathlib import Path

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_BUNDLED = Path(__file__).parent / "pose_landmarker_lite.task"
_CACHE = Path.home() / ".form_check" / "pose_landmarker_lite.task"


def get_model_path() -> str:
    """Return path to the pose landmarker model, downloading if necessary."""
    if _BUNDLED.exists():
        return str(_BUNDLED)
    if _CACHE.exists():
        return str(_CACHE)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading pose landmarker model (~5 MB)...")
    urllib.request.urlretrieve(_MODEL_URL, _CACHE)
    print(f"Model saved to {_CACHE}")
    return str(_CACHE)
