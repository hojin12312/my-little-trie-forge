"""Immutable program files and writable per-user data, for source or release."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL = (ROOT / "engine/release.json").is_file()
PACKAGED = WHEEL or (ROOT / "release.json").is_file()
RELEASE = ROOT / ("engine/release.json" if WHEEL else "release.json")
DATA = Path(os.environ.get("MLTF_DATA_ROOT", Path.home() / "Library/Application Support/MLTF")) if PACKAGED else ROOT
MODELS = DATA / "models" if PACKAGED else ROOT / "install/models"
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = (
    Path(sys.executable)
    if WHEEL
    else ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
)
BINARY = ROOT / ("engine/splash" if PACKAGED else "build/splash")
