"""Immutable program files and writable per-user data, for source or release."""

import json
import os
import sys
import tomllib
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
UPDATE_CACHE = (DATA if PACKAGED else RUNTIME) / "update/update-check.json"


def installed_version():
    """The one installed-version authority: release.json, else pyproject.toml."""
    try:
        if PACKAGED:
            value = json.loads(RELEASE.read_text())["version"]
        else:
            value = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
                "version"
            ]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) and value else None


def install_channel(root=ROOT, wheel=WHEEL, packaged=PACKAGED):
    """"homebrew" only for a runtime archive inside a Homebrew Cellar keg."""
    parts = Path(root).parts
    if packaged and not wheel and any(
        a == "Cellar" and b == "mltf" for a, b in zip(parts, parts[1:])
    ):
        return "homebrew"
    return None
