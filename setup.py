"""Build a platform wheel containing the native MLTF runtime."""

import hashlib
import json
import os
import platform
import subprocess
import tomllib
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeBuild(build_py):
    def run(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("MLTF native wheel requires an Apple Silicon Mac")
        binary, metallib = (
            ROOT / "build" / name for name in ("splash", "splash.metallib")
        )
        clean_env = {key: value for key, value in os.environ.items()
                     if not key.startswith("SPLASH_") and key not in
                     {"MAKEFLAGS", "MAX_BATCH_WIDTH", "Q8_PREFILL_ACTIVATION", "Q8_PREFILL_STAGEPAGE"}}
        subprocess.run(["make", "-j4", "all", "MAX_BATCH_WIDTH=4",
                        "Q8_PREFILL_ACTIVATION=a8", "Q8_PREFILL_STAGEPAGE=0"],
                       cwd=ROOT, check=True, env=clean_env)
        super().run()
        target = Path(self.build_lib) / "engine"
        target.mkdir(parents=True, exist_ok=True)
        for source in (binary, metallib):
            destination = target / source.name
            destination.write_bytes(source.read_bytes())
            destination.chmod(source.stat().st_mode & 0o777)
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ]
        (target / "release.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "binary_sha256": sha(binary),
                    "metallib_sha256": sha(metallib),
                },
                indent=2,
            )
            + "\n"
        )


class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.plat_name = "macosx_26_0_arm64"

    def get_tag(self):
        return "py3", "none", "macosx_26_0_arm64"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # The executable and metallib are platform binaries even though no
        # Python extension module is linked.
        return True


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": NativeBuild, "bdist_wheel": NativeWheel},
)
