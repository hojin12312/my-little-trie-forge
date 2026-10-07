import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_package_tool():
    spec = importlib.util.spec_from_file_location(
        "mltf_package_tool", ROOT / "dev/tools/package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StagedRuntimeFiles(unittest.TestCase):
    def test_every_install_module_is_staged(self):
        # launcher.py imports its sibling modules at the top level, so a module
        # missing from the archive breaks `mltf --help`, not only its own command.
        package = load_package_tool()
        modules = {path.name for path in (ROOT / "install").glob("*.py")}
        self.assertEqual(modules, {n for n in package.INSTALL_FILES if n.endswith(".py")})

    def test_staged_files_exist(self):
        package = load_package_tool()
        for folder, names in (
            ("install", package.INSTALL_FILES),
            ("install/completions", package.COMPLETION_FILES),
            ("server", package.SERVER_FILES),
        ):
            for name in names:
                self.assertTrue((ROOT / folder / name).is_file(), f"{folder}/{name}")


if __name__ == "__main__":
    unittest.main()
