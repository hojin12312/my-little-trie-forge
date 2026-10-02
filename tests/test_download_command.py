import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from install import launcher, models


class DownloadContract(unittest.TestCase):
    def test_requires_explicit_model_license_consent(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["download", "--model", "owner/prepared"])

    def test_unsafe_repo_rejected_before_download(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(
                ["download", "--model", "owner/../private", "--accept-model-licenses"]
            )

    def test_download_is_not_serve_or_source_build(self):
        args = launcher.parse_args(
            ["download", "--model", "owner/prepared", "--accept-model-licenses"]
        )
        with (
            patch.object(models, "prepare") as prepare,
            patch.object(models, "verify_installed") as verify,
            patch.object(
                launcher,
                "_ensure_installed",
                side_effect=AssertionError("unexpected build/server"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(launcher.download_model(args), 0)
        prepare.assert_called_once_with(args)
        verify.assert_called_once_with(
            args.models.resolve(), model_id=args.model, full=True
        )

    def test_low_space_refuses_preserving_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "user-data"
            sentinel.write_bytes(b"keep")
            manifest = {
                "artifacts": [{"path": "target", "size": 100, "sha256": "0" * 64}]
            }
            with patch.object(
                models.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=2 * 1024**3 + 99),
            ):
                with self.assertRaises(models.ModelError):
                    models.download_space_preflight(root, manifest)
            self.assertEqual(sentinel.read_bytes(), b"keep")
            self.assertFalse((root / "target").exists())

    def test_corrupt_same_size_blob_is_counted_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target").write_bytes(b"bad")
            manifest = {
                "artifacts": [
                    {
                        "path": "target",
                        "size": 3,
                        "sha256": hashlib.sha256(b"good").hexdigest(),
                    }
                ]
            }
            with patch.object(
                models.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=2 * 1024**3 + 3),
            ):
                self.assertEqual(
                    models.download_space_preflight(root, manifest)["missing_bytes"], 3
                )
            self.assertEqual((root / "target").read_bytes(), b"bad")

    def test_non_symlink_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "user-model"
            destination.mkdir()
            sentinel = destination / "data"
            sentinel.write_bytes(b"keep")
            with self.assertRaises(models.ModelError):
                models.install_snapshot(root, destination)
            self.assertEqual(sentinel.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
