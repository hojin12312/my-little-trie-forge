import os
import unittest
from install import models

class ReleaseContract(unittest.TestCase):
    def test_reject_unsafe_repository(self):
        for name in ("../private", "owner/../repo", "/etc/passwd", "owner/repo.git"):
            with self.assertRaises(models.ModelError):
                models.validate_repo_id(name)

    def test_public_model_identity(self):
        self.assertEqual(models.validate_repo_id("local/Qwen3.8-27B-q8c"), "local/Qwen3.8-27B-q8c")
