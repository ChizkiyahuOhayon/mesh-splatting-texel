import os
import subprocess
import unittest
from unittest.mock import patch

from svsr_metadata import source_revision


class SourceRevisionTest(unittest.TestCase):
    def test_archive_revision_override_avoids_git_lookup(self):
        with patch.dict(os.environ, {"SVSR_SOURCE_REVISION": "abc123"}, clear=False):
            with patch("subprocess.check_output") as check_output:
                self.assertEqual(source_revision(), "abc123")
                check_output.assert_not_called()

    def test_archive_without_revision_fails_clearly(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(128, "git")):
                with self.assertRaisesRegex(RuntimeError, "SVSR_SOURCE_REVISION"):
                    source_revision()


if __name__ == "__main__":
    unittest.main()
