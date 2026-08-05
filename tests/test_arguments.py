import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

from arguments import get_combined_args


class GetCombinedArgsTest(unittest.TestCase):
    """`get_combined_args` starts from the config file's namespace and overlays
    only the command-line values that are not None. An optional argument left
    at a None default is therefore absent from the result rather than present
    and None, and reading it raises AttributeError — the failure that cost
    twelve SAC-G1 evaluations. Callers must use `getattr(args, name, default)`.
    """

    def _combined(self, argv_extra=()):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "cfg_args").write_text("Namespace(trained_flag=True)")
            parser = ArgumentParser()
            parser.add_argument("--model_path", "-m", default="")
            parser.add_argument("--optional", default=None)
            parser.add_argument("--defaulted", default=7)
            argv = ["prog", "-m", directory, *argv_extra]
            with patch.object(sys, "argv", argv):
                return get_combined_args(parser)

    def test_a_none_default_is_absent_from_the_namespace(self):
        combined = self._combined()
        self.assertFalse(hasattr(combined, "optional"))

    def test_an_explicit_value_survives(self):
        combined = self._combined(("--optional", "5"))
        self.assertEqual(combined.optional, "5")

    def test_non_none_defaults_and_config_entries_survive(self):
        combined = self._combined()
        self.assertEqual(combined.defaulted, 7)
        self.assertTrue(combined.trained_flag)


if __name__ == "__main__":
    unittest.main()
