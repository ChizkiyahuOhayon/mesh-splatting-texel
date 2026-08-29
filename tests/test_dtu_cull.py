import tempfile
import unittest
from pathlib import Path

from sota.dtu_cull_core import select_dtu_masks


class DTUCullMaskSelectionTest(unittest.TestCase):
    def test_extra_pngs_do_not_change_official_mask_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("000.png", "001.png", "000_extra.png", "001_extra.png"):
                (root / name).touch()

            self.assertEqual(
                [path.name for path in select_dtu_masks(root, 2)],
                ["000.png", "001.png"],
            )

    def test_missing_official_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "000.png").touch()

            with self.assertRaisesRegex(ValueError, "001.png"):
                select_dtu_masks(root, 2)


if __name__ == "__main__":
    unittest.main()
