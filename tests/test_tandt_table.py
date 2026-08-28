import json
import tempfile
import unittest
from pathlib import Path

from sota.tandt_table import ARMS, METRICS, SCENES, build_table


class TanksAndTemplesTableTest(unittest.TestCase):
    def test_aggregates_matched_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene_index, scene in enumerate(SCENES):
                for arm_index, arm in enumerate(ARMS):
                    values = {
                        key: float(scene_index + arm_index + metric_index + 1)
                        for metric_index, key in enumerate(METRICS)
                    }
                    result = {
                        "scene": scene,
                        "arm": arm,
                        "metrics": {key: values[key] for key in METRICS[:5]},
                    } | {key: values[key] for key in METRICS[5:]}
                    path = root / scene / arm
                    path.mkdir(parents=True)
                    (path / "result.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )

            table = build_table(root)

        self.assertEqual(table["scenes"], ["train", "truck"])
        self.assertEqual(table["means"]["stock"]["psnr"], 2.5)
        self.assertEqual(table["delta_mean_vs_stock"]["ours_quality"]["psnr"], 2.0)
        self.assertEqual(table["win_counts_vs_stock"]["ours_quality"]["psnr"], 2)


if __name__ == "__main__":
    unittest.main()
