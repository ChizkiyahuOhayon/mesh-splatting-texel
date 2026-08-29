import json
import struct
import tempfile
import unittest
from pathlib import Path

from sota.dtu_table import ARMS, PAPER_CHAMFER, SCANS, build_table


def write_binary_ply(path, vertices, faces):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + struct.pack("<fff", 0.0, 0.0, 0.0))


class DTUTableTest(unittest.TestCase):
    def test_aggregates_all_official_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            runs = Path(directory) / "runs"
            root.mkdir()
            (root / "source_revision.txt").write_text("revision\n")
            for scan_index, scan in enumerate(SCANS):
                for arm_index, arm in enumerate(ARMS):
                    output = root / f"scan{scan}" / arm
                    output.mkdir(parents=True)
                    value = float(scan_index + arm_index + 1)
                    (output / "results.json").write_text(
                        json.dumps(
                            {
                                "mean_d2s": value,
                                "mean_s2d": value + 1.0,
                                "overall": value + 0.5,
                            }
                        ),
                        encoding="utf-8",
                    )
                    write_binary_ply(output / "mesh.ply", 10, 20)

                    run_arm = "stock" if arm == "stock" else "opacity08"
                    checkpoint = (
                        runs
                        / f"{run_arm}__scan{scan}"
                        / "point_cloud"
                        / "iteration_30000"
                        / "point_cloud_state_dict.pt"
                    )
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"checkpoint")

            table = build_table(root, runs)

        self.assertEqual(len(table["scans"]), 15)
        self.assertEqual(table["scans"][0], 24)
        self.assertEqual(table["scans"][-1], 122)
        self.assertEqual(table["means"]["stock"]["chamfer"], 8.5)
        self.assertEqual(
            table["delta_mean_vs_stock"]["ours_quality"]["chamfer"], 1.0
        )
        self.assertEqual(
            table["win_counts_vs_stock"]["ours_quality"]["chamfer"], 0
        )
        self.assertEqual(table["source_revision"], "revision")
        self.assertEqual(table["rows"]["scan24"]["stock"]["vertices"], 10)
        self.assertEqual(table["rows"]["scan24"]["stock"]["triangles"], 20)
        self.assertEqual(PAPER_CHAMFER[24], 0.77)
        self.assertAlmostEqual(table["paper_mean_chamfer"], 0.79, places=2)


if __name__ == "__main__":
    unittest.main()
