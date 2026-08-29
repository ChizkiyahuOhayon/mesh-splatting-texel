"""Aggregate the matched DTU geometry experiment."""

import json
import sys
from pathlib import Path


SCANS = (24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122)
ARMS = ("stock", "ours_quality")
METRICS = (
    "accuracy",
    "completeness",
    "chamfer",
    "checkpoint_bytes",
    "triangles",
    "vertices",
)
LOWER_IS_BETTER = set(METRICS)
PAPER_CHAMFER = {
    24: 0.77,
    37: 0.72,
    40: 0.74,
    55: 0.60,
    63: 0.89,
    65: 1.00,
    69: 0.81,
    83: 1.09,
    97: 1.19,
    105: 0.58,
    106: 0.68,
    110: 0.93,
    114: 0.63,
    118: 0.66,
    122: 0.59,
}
PAPER_MEAN_CHAMFER = 0.79


def ply_counts(path):
    vertices = faces = None
    with open(path, "rb") as handle:
        if handle.readline() != b"ply\n":
            raise ValueError(f"not a PLY file: {path}")
        for raw_line in handle:
            line = raw_line.decode("ascii").strip()
            if line.startswith("element vertex "):
                vertices = int(line.rsplit(" ", 1)[1])
            elif line.startswith("element face "):
                faces = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
    if vertices is None or faces is None:
        raise ValueError(f"missing vertex or face count in {path}")
    return vertices, faces


def build_table(root, runs):
    root = Path(root).resolve()
    runs = Path(runs).resolve()
    rows = {}
    for scan in SCANS:
        scene = f"scan{scan}"
        rows[scene] = {}
        for arm in ARMS:
            output = root / scene / arm
            with open(output / "results.json", encoding="utf-8") as handle:
                result = json.load(handle)
            vertices, triangles = ply_counts(output / "mesh.ply")
            run_arm = "stock" if arm == "stock" else "opacity08"
            checkpoint = (
                runs
                / f"{run_arm}__{scene}"
                / "point_cloud"
                / "iteration_30000"
                / "point_cloud_state_dict.pt"
            )
            rows[scene][arm] = {
                "accuracy": result["mean_d2s"],
                "completeness": result["mean_s2d"],
                "chamfer": result["overall"],
                "checkpoint_bytes": checkpoint.stat().st_size,
                "triangles": triangles,
                "vertices": vertices,
            }

    means = {
        arm: {
            metric: sum(rows[f"scan{scan}"][arm][metric] for scan in SCANS)
            / len(SCANS)
            for metric in METRICS
        }
        for arm in ARMS
    }
    return {
        "experiment": "formal-dtu-geometry-v1",
        "source_revision": (root / "source_revision.txt").read_text().strip(),
        "scans": list(SCANS),
        "rows": rows,
        "means": means,
        "delta_mean_vs_stock": {
            "ours_quality": {
                metric: means["ours_quality"][metric] - means["stock"][metric]
                for metric in METRICS
            }
        },
        "win_counts_vs_stock": {
            "ours_quality": {
                metric: sum(
                    rows[f"scan{scan}"]["ours_quality"][metric]
                    < rows[f"scan{scan}"]["stock"][metric]
                    for scan in SCANS
                )
                for metric in LOWER_IS_BETTER
            }
        },
        "paper_chamfer": {f"scan{scan}": PAPER_CHAMFER[scan] for scan in SCANS},
        "paper_mean_chamfer": PAPER_MEAN_CHAMFER,
        "delta_mean_chamfer_vs_paper": means["ours_quality"]["chamfer"]
        - PAPER_MEAN_CHAMFER,
    }


def main(root, runs):
    root = Path(root).resolve()
    table = build_table(root, runs)
    with open(root / "dtu_table.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m sota.dtu_table OUTPUT_ROOT RUNS_ROOT")
    main(sys.argv[1], sys.argv[2])
