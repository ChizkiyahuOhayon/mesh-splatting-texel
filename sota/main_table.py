"""Aggregate matched per-scene evaluations into one main-table artifact."""

import json
import sys
from pathlib import Path


METRICS = ("l1", "psnr", "ssim", "lpips_vgg", "fps")
SCENES = ("garden", "room", "bicycle")


def main(root):
    root = Path(root).resolve()
    rows = {}
    for scene in SCENES:
        rows[scene] = {}
        for arm in ("stock", "ours"):
            path = root / scene / arm / "result.json"
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            if result["scene"] != scene or result["arm"] != arm:
                raise ValueError(f"identity mismatch in {path}")
            metrics = result["metrics"]
            rows[scene][arm] = {
                metric: metrics[metric] for metric in METRICS
            } | {
                "checkpoint_bytes": result["checkpoint_bytes"],
                "triangles": result["triangles"],
                "vertices": result["vertices"],
            }
        rows[scene]["delta_ours_minus_stock"] = {
            key: rows[scene]["ours"][key] - rows[scene]["stock"][key]
            for key in (*METRICS, "checkpoint_bytes", "triangles", "vertices")
        }

    wins = {
        scene: {
            "l1": rows[scene]["ours"]["l1"] < rows[scene]["stock"]["l1"],
            "psnr": rows[scene]["ours"]["psnr"] > rows[scene]["stock"]["psnr"],
            "ssim": rows[scene]["ours"]["ssim"] > rows[scene]["stock"]["ssim"],
            "lpips_vgg": (
                rows[scene]["ours"]["lpips_vgg"]
                < rows[scene]["stock"]["lpips_vgg"]
            ),
            "fps": rows[scene]["ours"]["fps"] > rows[scene]["stock"]["fps"],
            "checkpoint_bytes": (
                rows[scene]["ours"]["checkpoint_bytes"]
                < rows[scene]["stock"]["checkpoint_bytes"]
            ),
        }
        for scene in SCENES
    }
    table = {
        "experiment": "matched-main-table-v0",
        "rows": rows,
        "wins": wins,
        "win_counts": {
            metric: sum(wins[scene][metric] for scene in SCENES)
            for metric in wins[SCENES[0]]
        },
    }
    with open(root / "main_table.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sota.main_table OUTPUT_ROOT")
    main(sys.argv[1])
