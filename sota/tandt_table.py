"""Aggregate the matched Tanks & Temples experiment."""

import json
import sys
from pathlib import Path


SCENES = ("train", "truck")
ARMS = ("stock", "ours_speed", "ours_quality")
METRICS = (
    "l1",
    "psnr",
    "ssim",
    "lpips_vgg",
    "fps",
    "checkpoint_bytes",
    "triangles",
    "vertices",
)
LOWER_IS_BETTER = {"l1", "lpips_vgg", "checkpoint_bytes", "triangles", "vertices"}
PAPER = {"psnr": 20.52, "ssim": 0.745, "lpips_vgg": 0.287}


def build_table(root):
    root = Path(root).resolve()
    rows = {scene: {} for scene in SCENES}
    for scene in SCENES:
        for arm in ARMS:
            path = root / scene / arm / "result.json"
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            if result["scene"] != scene or result["arm"] != arm:
                raise ValueError(f"identity mismatch in {path}")
            rows[scene][arm] = {
                key: result["metrics"][key] for key in METRICS[:5]
            } | {
                key: result[key] for key in METRICS[5:]
            }

    means = {
        arm: {
            key: sum(rows[scene][arm][key] for scene in SCENES) / len(SCENES)
            for key in METRICS
        }
        for arm in ARMS
    }
    return {
        "experiment": "formal-tandt-main-table-v1",
        "scenes": list(SCENES),
        "rows": rows,
        "means": means,
        "delta_mean_vs_stock": {
            arm: {
                key: means[arm][key] - means["stock"][key] for key in METRICS
            }
            for arm in ARMS[1:]
        },
        "win_counts_vs_stock": {
            arm: {
                key: sum(
                    rows[scene][arm][key] < rows[scene]["stock"][key]
                    if key in LOWER_IS_BETTER
                    else rows[scene][arm][key] > rows[scene]["stock"][key]
                    for scene in SCENES
                )
                for key in METRICS
            }
            for arm in ARMS[1:]
        },
        "paper_reference": PAPER,
        "delta_mean_vs_paper": {
            arm: {key: means[arm][key] - value for key, value in PAPER.items()}
            for arm in ARMS
        },
    }


def main(root):
    root = Path(root).resolve()
    table = build_table(root)
    with open(root / "formal_table.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sota.tandt_table OUTPUT_ROOT")
    main(sys.argv[1])
