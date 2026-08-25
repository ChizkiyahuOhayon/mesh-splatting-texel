"""Aggregate the two-scene trained terminal-opacity sensitivity study."""

import json
import sys
from pathlib import Path

from sota.formal_table import METRICS, mean


SCENES = ("bicycle", "room")
ARMS = ("stock", "ours_opacity07", "ours_opacity08", "ours_opacity09")
FLOORS = {
    "stock": 0.9999,
    "ours_opacity07": 0.7,
    "ours_opacity08": 0.8,
    "ours_opacity09": 0.9,
}


def extract(result):
    return {
        key: result["metrics"][key] for key in METRICS[:5]
    } | {
        key: result[key] for key in METRICS[5:]
    }


def main(formal_root, opacity_root, output_root):
    formal_root = Path(formal_root).resolve()
    opacity_root = Path(opacity_root).resolve()
    output_root = Path(output_root).resolve()
    with open(formal_root / "formal_table.json", encoding="utf-8") as handle:
        formal = json.load(handle)
    with open(opacity_root / "ablation_table.json", encoding="utf-8") as handle:
        opacity = json.load(handle)

    rows = {}
    for scene in SCENES:
        rows[scene] = {
            "stock": formal["rows"][scene]["stock"],
            "ours_opacity08": opacity["rows"][scene]["ours_opacity"],
        }
        for arm in ("ours_opacity07", "ours_opacity09"):
            path = output_root / scene / arm / "result.json"
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            if result["scene"] != scene or result["arm"] != arm:
                raise ValueError(f"identity mismatch in {path}")
            rows[scene][arm] = extract(result)

    means = {
        arm: {
            key: mean([rows[scene][arm] for scene in SCENES], key)
            for key in METRICS
        }
        for arm in ARMS
    }
    table = {
        "experiment": "trained-terminal-opacity-sensitivity-v1",
        "scenes": list(SCENES),
        "opacity_floor": FLOORS,
        "rows": rows,
        "means": means,
        "best_mean": {
            "l1": min(ARMS, key=lambda arm: means[arm]["l1"]),
            "psnr": max(ARMS, key=lambda arm: means[arm]["psnr"]),
            "ssim": max(ARMS, key=lambda arm: means[arm]["ssim"]),
            "lpips_vgg": min(ARMS, key=lambda arm: means[arm]["lpips_vgg"]),
        },
    }
    with open(output_root / "sensitivity.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python -m sota.opacity_sensitivity "
            "FORMAL_ROOT OPACITY_ROOT OUTPUT_ROOT"
        )
    main(sys.argv[1], sys.argv[2], sys.argv[3])
