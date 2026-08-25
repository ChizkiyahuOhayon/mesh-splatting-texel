"""Aggregate the formal nine-scene opacity-only ablation."""

import json
import sys
from pathlib import Path

from sota.formal_table import LOWER_IS_BETTER, METRICS, SCENES, mean


ARM = "ours_opacity"


def main(formal_root, output_root):
    formal_root = Path(formal_root).resolve()
    output_root = Path(output_root).resolve()
    with open(formal_root / "formal_table.json", encoding="utf-8") as handle:
        formal = json.load(handle)

    rows = {}
    for scene in SCENES:
        path = output_root / scene / ARM / "result.json"
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        if result["scene"] != scene or result["arm"] != ARM:
            raise ValueError(f"identity mismatch in {path}")
        candidate = {
            key: result["metrics"][key] for key in METRICS[:5]
        } | {
            key: result[key] for key in METRICS[5:]
        }
        rows[scene] = {
            "stock": formal["rows"][scene]["stock"],
            ARM: candidate,
        }

    means = {
        arm: {
            key: mean([rows[scene][arm] for scene in SCENES], key)
            for key in METRICS
        }
        for arm in ("stock", ARM)
    }
    table = {
        "experiment": "formal-nine-scene-opacity-ablation-v1",
        "scenes": list(SCENES),
        "rows": rows,
        "means": means,
        "delta_mean_vs_stock": {
            key: means[ARM][key] - means["stock"][key] for key in METRICS
        },
        "win_counts_vs_stock": {
            key: sum(
                rows[scene][ARM][key] < rows[scene]["stock"][key]
                if key in LOWER_IS_BETTER
                else rows[scene][ARM][key] > rows[scene]["stock"][key]
                for scene in SCENES
            )
            for key in METRICS
        },
    }
    with open(output_root / "ablation_table.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m sota.opacity_ablation FORMAL_ROOT OUTPUT_ROOT"
        )
    main(sys.argv[1], sys.argv[2])
