"""Apply the locked RITS-T0 decision rule to the six per-arm results files."""

import json
from argparse import ArgumentParser
from pathlib import Path


MIN_MEAN_GAIN_VS_UNSPLIT = 0.15
MAX_SCENE_REGRESSION_VS_UNSPLIT = 0.05
MIN_MEAN_MARGIN_VS_ABRUPT = 0.05
SCENES = ("garden", "room")
ARMS = ("unsplit", "abrupt", "rits")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        results = json.load(handle)
    return results["final_metrics"]["mean"]


def decide(metrics):
    """metrics[scene][arm] -> {psnr, ssim, lpips_vgg}; returns the locked verdict."""
    psnr = {
        scene: {arm: metrics[scene][arm]["psnr"] for arm in ARMS} for scene in SCENES
    }
    lpips = {
        scene: {arm: metrics[scene][arm]["lpips_vgg"] for arm in ARMS}
        for scene in SCENES
    }
    mean_gain_vs_unsplit = sum(
        psnr[scene]["rits"] - psnr[scene]["unsplit"] for scene in SCENES
    ) / len(SCENES)
    mean_margin_vs_abrupt = sum(
        psnr[scene]["rits"] - psnr[scene]["abrupt"] for scene in SCENES
    ) / len(SCENES)
    checks = {
        "mean_psnr_gain_vs_unsplit_at_least_0p15": mean_gain_vs_unsplit
        >= MIN_MEAN_GAIN_VS_UNSPLIT,
        "no_scene_below_unsplit_by_more_than_0p05": all(
            psnr[scene]["rits"] >= psnr[scene]["unsplit"] - MAX_SCENE_REGRESSION_VS_UNSPLIT
            for scene in SCENES
        ),
        "lpips_improves_vs_unsplit_on_both_scenes": all(
            lpips[scene]["rits"] < lpips[scene]["unsplit"] for scene in SCENES
        ),
        "psnr_beats_abrupt_on_both_scenes": all(
            psnr[scene]["rits"] > psnr[scene]["abrupt"] for scene in SCENES
        ),
        "mean_psnr_margin_vs_abrupt_at_least_0p05": mean_margin_vs_abrupt
        >= MIN_MEAN_MARGIN_VS_ABRUPT,
        "lpips_not_worse_than_abrupt_on_either_scene": all(
            lpips[scene]["rits"] <= lpips[scene]["abrupt"] for scene in SCENES
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "mean_psnr_gain_vs_unsplit": mean_gain_vs_unsplit,
        "mean_psnr_margin_vs_abrupt": mean_margin_vs_abrupt,
        "per_scene": {
            scene: {arm: metrics[scene][arm] for arm in ARMS} for scene in SCENES
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="RITS-T0 locked decision")
    for scene in SCENES:
        for arm in ARMS:
            parser.add_argument(f"--{scene}_{arm}", required=True)
    args = parser.parse_args()
    verdict = decide(
        {
            scene: {
                arm: _load(Path(getattr(args, f"{scene}_{arm}")) / "results.json")
                for arm in ARMS
            }
            for scene in SCENES
        }
    )
    print(json.dumps(verdict, indent=2))
