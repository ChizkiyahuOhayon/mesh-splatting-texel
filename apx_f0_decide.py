"""APX-F0 reading: is there appearance capacity left, is it concentrated, findable?

Three questions, each able to close the direction on its own, all on held-out pixels.
Kept free of torch's CUDA path so the rule can be exercised without a checkpoint.
Locked by experiments/apx_f0/protocol.md.
"""

CEILING_DB = 0.30
CONCENTRATION_FRACTION = 0.10
CONCENTRATION_CAPTURE = 0.50

# XVR-G0's locked numbers, reused verbatim. A pass here against a fail there is then
# a statement about the two questions differing, not about two different bars.
LIFT_MIN = 1.75
CONTROL_MARGIN = 0.10

PRIMARY_SIGNAL = "residual_mass"
NON_RESIDUAL_CONTROLS = ("max_blending", "projected_coverage", "world_area")

CEILING_ORDER = 4


def scene_checks(scene):
    """The three locked conditions for one scene, each with the number behind it."""
    ceiling = scene["ceiling_db"][str(CEILING_ORDER)]
    concentration = scene["concentration"][f"top_{int(CONCENTRATION_FRACTION * 100)}pct"]
    primary = scene["signals"][PRIMARY_SIGNAL]["top_10pct"]
    best_control = max(
        scene["signals"][name]["top_10pct"]["capture"] for name in NON_RESIDUAL_CONTROLS
    )
    return {
        "ceiling": {
            "pass": ceiling >= CEILING_DB,
            "measured_db": ceiling,
            "required_db": CEILING_DB,
        },
        "concentration": {
            "pass": concentration >= CONCENTRATION_CAPTURE,
            "capture": concentration,
            "required": CONCENTRATION_CAPTURE,
        },
        "predictability": {
            "pass": primary["lift"] >= LIFT_MIN
            and primary["capture"] >= best_control * (1.0 + CONTROL_MARGIN),
            "lift": primary["lift"],
            "capture": primary["capture"],
            "best_non_residual_control": best_control,
            "required_lift": LIFT_MIN,
            "required_capture": best_control * (1.0 + CONTROL_MARGIN),
        },
    }


def decide(scenes):
    """Both scenes must pass all three conditions.

    The first failing condition is named, because the three close the direction for
    different reasons and the difference matters: no capacity left, capacity spread
    too evenly for adaptive allocation to beat the uniform allocation that already
    failed, or capacity that exists and is concentrated but cannot be found cheaply.
    """
    if not scenes:
        return {"decision": "INCONCLUSIVE", "reason": "no scenes measured"}

    per_scene = {name: scene_checks(scene) for name, scene in scenes.items()}
    failures = [
        f"{name}:{condition}"
        for name, checks in sorted(per_scene.items())
        for condition, result in checks.items()
        if not result["pass"]
    ]
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failed_conditions": failures,
        "per_scene": per_scene,
    }
