import json
from pathlib import Path
import tempfile
import unittest

from gorfe_v1 import BUDGETS, FAMILY_NAMES, SELECTOR_NAMES, decide_scene_family
from gorfe_v1_decide import decide


def _metrics(family):
    zeros = {name: [0.0] * 4 for name in SELECTOR_NAMES}
    return {
        "family": family,
        "eligible_cost_units": 0,
        "metrics_applicable": False,
        "rho": zeros,
        "portfolio": {
            str(budget): {name: [0.0] * 4 for name in SELECTOR_NAMES}
            for budget in BUDGETS
        },
        "small_budget_jaccard_primary_permuted": [1.0] * 4,
    }


def _result(scene):
    families = {}
    for family in FAMILY_NAMES:
        metrics = _metrics(family)
        families[family] = {"metrics": metrics, "decision": decide_scene_family(metrics)}
    return {
        "experiment": "GoRFE-V1",
        "phase": "evaluate",
        "scene": scene,
        "decision": "fail",
        "passing_families": [],
        "families": families,
    }


class OverallDecisionTest(unittest.TestCase):
    def test_two_valid_scene_failures_produce_a_valid_overall_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for scene in ("garden", "room"):
                paths[scene] = root / f"{scene}.json"
                paths[scene].write_text(json.dumps(_result(scene)), encoding="utf-8")
            observed = decide(paths["garden"], paths["room"])
        self.assertEqual(observed["decision"], "fail")
        self.assertIsNone(observed["advanced_family"])
        self.assertEqual(set(observed["scene_result_sha256"]), {"garden", "room"})

    def test_a_stored_scene_decision_cannot_disagree_with_its_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            garden = _result("garden")
            garden["families"]["DC"]["decision"]["pass"] = True
            garden_path = root / "garden.json"
            room_path = root / "room.json"
            garden_path.write_text(json.dumps(garden), encoding="utf-8")
            room_path.write_text(json.dumps(_result("room")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stored decision"):
                decide(garden_path, room_path)


if __name__ == "__main__":
    unittest.main()
