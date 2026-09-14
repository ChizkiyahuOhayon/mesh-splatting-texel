import unittest
from pathlib import Path


class SoftTailV2BatchContractTest(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.gate = (repo / "sota" / "batch33.sh").read_text(encoding="utf-8")
        self.control = (repo / "sota" / "batch34.sh").read_text(encoding="utf-8")
        self.train = (repo / "train.py").read_text(encoding="utf-8")
        self.model = (repo / "scene" / "triangle_model.py").read_text(encoding="utf-8")
        self.eval = (repo / "sota" / "main_table_eval.py").read_text(encoding="utf-8")

    def test_gate_uses_a_new_root_and_one_shared_configuration(self):
        self.assertIn("experiments/softtail_v2_01", self.gate)
        self.assertIn('SCENES=(room bicycle garden)', self.gate)
        self.assertIn('"$HERE/run.sh" adaptive "$SCENE"', self.gate)
        self.assertIn("--final_opacity 0.8 --adaptive_opacity --adaptive_opacity_low", self.gate)
        self.assertNotIn("--adaptive_opacity_control", self.gate)
        for arm in ("adaptive_quality", "adaptive_speed", "adaptive_opacity"):
            self.assertIn(arm, self.gate)
        self.assertIn("sota.v2_gate", self.gate)
        self.assertIn("sota.visibility_diag", self.gate)
        self.assertIn("premise falsified", self.gate)

    def test_gate_reads_v1_roots_and_never_writes_them(self):
        self.assertIn("formal_main_table_01", self.gate)
        self.assertIn("formal_opacity_ablation_01", self.gate)
        self.assertIn("opacity_floor_01", self.gate)
        self.assertNotIn("RUNS=${RUNS:-$NAS_ROOT/experiments/opacity_floor_01}", self.gate)

    def test_control_shuffles_on_room_and_bicycle_only(self):
        self.assertIn("SCENES=(room bicycle)", self.control)
        self.assertIn("--adaptive_opacity_control shuffle", self.control)
        self.assertIn("experiments/softtail_v2_control_01", self.control)
        self.assertIn("sota.v2_gate --control", self.control)

    def test_training_path_is_gated_by_the_flag(self):
        self.assertIn('vato_active = bool(getattr(opt, "adaptive_opacity", False))', self.train)
        self.assertIn("if vato_tracker is not None:", self.train)
        self.assertIn("triangles.update_min_weight(floor_v, reference_floor=current_opacity)", self.train)
        # The pre-Delaunay floor update is untouched.
        self.assertEqual(self.train.count("triangles.update_min_weight(current_opacity)"), 2)

    def test_model_serializes_the_per_vertex_floor(self):
        self.assertIn('point_cloud_state_dict["opacity_floor_vertex"]', self.model)
        self.assertIn('point_cloud_state_dict["adaptive_opacity"]', self.model)
        self.assertIn('point_cloud_state_dict["visibility_dominance"]', self.model)
        self.assertIn('if "opacity_floor_vertex" in state:', self.model)
        self.assertIn('point_cloud_state_dict["opacity_floor"] = float(self.opacity_floor)', self.model)

    def test_eval_arms_enforce_the_checkpoint_contract(self):
        self.assertIn("check_adaptive_contract(", self.eval)
        for arm in ("adaptive_quality", "adaptive_speed", "adaptive_opacity"):
            self.assertIn(f'"{arm}": {{', self.eval)


if __name__ == "__main__":
    unittest.main()
