import unittest
from pathlib import Path


class SOTANativeContractTest(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.environment = (repo / "sota" / "ensure_environment.sh").read_text(
            encoding="utf-8"
        )
        self.batch = (repo / "sota" / "batch9.sh").read_text(encoding="utf-8")
        self.spatial_batch = (repo / "sota" / "batch10.sh").read_text(encoding="utf-8")
        self.opacity_batch = (repo / "sota" / "batch12.sh").read_text(encoding="utf-8")
        self.opacity_garden_batch = (repo / "sota" / "batch13.sh").read_text(
            encoding="utf-8"
        )
        self.opacity_bicycle_batch = (repo / "sota" / "batch14.sh").read_text(
            encoding="utf-8"
        )
        self.stock_bicycle_batch = (repo / "sota" / "batch15.sh").read_text(
            encoding="utf-8"
        )
        self.tail_culling_batch = (repo / "sota" / "batch16.sh").read_text(
            encoding="utf-8"
        )
        self.tail_absorption_batch = (repo / "sota" / "batch17.sh").read_text(
            encoding="utf-8"
        )
        self.supersampling_batch = (repo / "sota" / "batch18.sh").read_text(
            encoding="utf-8"
        )
        self.main_table_batch = (repo / "sota" / "batch19.sh").read_text(
            encoding="utf-8"
        )
        self.adaptive_sampling_batch = (repo / "sota" / "batch20.sh").read_text(
            encoding="utf-8"
        )
        self.corrected_main_table_batch = (repo / "sota" / "batch21.sh").read_text(
            encoding="utf-8"
        )
        self.quality_main_table_batch = (repo / "sota" / "batch22.sh").read_text(
            encoding="utf-8"
        )
        self.balanced_main_table_batch = (repo / "sota" / "batch23.sh").read_text(
            encoding="utf-8"
        )
        self.formal_training_batch = (repo / "sota" / "batch24.sh").read_text(
            encoding="utf-8"
        )
        self.formal_evaluation_batch = (repo / "sota" / "batch25.sh").read_text(
            encoding="utf-8"
        )
        self.opacity_ablation_batch = (repo / "sota" / "batch26.sh").read_text(
            encoding="utf-8"
        )
        self.overnight_batch = (repo / "sota" / "batch27.sh").read_text(
            encoding="utf-8"
        )
        self.qualitative = (repo / "sota" / "qualitative.py").read_text(
            encoding="utf-8"
        )
        self.runner = (repo / "sota" / "run.sh").read_text(encoding="utf-8")
        self.triangle_model = (repo / "scene" / "triangle_model.py").read_text(
            encoding="utf-8"
        )
        self.renderer = (repo / "triangle_renderer" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.native_forward = (
            repo / "submodules" / "diff-triangle-mesh-rasterization"
            / "cuda_rasterizer" / "forward.cu"
        ).read_text(encoding="utf-8")

    def test_stale_native_install_is_rebuilt(self):
        self.assertIn(
            "rev-parse HEAD:submodules/diff-triangle-mesh-rasterization",
            self.environment,
        )
        self.assertIn('"screen_space_gradients"', self.environment)
        self.assertIn('"transmittance_threshold"', self.environment)
        self.assertIn('"absorb_transmittance_tail"', self.environment)
        self.assertIn('"sigma_face"', self.environment)
        self.assertIn("--force-reinstall", self.environment)

    def test_one_interpreter_is_used_for_preflight_and_training(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.batch)
        self.assertIn('TRAIN_PYTHON=${MESH_SPLATTING_PYTHON:-python}', self.runner)
        self.assertIn('"$TRAIN_PYTHON" -u train.py', self.runner)
        self.assertIn('grep "Evaluating test" "$OUT/metrics.txt" | tail -1', self.runner)

    def test_nvcc_is_bound_to_the_pytorch_environment(self):
        self.assertIn("SOTA_CUDA_HOME=$(\"$SOTA_PYTHON\"", self.environment)
        self.assertIn('export CUDA_HOME=$SOTA_CUDA_HOME', self.environment)
        self.assertIn('export PATH="$CUDA_HOME/bin:$PATH"', self.environment)
        self.assertIn("nvcc/PyTorch CUDA mismatch", self.environment)

    def test_spatial_detail_pilot_is_one_locked_arm(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.spatial_batch)
        self.assertIn('"$HERE/run.sh" tex2_sh2', self.spatial_batch)
        self.assertIn(
            "--max_points 2800000 --texel_order 2 --sh_degree 2",
            self.spatial_batch,
        )

    def test_terminal_opacity_pilot_changes_one_training_argument(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.opacity_batch)
        self.assertIn(
            '"$HERE/run.sh" opacity08 room --final_opacity 0.8',
            self.opacity_batch,
        )
        self.assertIn('state["opacity_floor"]', self.opacity_batch)
        self.assertIn("value != 0.8", self.opacity_batch)

    def test_terminal_opacity_is_persisted_with_legacy_compatibility(self):
        self.assertIn(
            'point_cloud_state_dict["opacity_floor"] = float(self.opacity_floor)',
            self.triangle_model,
        )
        self.assertIn(
            'state.get("opacity_floor", 0.999)',
            self.triangle_model,
        )
        self.assertIn(
            "self.opacity_floor = restored_opacity_floor",
            self.triangle_model,
        )

    def test_garden_transfer_keeps_the_room_setting(self):
        self.assertIn(
            '"$HERE/run.sh" opacity08 garden --final_opacity 0.8',
            self.opacity_garden_batch,
        )
        self.assertIn('state["opacity_floor"]', self.opacity_garden_batch)

    def test_bicycle_transfer_keeps_the_same_setting(self):
        self.assertIn(
            '"$HERE/run.sh" opacity08 bicycle --final_opacity 0.8',
            self.opacity_bicycle_batch,
        )
        self.assertIn('state["opacity_floor"]', self.opacity_bicycle_batch)

    def test_bicycle_control_uses_the_published_default(self):
        self.assertIn(
            '"$HERE/run.sh" stock bicycle',
            self.stock_bicycle_batch,
        )
        self.assertNotIn("--final_opacity", self.stock_bicycle_batch)
        self.assertIn("value != 0.9999", self.stock_bicycle_batch)

    def test_tail_culling_uses_the_frozen_room_checkpoint(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.tail_culling_batch)
        self.assertIn(
            "opacity_floor_01/opacity08__room",
            self.tail_culling_batch,
        )
        self.assertIn("-m sota.tail_culling", self.tail_culling_batch)

    def test_tail_cutoff_is_explicit_and_the_default_is_preserved(self):
        self.assertIn("transmittance_threshold_override=None", self.renderer)
        self.assertIn(
            "1e-4 if transmittance_threshold_override is None",
            self.renderer,
        )
        self.assertIn("test_T < transmittance_threshold", self.native_forward)
        self.assertNotIn("test_T < 0.0001f", self.native_forward)

    def test_tail_absorption_is_a_frozen_evaluation_arm(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.tail_absorption_batch)
        self.assertIn("opacity_floor_01/opacity08__room", self.tail_absorption_batch)
        self.assertIn("--absorb-tail", self.tail_absorption_batch)
        self.assertIn("absorb_tail ? T : alpha * T", self.native_forward)
        self.assertIn("T = absorb_tail ? 0.0f : test_T", self.native_forward)

    def test_supersampling_screen_reuses_the_frozen_checkpoint(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.supersampling_batch)
        self.assertIn("opacity_floor_01/opacity08__room", self.supersampling_batch)
        self.assertIn("-m sota.supersampling", self.supersampling_batch)

    def test_main_table_uses_one_method_configuration_on_three_scenes(self):
        self.assertIn("for SCENE in garden room bicycle", self.main_table_batch)
        self.assertIn("--arm stock", self.main_table_batch)
        self.assertIn("--arm ours", self.main_table_batch)
        self.assertIn("-m sota.main_table ", self.main_table_batch)

    def test_sampling_rule_transfers_without_retraining(self):
        self.assertIn("for SCENE in garden bicycle", self.adaptive_sampling_batch)
        self.assertIn("-m sota.supersampling", self.adaptive_sampling_batch)
        self.assertNotIn("train.py", self.adaptive_sampling_batch)

    def test_outdoor_evaluation_uses_the_training_image_pyramid(self):
        for batch in (self.main_table_batch, self.adaptive_sampling_batch):
            self.assertIn("images_4", batch)
            self.assertIn('-i "$IMAGES"', batch)
        self.assertIn("main_table_02", self.corrected_main_table_batch)
        self.assertIn('exec "$HERE/batch19.sh"', self.corrected_main_table_batch)

    def test_quality_main_table_changes_only_the_global_sampling_factor(self):
        self.assertIn("METHOD_UPSAMPLE=${METHOD_UPSAMPLE:-3}", self.main_table_batch)
        self.assertIn('--method-upsample "$METHOD_UPSAMPLE"', self.main_table_batch)
        self.assertIn("export METHOD_UPSAMPLE=4", self.quality_main_table_batch)
        self.assertIn('exec "$HERE/batch19.sh"', self.quality_main_table_batch)

    def test_balanced_main_table_changes_only_the_global_tail_threshold(self):
        self.assertIn(
            "METHOD_THRESHOLD=${METHOD_THRESHOLD:-0.01}", self.main_table_batch
        )
        self.assertIn('--method-threshold "$METHOD_THRESHOLD"', self.main_table_batch)
        self.assertIn("export METHOD_UPSAMPLE=4", self.balanced_main_table_batch)
        self.assertIn("export METHOD_THRESHOLD=0.03", self.balanced_main_table_batch)
        self.assertIn('exec "$HERE/batch19.sh"', self.balanced_main_table_batch)

    def test_formal_training_adds_only_the_six_missing_scene_arms(self):
        self.assertIn(
            "MISSING_SCENES=(flowers stump treehill counter kitchen bonsai)",
            self.formal_training_batch,
        )
        self.assertIn('"$HERE/run.sh" stock "$SCENE"', self.formal_training_batch)
        self.assertIn(
            '"$HERE/run.sh" opacity08 "$SCENE" --final_opacity 0.8',
            self.formal_training_batch,
        )
        self.assertIn("for SCENE in bicycle garden room", self.formal_training_batch)
        self.assertIn("STOCK_GPU=${STOCK_GPU:-0}", self.formal_training_batch)
        self.assertIn("METHOD_GPU=${METHOD_GPU:-1}", self.formal_training_batch)
        self.assertIn("--untracked-files=no", self.formal_training_batch)

    def test_training_lock_is_scoped_to_one_arm_and_scene(self):
        self.assertIn('LOCK="$RUNS/.${ARM}__${SCENE}.lock"', self.runner)
        self.assertNotIn('exec 9> "$RUNS/.lock"', self.runner)

    def test_new_runs_record_the_source_revision(self):
        self.assertIn('git rev-parse HEAD > "$OUT/source_revision.txt"', self.runner)

    def test_formal_evaluation_uses_two_fixed_operating_points(self):
        self.assertIn("for SCENE in", self.formal_evaluation_batch)
        self.assertIn("ours_speed", self.formal_evaluation_batch)
        self.assertIn("ours_quality", self.formal_evaluation_batch)
        self.assertIn("-m sota.formal_table", self.formal_evaluation_batch)

    def test_formal_evaluation_serializes_arms_on_one_gpu(self):
        self.assertIn(
            'if [ "$STOCK_GPU" = "$METHOD_GPU" ]',
            self.formal_evaluation_batch,
        )
        self.assertIn("  evaluate_stock\n  evaluate_method", self.formal_evaluation_batch)

    def test_opacity_ablation_changes_no_inference_setting(self):
        self.assertIn("--arm ours_opacity", self.opacity_ablation_batch)
        self.assertIn("-m sota.opacity_ablation", self.opacity_ablation_batch)

    def test_overnight_batch_is_a_two_scene_sensitivity_study(self):
        self.assertIn("for SCENE in bicycle room", self.overnight_batch)
        self.assertIn("for FLOOR in 07 09", self.overnight_batch)
        self.assertIn("-m sota.opacity_sensitivity", self.overnight_batch)

    def test_qualitative_export_uses_all_sorted_test_views(self):
        self.assertIn("sorted(scene.getTestCameras()", self.qualitative)
        self.assertIn('ARMS = ("stock", "ours_quality")', self.qualitative)
        self.assertIn("ERROR_SCALE = 4.0", self.qualitative)


if __name__ == "__main__":
    unittest.main()
