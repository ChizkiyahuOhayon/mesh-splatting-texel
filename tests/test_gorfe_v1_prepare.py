import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import gorfe_v1_prepare as prepare
from gorfe_v1_prepare_core import (
    CandidateTopology,
    ColmapMetadataSplit,
    MetadataCamera,
    TargetAccessError,
)
from gorfe_v1_state import load_candidate_state


class _Inspection:
    def __init__(self, path):
        self.path = str(path.resolve())
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.bytes = path.stat().st_size
        self.face_count = 1

    def to_manifest(self):
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "keys": ["_triangle_indices"],
            "tensor_shapes": {"_triangle_indices": [1, 3]},
            "tensor_dtypes": {"_triangle_indices": "torch.int32"},
            "vertex_count": 3,
            "face_count": 1,
            "active_sh_degree": 3,
            "texel_order": 0,
            "activated_sigma": 1e-4,
        }


class _Monitor:
    def __init__(self, physical_gpu, *, exclusive_pid=None):
        self.physical_gpu = physical_gpu
        self.exclusive_pid = exclusive_pid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def record(self):
        return {
            "physical_gpu": self.physical_gpu,
            "poll_interval_seconds": 1.0,
            "samples": 2,
            "peak_used_mib": 2000,
            "minimum_free_mib": 44000,
            "exclusive_pid": self.exclusive_pid,
            "observed_compute_pids": [self.exclusive_pid]
            if self.exclusive_pid is not None
            else [],
        }


class PrepareSceneTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        sparse = self.dataset / "sparse" / "0"
        sparse.mkdir(parents=True)
        (self.dataset / "images").mkdir()
        (sparse / "cameras.bin").write_bytes(b"camera metadata\n")
        (sparse / "images.bin").write_bytes(b"image metadata\n")
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        self.checkpoint = self.model_root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"frozen-checkpoint")
        self.extension = self.root / "native.so"
        self.extension.write_bytes(b"native-extension")
        self.output = self.root / "attempt" / "garden"

        cameras = []
        for rank in range(16):
            cameras.append(
                MetadataCamera(
                    image_id=rank + 1,
                    camera_id=1,
                    image_name=f"train_{rank:03d}",
                    relative_image_path=f"train_{rank:03d}.png",
                    width=8,
                    height=4,
                    model="PINHOLE",
                    parameters=(4.0, 4.0, 4.0, 2.0),
                    qvec=(1.0, 0.0, 0.0, 0.0),
                    tvec=(0.0, 0.0, 0.0),
                    fold=rank % 4,
                )
            )
        fold_map = {camera.image_name: camera.fold for camera in cameras}
        self.split = ColmapMetadataSplit(
            train_cameras=tuple(cameras),
            test_names=("held_out",),
            train_name_sha256="train-sha",
            test_name_sha256="test-sha",
            fold_map_sha256="fold-sha",
            fold_sizes=(4, 4, 4, 4),
            metadata_format="colmap_binary",
        )
        self.fold_map = fold_map
        self.inspection = _Inspection(self.checkpoint)
        self.state = {
            "_triangle_indices": torch.tensor([[0, 1, 2]], dtype=torch.int32)
        }
        self.topology = CandidateTopology(
            edge_count=3,
            vertex_stride=3,
            candidate_edge_indices=torch.tensor([0], dtype=torch.int64),
            candidate_edges=torch.tensor([[0, 1]], dtype=torch.int64),
            candidate_hashes=np.array([7], dtype=np.uint64),
            candidate_incident_face_counts=torch.tensor([1], dtype=torch.int32),
            face_candidate_edges=torch.tensor([[0, -1, -1]], dtype=torch.int32),
        )
        self.runtime = {
            "source_revision": "revision-1",
            "tracked_checkout_clean": True,
            "logical_device": "cpu",
            "cuda_context_initialized": False,
            "cuda_visible_devices": "3",
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "python": "test",
            "python_executable": "test-python",
            "platform": "test-platform",
            "hostname": "test-host",
            "hardware_start": {
                "physical_gpu": 3,
                "name": "NVIDIA A40",
                "total_mib": 46068,
                "used_mib": 1000,
                "free_mib": 45000,
                "compute_processes_before_cuda_init": [],
            },
            "native_extension": {
                "path": str(self.extension.resolve()),
                "bytes": self.extension.stat().st_size,
                "sha256": hashlib.sha256(self.extension.read_bytes()).hexdigest(),
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _camera(metadata, *, uid, device):
        return SimpleNamespace(
            uid=uid,
            image_name=metadata.image_name,
            image_width=8,
            image_height=4,
        )

    @staticmethod
    def _model(state, *, device):
        return SimpleNamespace(
            scaling=4,
            active_sh_degree=3,
            texel_order=0,
            optimizer=None,
            texel_optimizer=None,
        )

    @staticmethod
    def _design(view, model, background, face_edge_ids, candidate_count):
        del view
        assert model.scaling == 4
        assert candidate_count == 1
        assert background.device.type == "cpu"
        assert torch.equal(background, torch.zeros(3))
        assert face_edge_ids.dtype == torch.int32
        rows = 32
        sh1 = torch.eye(3, dtype=torch.float32).repeat((rows + 2) // 3, 1)[:rows]
        features = torch.cat((torch.ones((rows, 1)), sh1), dim=1)
        return {
            "gorfe_design": {
                "pixel_ids": torch.arange(rows, dtype=torch.int32),
                "group_ids": torch.zeros(rows, dtype=torch.int32),
                "features": features,
                "diagnostics": {
                    "raw_rows": rows,
                    "count_write_mismatch": 0,
                },
            }
        }

    def _identity(self, scene, split, constants):
        self.assertEqual(scene, "garden")
        self.assertIs(split, self.split)
        return {
            "scene": scene,
            "observed": {
                "train_count": 16,
                "test_count": 1,
                "train_name_sha256": "train-sha",
                "test_name_sha256": "test-sha",
                "fold_map_sha256": "fold-sha",
                "fold_sizes": [4, 4, 4, 4],
            },
            "checks": {"fixture": True},
        }

    @contextlib.contextmanager
    def _patched(self, *, render=None, runtime=None, host_rss=1024):
        runtime = self.runtime if runtime is None else runtime
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(prepare, "read_colmap_metadata", return_value=self.split))
            stack.enter_context(mock.patch.object(prepare, "validate_scene_camera_identity", side_effect=self._identity))
            stack.enter_context(
                mock.patch.object(
                    prepare,
                    "load_validated_checkpoint",
                    return_value=(self.state, self.inspection),
                )
            )
            stack.enter_context(mock.patch.object(prepare, "_runtime_preflight", return_value=runtime))
            stack.enter_context(mock.patch.object(prepare, "build_candidate_topology", return_value=self.topology))
            stack.enter_context(mock.patch.object(prepare, "make_target_free_minicam", side_effect=self._camera))
            stack.enter_context(mock.patch.object(prepare, "load_frozen_triangle_model", side_effect=self._model))
            stack.enter_context(mock.patch.object(prepare, "_render_design", side_effect=render or self._design))
            stack.enter_context(mock.patch.object(prepare, "PhysicalGpuMonitor", _Monitor))
            stack.enter_context(mock.patch.object(prepare, "host_peak_rss_bytes", return_value=host_rss))
            stack.enter_context(mock.patch.object(prepare, "source_revision", return_value="revision-1"))
            stack.enter_context(mock.patch.object(prepare, "_tracked_checkout_is_clean", return_value=True))
            yield

    def test_success_seals_state_and_writes_done_last(self):
        with self._patched():
            result = prepare.prepare_scene(
                scene="garden",
                dataset_root=self.dataset,
                model_root=self.model_root,
                output=self.output,
                physical_gpu=3,
            )
        self.assertEqual(result["decision"], "prepared")
        self.assertEqual((self.output / "DONE").read_text(), "complete\n")
        self.assertFalse((self.output / "INVALID").exists())
        self.assertFalse((self.output / "FAILED").exists())
        for name in (
            "candidate_state.pt",
            "candidate_manifest.json",
            "result.json",
            "SHA256SUMS",
        ):
            self.assertTrue((self.output / name).is_file(), name)

        state, statistics, masks = load_candidate_state(
            self.output / "candidate_state.pt", expected_scene="garden"
        )
        self.assertTrue(torch.equal(state["candidate_edges"], torch.tensor([[0, 1]])))
        self.assertTrue(bool(masks.dc[0]))
        self.assertTrue(bool(masks.sh1[0]))
        self.assertTrue(torch.equal(statistics.fold_full_rss, torch.zeros(4, dtype=torch.float64)))

        manifest = json.loads((self.output / "candidate_manifest.json").read_text())
        self.assertEqual(manifest["schema"], prepare.CANDIDATE_MANIFEST_SCHEMA)
        self.assertEqual(manifest["candidate_state"]["sha256"], result["candidate_state"]["sha256"])
        self.assertEqual(manifest["dataset"]["fold"]["name_to_fold"], self.fold_map)
        self.assertEqual(len(manifest["cameras"]), 16)
        self.assertEqual(manifest["cameras"][0]["reduction"]["input_rows"], 32)
        self.assertEqual(manifest["cameras"][0]["reduction"]["reduced_rows"], 32)
        self.assertEqual(manifest["topology"]["candidate_incidence_histogram"], {"1": 1})
        self.assertEqual(
            manifest["topology"]["incidence_greater_than_two_candidate_count"], 0
        )
        self.assertEqual(manifest["topology"]["maximum_candidate_incidence"], 1)
        self.assertEqual(manifest["resources"]["persistent_artifact_bytes"], prepare.directory_bytes(self.output))
        # Endpoint tensors have exactly one persisted source; JSON artifacts hold
        # only the candidate-state identity and aggregate topology counts.
        self.assertNotIn("candidate_edges", (self.output / "candidate_manifest.json").read_text())
        self.assertNotIn("candidate_edges", (self.output / "result.json").read_text())

    def test_topology_summary_records_the_complete_incidence_distribution(self):
        topology = SimpleNamespace(
            edge_count=8,
            vertex_stride=6,
            candidate_edges=torch.empty((5, 2), dtype=torch.int64),
            candidate_incident_face_counts=torch.tensor(
                [1, 2, 3, 3, 1], dtype=torch.int32
            ),
            face_candidate_edges=torch.tensor(
                [[0, 1, 1], [2, 2, 2], [3, 3, 3], [4, -1, -1]],
                dtype=torch.int32,
            ),
        )
        summary = prepare._topology_summary(topology, face_count=4, cap=5, seed=7)
        self.assertEqual(
            summary["candidate_incidence_histogram"], {"1": 2, "2": 1, "3": 2}
        )
        self.assertEqual(summary["incidence_one_candidate_count"], 2)
        self.assertEqual(summary["incidence_two_candidate_count"], 1)
        self.assertEqual(summary["incidence_greater_than_two_candidate_count"], 2)
        self.assertEqual(summary["maximum_candidate_incidence"], 3)
        self.assertEqual(summary["mapped_face_local_slots"], 10)

    def test_preflight_failure_leaves_no_output_directory(self):
        with self._patched(), mock.patch.object(
            prepare, "_runtime_preflight", side_effect=prepare.PrepareInvalidError("runtime")
        ):
            with self.assertRaises(prepare.PrepareInvalidError):
                prepare.prepare_scene(
                    scene="garden",
                    dataset_root=self.dataset,
                    model_root=self.model_root,
                    output=self.output,
                    physical_gpu=3,
                )
        self.assertFalse(self.output.exists())

    def test_target_attempt_invalidates_and_never_writes_done(self):
        def target_read(*args):
            with open(self.dataset / "images" / "train_000.png", "rb"):
                pass

        with self._patched(render=target_read):
            with self.assertRaises(TargetAccessError):
                prepare.prepare_scene(
                    scene="garden",
                    dataset_root=self.dataset,
                    model_root=self.model_root,
                    output=self.output,
                    physical_gpu=3,
                )
        self.assertTrue((self.output / "INVALID").is_file())
        self.assertFalse((self.output / "DONE").exists())

    def test_caught_target_attempt_is_sticky_and_cannot_reach_done(self):
        def caught_target_read(*args):
            try:
                with open(self.dataset / "images" / "train_000.png", "rb"):
                    pass
            except TargetAccessError:
                pass
            return self._design(*args)

        with self._patched(render=caught_target_read):
            with self.assertRaises(TargetAccessError):
                prepare.prepare_scene(
                    scene="garden",
                    dataset_root=self.dataset,
                    model_root=self.model_root,
                    output=self.output,
                    physical_gpu=3,
                )
        self.assertTrue((self.output / "INVALID").is_file())
        self.assertFalse((self.output / "candidate_state.pt").exists())
        self.assertFalse((self.output / "DONE").exists())

    def test_python_row_overflow_is_invalid_not_failed(self):
        with self._patched(), mock.patch.object(
            prepare.GoRFEV1Accumulator,
            "add_camera",
            side_effect=OverflowError("camera raw-row limit exceeded"),
        ):
            with self.assertRaises(OverflowError):
                prepare.prepare_scene(
                    scene="garden",
                    dataset_root=self.dataset,
                    model_root=self.model_root,
                    output=self.output,
                    physical_gpu=3,
                )
        self.assertIn("OverflowError", (self.output / "INVALID").read_text())
        self.assertFalse((self.output / "FAILED").exists())
        self.assertFalse((self.output / "DONE").exists())

    def test_resource_overrun_writes_invalid_with_complete_provenance(self):
        with self._patched(host_rss=2**70):
            result = prepare.prepare_scene(
                scene="garden",
                dataset_root=self.dataset,
                model_root=self.model_root,
                output=self.output,
                physical_gpu=3,
            )
        self.assertEqual(result["decision"], "invalid")
        self.assertIn("host_peak_rss_bytes", result["resources"]["violations"])
        self.assertTrue((self.output / "INVALID").is_file())
        self.assertFalse((self.output / "DONE").exists())
        self.assertTrue((self.output / "SHA256SUMS").is_file())

    def test_checkpoint_drift_is_failed_before_candidate_state(self):
        calls = 0

        def drift(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.checkpoint.write_bytes(b"drifted")
            return self._design(*args)

        with self._patched(render=drift):
            with self.assertRaisesRegex(Exception, "checkpoint"):
                prepare.prepare_scene(
                    scene="garden",
                    dataset_root=self.dataset,
                    model_root=self.model_root,
                    output=self.output,
                    physical_gpu=3,
                )
        self.assertTrue((self.output / "FAILED").is_file())
        self.assertFalse((self.output / "candidate_state.pt").exists())
        self.assertFalse((self.output / "DONE").exists())

    def test_public_preflight_does_not_create_output_or_cuda_context(self):
        with self._patched() as _:
            record = prepare.preflight_scene(
                scene="garden",
                dataset_root=self.dataset,
                model_root=self.model_root,
                physical_gpu=3,
            )
        self.assertEqual(record["phase"], "prepare-preflight")
        self.assertTrue(record["checks"]["runtime_without_cuda_context"])
        self.assertFalse(self.output.exists())

    def test_metadata_preflight_does_not_import_native_extension(self):
        constants = {
            "torch_version": "2.7.1+cu126",
            "torch_cuda_build": "12.6",
            "resource_limits": {"gpu_free_mib_at_start": 40000},
        }
        hardware = {
            "physical_gpu": 3,
            "name": "NVIDIA A40",
            "total_mib": 46068,
            "used_mib": 0,
            "free_mib": 46000,
            "compute_processes_before_cuda_init": [],
        }
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"}), mock.patch.object(
            prepare.torch, "__version__", "2.7.1+cu126"
        ), mock.patch.object(
            prepare.torch.version, "cuda", "12.6"
        ), mock.patch.object(
            prepare, "_tracked_checkout_is_clean", return_value=True
        ), mock.patch.object(
            prepare, "_physical_gpu_preflight", return_value=hardware
        ), mock.patch.object(
            prepare,
            "_native_extension_identity",
            side_effect=AssertionError("native imported during metadata preflight"),
        ):
            observed = prepare._runtime_preflight(3, constants, initialize_cuda=False)
        self.assertIsNone(observed["native_extension"])
        self.assertFalse(observed["cuda_context_initialized"])

    def test_preflight_only_cli_does_not_require_output(self):
        record = {"phase": "prepare-preflight", "scene": "garden"}
        stdout = io.StringIO()
        with mock.patch.object(prepare, "preflight_scene", return_value=record), contextlib.redirect_stdout(stdout):
            code = prepare.main(
                [
                    "--preflight-only",
                    "--scene",
                    "garden",
                    "--dataset-root",
                    str(self.dataset),
                    "--model-root",
                    str(self.model_root),
                    "--physical-gpu",
                    "3",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), record)


if __name__ == "__main__":
    unittest.main()
