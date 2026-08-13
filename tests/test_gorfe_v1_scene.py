import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from gorfe_v1_prepare_core import (
    ColmapMetadataSplit,
    MetadataCamera,
    OfficialSplitGuard,
    TargetAccessError,
)
from gorfe_v1_scene import (
    LOCKED_SCALING,
    load_frozen_triangle_model,
    load_training_rgb,
    make_target_free_minicam,
    resolution_minus_one_size,
)
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


def _camera(
    name="train",
    *,
    width=2001,
    height=1001,
    model="PINHOLE",
    parameters=(1000.0, 900.0, 1000.0, 500.0),
    qvec=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
    tvec=(1.0, 2.0, 3.0),
):
    return MetadataCamera(
        image_id=7,
        camera_id=3,
        image_name=name,
        relative_image_path=f"nested/{name}.png",
        width=width,
        height=height,
        model=model,
        parameters=parameters,
        qvec=qvec,
        tvec=tvec,
        fold=0,
    )


def _split(train_camera):
    return ColmapMetadataSplit(
        train_cameras=(train_camera,),
        test_names=("heldout",),
        train_name_sha256="train",
        test_name_sha256="test",
        fold_map_sha256="fold",
        fold_sizes=(1, 0, 0, 0),
        metadata_format="fixture",
    )


def _checkpoint_state():
    faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int64)
    return {
        "triangles_points": torch.arange(12, dtype=torch.float64).reshape(4, 3),
        "_triangle_indices": faces,
        "vertex_weight": torch.tensor([[0.0], [1.0], [-1.0], [2.0]], dtype=torch.float64),
        "sigma": math.log(1e-4),
        "active_sh_degree": 3,
        "features_dc": torch.arange(12, dtype=torch.float64).reshape(4, 1, 3),
        "features_rest": torch.arange(180, dtype=torch.float64).reshape(4, 15, 3),
        "importance_score": torch.zeros(2, dtype=torch.float32),
        "image_size": torch.zeros(2, dtype=torch.float32),
        "pixel_count": torch.zeros(2, dtype=torch.int64),
        "texel_order": 0,
    }


class _FakeTriangleModel:
    def __init__(self, sh_degree):
        self.max_sh_degree = sh_degree
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._triangles = torch.empty(0)
        self._texels = torch.empty(0)
        self._sigma = 0.0
        self.active_sh_degree = 0
        self.texel_order = 0
        self.scaling = 1
        self.optimizer = None
        self.texel_optimizer = None
        self.image_size = 0
        self.importance_score = 0
        self.pixel_count = 0
        self.eps = 1e-6
        self.opacity_floor = 0.0

    @property
    def get_sigma(self):
        return math.exp(self._sigma)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_vertex_weight(self):
        return self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(
            self.vertex_weight
        )


class CameraGeometryTest(unittest.TestCase):
    def test_resolution_minus_one_matches_the_published_1600_rule(self):
        self.assertEqual(resolution_minus_one_size(1200, 701), (1200, 701))
        global_down = 2001 / 1600
        self.assertEqual(
            resolution_minus_one_size(2001, 1001),
            (int(2001 / global_down), int(1001 / global_down)),
        )
        with self.assertRaises(ValueError):
            resolution_minus_one_size(0, 10)

    def test_metadata_camera_matches_existing_camera_transforms(self):
        metadata = _camera()
        camera = make_target_free_minicam(metadata, uid=11, device="cpu")

        # Known +90-degree z quaternion.  Dataset CameraInfo stores the
        # transpose of this quaternion rotation.
        rotation = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        translation = np.asarray(metadata.tvec)
        fovx = focal2fov(metadata.parameters[0], metadata.width)
        fovy = focal2fov(metadata.parameters[1], metadata.height)
        expected_world = torch.tensor(getWorld2View2(rotation, translation)).transpose(0, 1)
        expected_projection = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
        ).transpose(0, 1)
        expected_full = expected_world.unsqueeze(0).bmm(
            expected_projection.unsqueeze(0)
        ).squeeze(0)

        self.assertTrue(torch.allclose(camera.world_view_transform, expected_world, atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(camera.full_proj_transform, expected_full, atol=1e-7, rtol=0))
        self.assertTrue(
            torch.allclose(camera.camera_center, torch.inverse(expected_world)[3, :3], atol=1e-7, rtol=0)
        )
        self.assertEqual(
            (camera.image_width, camera.image_height),
            resolution_minus_one_size(metadata.width, metadata.height),
        )
        self.assertEqual(camera.uid, 11)
        self.assertEqual(camera.colmap_id, metadata.camera_id)
        self.assertFalse(hasattr(camera, "original_image"))

    def test_simple_pinhole_uses_one_focal_for_both_axes(self):
        metadata = _camera(
            width=800,
            height=600,
            model="SIMPLE_PINHOLE",
            parameters=(500.0, 400.0, 300.0),
        )
        camera = make_target_free_minicam(metadata, uid=0, device="cpu")
        self.assertEqual(camera.FoVx, focal2fov(500.0, 800))
        self.assertEqual(camera.FoVy, focal2fov(500.0, 600))

    def test_invalid_camera_model_is_refused(self):
        metadata = _camera(model="OPENCV", parameters=(1.0,) * 8)
        with self.assertRaisesRegex(ValueError, "SIMPLE_PINHOLE"):
            make_target_free_minicam(metadata, uid=0, device="cpu")


class TrainingTargetTest(unittest.TestCase):
    def test_rgb_decode_occurs_only_after_guard_and_matches_pil_to_torch(self):
        metadata = _camera(
            width=1601,
            height=3,
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 0.0),
        )
        guard = OfficialSplitGuard(_split(metadata))
        array = np.arange(1601 * 3 * 3, dtype=np.uint32).reshape(3, 1601, 3)
        array = (array % 256).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory)
            image_path = image_root / "train.png"
            Image.fromarray(array, mode="RGB").save(image_path)
            resolution = resolution_minus_one_size(1601, 3)
            with Image.open(image_path) as image:
                expected = PILtoTorch(image, resolution)
            observed = load_training_rgb(
                metadata, split_guard=guard, image_root=image_root
            )
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(observed.shape, (3, resolution[1], resolution[0]))

    def test_official_test_is_rejected_before_nonexistent_path_is_opened(self):
        train = _camera(name="train", width=10, height=10)
        heldout = _camera(name="heldout", width=10, height=10)
        guard = OfficialSplitGuard(_split(train))
        with self.assertRaisesRegex(TargetAccessError, "official test"):
            load_training_rgb(
                heldout,
                split_guard=guard,
                image_root="/path/that/must/not/be/accessed",
            )

    def test_image_dimension_drift_is_refused(self):
        metadata = _camera(width=10, height=10)
        guard = OfficialSplitGuard(_split(metadata))
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory)
            Image.new("RGB", (9, 10)).save(image_root / "train.png")
            with self.assertRaisesRegex(ValueError, "dimensions drifted"):
                load_training_rgb(
                    metadata, split_guard=guard, image_root=image_root
                )

    def test_non_rgb_target_is_refused(self):
        metadata = _camera(width=10, height=10)
        guard = OfficialSplitGuard(_split(metadata))
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory)
            Image.new("L", (10, 10)).save(image_root / "train.png")
            with self.assertRaisesRegex(ValueError, "not RGB"):
                load_training_rgb(
                    metadata, split_guard=guard, image_root=image_root
                )


class FrozenTriangleModelTest(unittest.TestCase):
    def _load(self, state):
        scene_package = types.ModuleType("scene")
        scene_package.__path__ = []
        triangle_module = types.ModuleType("scene.triangle_model")
        triangle_module.TriangleModel = _FakeTriangleModel
        with mock.patch.dict(
            sys.modules,
            {"scene": scene_package, "scene.triangle_model": triangle_module},
        ):
            return load_frozen_triangle_model(state, device="cpu")

    def test_direct_loader_matches_render_state_without_training_allocations(self):
        state = _checkpoint_state()
        original_vertices = state["triangles_points"].clone()
        model = self._load(state)

        self.assertEqual(model.vertices.dtype, torch.float32)
        self.assertEqual(model._triangle_indices.dtype, torch.int32)
        self.assertTrue(torch.equal(model.vertices, original_vertices.float()))
        self.assertTrue(
            torch.equal(
                model.get_features,
                torch.cat((state["features_dc"], state["features_rest"]), dim=1).float(),
            )
        )
        expected_weight = 0.999 + 0.001 * torch.sigmoid(
            state["vertex_weight"].float()
        )
        self.assertTrue(torch.equal(model.get_vertex_weight, expected_weight))
        self.assertEqual(model.get_sigma, math.exp(state["sigma"]))
        self.assertEqual(model.scaling, LOCKED_SCALING)
        self.assertEqual(model.active_sh_degree, 3)
        self.assertEqual(model.texel_order, 0)
        self.assertIsNone(model.optimizer)
        self.assertIsNone(model.texel_optimizer)
        self.assertFalse(torch.is_tensor(model.image_size))
        self.assertFalse(torch.is_tensor(model.importance_score))
        self.assertFalse(torch.is_tensor(model.pixel_count))
        self.assertTrue(
            all(
                not tensor.requires_grad
                for tensor in (
                    model.vertices,
                    model.vertex_weight,
                    model._features_dc,
                    model._features_rest,
                )
            )
        )

        state["triangles_points"][0, 0] = -999
        self.assertEqual(model.vertices[0, 0], original_vertices[0, 0])

    def test_checkpoint_is_validated_before_triangle_model_import(self):
        state = _checkpoint_state()
        state.pop("features_rest")
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            load_frozen_triangle_model(state, device="cpu")


if __name__ == "__main__":
    unittest.main()
