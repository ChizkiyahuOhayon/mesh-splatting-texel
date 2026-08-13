import inspect
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


try:
    import diff_triangle_rasterization as dtr
except ModuleNotFoundError:
    source_package = "diff_triangle_rasterization"
    package_name = "_gorfe_export_api_wrapper"
    package_root = (
        Path(__file__).parents[1]
        / "submodules"
        / "diff-triangle-mesh-rasterization"
        / source_package
    )
    native_stub = types.ModuleType(f"{package_name}._C")
    sys.modules[f"{package_name}._C"] = native_stub
    specification = importlib.util.spec_from_file_location(
        package_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    dtr = importlib.util.module_from_spec(specification)
    sys.modules[package_name] = dtr
    specification.loader.exec_module(dtr)


DIAGNOSTICS = torch.tensor(
    [1, 5, 4, 5, 4, 0, 0, 0, 0, 16, 1, 2], dtype=torch.int64
)


class GoRFEExportAPITest(unittest.TestCase):
    def setUp(self):
        settings = dtr.TriangleRasterizationSettings(
            image_height=4,
            image_width=4,
            tanfovx=1.0,
            tanfovy=1.0,
            bg=torch.zeros(3),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4),
            projmatrix=torch.eye(4),
            sh_degree=0,
            campos=torch.zeros(3),
            prefiltered=False,
            debug=False,
        )
        self.rasterizer = dtr.TriangleRasterizer(settings)
        self.vertices = torch.zeros((3, 3), dtype=torch.float32)
        self.faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.weights = torch.ones(3, dtype=torch.float32)
        self.colors = torch.zeros((3, 3), dtype=torch.float32)
        self.scaling = torch.zeros(1, dtype=torch.float32)
        self.candidate_map = torch.tensor([[0, -1, -1]], dtype=torch.int32)

    def _arguments(self):
        return dict(
            vertices=self.vertices,
            triangles_indices=self.faces,
            vertex_weights=self.weights,
            sigma=1e-4,
            scaling=self.scaling,
            colors_precomp=self.colors,
            gorfe_face_edge_ids=self.candidate_map,
            gorfe_edge_count=1,
            output_height=1,
            output_width=1,
            output_scaling=4,
        )

    @staticmethod
    def _forward_result(*args):
        faces = args[2]
        height, width = args[18:20]
        return (
            7,
            torch.zeros((3, height, width)),
            torch.zeros((7, height, width)),
            torch.zeros(faces.shape[0], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.zeros(3, dtype=torch.uint8),
            torch.zeros(4, dtype=torch.uint8),
            torch.zeros(5, dtype=torch.uint8),
            args[13],
            torch.zeros(faces.shape[0]),
        )

    def test_default_forward_signature_is_unchanged(self):
        parameters = tuple(inspect.signature(dtr.TriangleRasterizer.forward).parameters)
        self.assertEqual(
            parameters,
            (
                "self",
                "vertices",
                "triangles_indices",
                "vertex_weights",
                "sigma",
                "scaling",
                "shs",
                "colors_precomp",
                "texels",
                "edge_details",
                "face_edge_ids",
                "window_donors",
            ),
        )

    def test_candidate_map_is_not_passed_to_parent_forward(self):
        captured = {}

        def fake_forward(*args):
            captured["parent_face_edge_ids"] = args[9]
            return self._forward_result(*args)

        def fake_export(*args):
            captured["candidate_map"] = args[3]
            captured["native_export_argument_count"] = len(args)
            return (
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.ones((1, 4), dtype=torch.float32),
                DIAGNOSTICS.clone(),
            )

        with mock.patch.object(
            dtr._C, "rasterize_triangles", fake_forward, create=True
        ), mock.patch.object(
            dtr._C, "export_gorfe_rows", fake_export, create=True
        ):
            result = self.rasterizer.forward_with_gorfe_design(**self._arguments())

        self.assertEqual(len(result), 10)
        self.assertEqual(captured["parent_face_edge_ids"].numel(), 0)
        self.assertIs(captured["candidate_map"], self.candidate_map)
        self.assertEqual(captured["native_export_argument_count"], 16)
        self.assertEqual(result[6].dtype, torch.int32)
        self.assertEqual(result[7].dtype, torch.int32)
        self.assertEqual(result[8].shape, (1, 4))
        self.assertEqual(result[9]["forward_alpha_accepted_fragments"], 5)

    def test_saved_forward_acceptance_mismatch_is_refused(self):
        mismatched = DIAGNOSTICS.clone()
        mismatched[1] = 4

        with mock.patch.object(
            dtr._C, "rasterize_triangles", self._forward_result, create=True
        ), mock.patch.object(
            dtr._C,
            "export_gorfe_rows",
            return_value=(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.ones((1, 4), dtype=torch.float32),
                mismatched,
            ),
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "accepted count disagrees"):
                self.rasterizer.forward_with_gorfe_design(**self._arguments())

    def test_noncontiguous_camera_center_is_normalized_for_native_export(self):
        camera_center = torch.arange(6, dtype=torch.float32).reshape(3, 2)[:, 0]
        self.assertFalse(camera_center.is_contiguous())
        original_settings = self.rasterizer.raster_settings
        self.rasterizer.raster_settings = original_settings._replace(
            campos=camera_center
        )
        captured = {}

        def fake_export(*args):
            captured["campos"] = args[10]
            return (
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.ones((1, 4), dtype=torch.float32),
                DIAGNOSTICS.clone(),
            )

        try:
            with mock.patch.object(
                dtr._C, "rasterize_triangles", self._forward_result, create=True
            ), mock.patch.object(
                dtr._C, "export_gorfe_rows", fake_export, create=True
            ):
                self.rasterizer.forward_with_gorfe_design(**self._arguments())
        finally:
            self.rasterizer.raster_settings = original_settings

        self.assertTrue(captured["campos"].is_contiguous())
        self.assertTrue(torch.equal(captured["campos"], camera_center))

    def test_non_parent_inputs_fail_before_native_forward(self):
        with mock.patch.object(
            dtr._C,
            "rasterize_triangles",
            side_effect=AssertionError("native called"),
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "output_scaling"):
                self.rasterizer.forward_with_gorfe_design(
                    **{**self._arguments(), "output_scaling": 2}
                )
            with self.assertRaisesRegex(ValueError, "window donors"):
                self.rasterizer.forward_with_gorfe_design(
                    **{
                        **self._arguments(),
                        "window_donors": (
                            torch.empty(0, dtype=torch.int32),
                            torch.empty((0, 3), dtype=torch.int32),
                            1,
                        ),
                    }
                )
            with self.assertRaisesRegex(ValueError, "texel-free"):
                self.rasterizer.forward_with_gorfe_design(
                    **{
                        **self._arguments(),
                        "texels": torch.zeros((1, 1, 3), dtype=torch.float32),
                    }
                )
            with self.assertRaisesRegex(ValueError, "edge-detail-free"):
                self.rasterizer.forward_with_gorfe_design(
                    **{
                        **self._arguments(),
                        "edge_details": torch.zeros((1, 3), dtype=torch.float32),
                        "face_edge_ids": self.candidate_map,
                    }
                )

            original_settings = self.rasterizer.raster_settings
            self.rasterizer.raster_settings = original_settings._replace(texel_order=1)
            try:
                with self.assertRaisesRegex(ValueError, "texel-free"):
                    self.rasterizer.forward_with_gorfe_design(**self._arguments())
            finally:
                self.rasterizer.raster_settings = original_settings


if __name__ == "__main__":
    unittest.main()
