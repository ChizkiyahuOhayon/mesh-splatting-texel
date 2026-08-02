import unittest

import torch

from csu_split import install_midpoint_probe, midpoint_split_topology, midpoint_values


class DummyModel:
    def __init__(self):
        self.vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            requires_grad=True,
        )
        self.vertex_weight = torch.zeros(3, 1, requires_grad=True)
        self._features_dc = torch.arange(9.0).reshape(3, 1, 3).requires_grad_(True)
        self._features_rest = torch.arange(18.0).reshape(3, 2, 3).requires_grad_(True)
        self._triangle_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.texel_order = 0
        self.opacity_floor = 0.0
        self.eps = 1e-6
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.optimizer = object()
        self.image_size = torch.zeros(1)
        self.importance_score = torch.zeros(1)
        self.pixel_count = torch.zeros(1, dtype=torch.int)

    def validate_face_state(self):
        assert self.image_size.shape[0] == self._triangle_indices.shape[0]


class CSUSplitTest(unittest.TestCase):
    def test_single_parent_becomes_four_children(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        result = midpoint_split_topology(faces, torch.tensor([0]), vertex_count=3)

        self.assertEqual(result["unique_edges"].shape[0], 3)
        self.assertEqual(result["triangle_indices"].shape, (4, 3))
        self.assertEqual(int(result["triangle_indices"].min()), 0)
        self.assertEqual(int(result["triangle_indices"].max()), 5)

    def test_adjacent_parents_share_one_midpoint(self):
        faces = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.int32)
        result = midpoint_split_topology(faces, torch.tensor([1, 0]), vertex_count=4)

        self.assertEqual(result["unique_edges"].shape[0], 5)
        self.assertEqual(result["triangle_indices"].shape, (8, 3))
        shared_edge = torch.tensor([1, 2])
        shared_row = torch.nonzero(
            torch.all(result["unique_edges"] == shared_edge, dim=1), as_tuple=True
        )[0]
        self.assertEqual(shared_row.numel(), 1)
        shared_midpoint = 4 + int(shared_row[0])
        self.assertGreaterEqual(
            int((result["triangle_indices"] == shared_midpoint).sum()), 4
        )

    def test_midpoint_values_average_endpoints(self):
        values = torch.tensor([[0.0], [2.0], [6.0]])
        edges = torch.tensor([[0, 1], [1, 2]])
        torch.testing.assert_close(midpoint_values(values, edges), torch.tensor([[1.0], [4.0]]))

    def test_rejects_invalid_selection(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        with self.assertRaises(ValueError):
            midpoint_split_topology(faces, torch.tensor([], dtype=torch.long), 3)
        with self.assertRaises(IndexError):
            midpoint_split_topology(faces, torch.tensor([1]), 3)

    def test_installed_probe_preserves_prefix_and_exposes_new_leaves(self):
        model = DummyModel()
        base_vertices = model.vertices.detach().clone()
        result = install_midpoint_probe(model, torch.tensor([0]))

        self.assertTrue(result["prefix_unchanged"])
        self.assertTrue(result["topology_valid"])
        torch.testing.assert_close(model.vertices[:3], base_vertices)
        self.assertEqual(model.vertices.shape[0], 6)
        self.assertEqual(model._triangle_indices.shape[0], 4)

        loss = sum(value.sum() for value in (
            model.vertices,
            model.vertex_weight,
            model._features_dc,
            model._features_rest,
        ))
        gradients = torch.autograd.grad(
            loss, tuple(result["parameters"].values()), allow_unused=False
        )
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(float(gradient.abs().sum()) > 0.0 for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
