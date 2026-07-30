import unittest

import torch

from frt_freeze import assert_base_unchanged, freeze_base_tensors


class DummyModel:
    def __init__(self):
        self.vertices = torch.ones(2, 3, requires_grad=True)
        self._triangle_indices = torch.tensor([[0, 1, 1]], dtype=torch.int32)
        self.vertex_weight = torch.ones(2, 1, requires_grad=True)
        self._features_dc = torch.ones(2, 1, 3, requires_grad=True)
        self._features_rest = torch.ones(2, 3, 3, requires_grad=True)
        self._sigma = torch.tensor(0.1)


class FreezeBaseTest(unittest.TestCase):
    def test_freezes_and_accepts_unchanged_base(self):
        model = DummyModel()
        fingerprint = freeze_base_tensors(model)
        self.assertFalse(model.vertices.requires_grad)
        self.assertFalse(model._features_dc.requires_grad)
        assert_base_unchanged(model, fingerprint)

    def test_detects_in_place_change(self):
        model = DummyModel()
        fingerprint = freeze_base_tensors(model)
        with torch.no_grad():
            model.vertices.add_(1.0)
        with self.assertRaises(RuntimeError):
            assert_base_unchanged(model, fingerprint)


if __name__ == "__main__":
    unittest.main()
