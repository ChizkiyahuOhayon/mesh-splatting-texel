import unittest

import torch

from arguments import OptimizationParams
from sota.endpoint import endpoint_image, opacity_at_floor


class _ArgumentSink:
    def add_argument_group(self, _name):
        return self

    def add_argument(self, *_args, **_kwargs):
        return None


class EndpointImageTest(unittest.TestCase):
    def test_forward_is_hard_and_backward_is_soft(self):
        soft = torch.tensor([1.0, 2.0], requires_grad=True)
        hard = torch.tensor([4.0, 8.0])

        image = endpoint_image(soft, hard)
        torch.testing.assert_close(image, hard, rtol=0.0, atol=0.0)

        (image.square().sum()).backward()
        torch.testing.assert_close(soft.grad, 2.0 * hard, rtol=0.0, atol=0.0)

    def test_render_shapes_must_match(self):
        with self.assertRaisesRegex(ValueError, "render shapes differ"):
            endpoint_image(torch.zeros(2), torch.zeros(3))


class EndpointOpacityTest(unittest.TestCase):
    def test_floor_matches_the_model_parameterization(self):
        raw = torch.tensor([-1.0, 0.0, 1.0])
        expected = 0.9999 + 0.0001 * torch.sigmoid(raw)
        torch.testing.assert_close(opacity_at_floor(raw, 0.9999), expected)

    def test_invalid_floor_is_refused(self):
        with self.assertRaisesRegex(ValueError, "opacity floor"):
            opacity_at_floor(torch.zeros(1), 1.0)


class EndpointArgumentTest(unittest.TestCase):
    def test_endpoint_supervision_is_opt_in(self):
        self.assertFalse(OptimizationParams(_ArgumentSink()).endpoint_supervision)


if __name__ == "__main__":
    unittest.main()
