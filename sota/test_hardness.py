"""Tests for the per-face hardness carrier's three defining properties."""

import pytest
import torch

from sota.hardness import (
    INITIAL_ADDED_HARDNESS,
    added_hardness,
    face_sigma,
    raw_from_hardness,
)


SCHEDULED = [1.0, 0.633, 0.167, 1e-4]


@pytest.mark.parametrize("scheduled", SCHEDULED)
def test_zero_addition_reproduces_the_schedule(scheduled):
    raw = torch.full((5,), raw_from_hardness(1e-30))
    assert face_sigma(raw, scheduled) == pytest.approx(scheduled, rel=1e-6)


@pytest.mark.parametrize("scheduled", SCHEDULED)
def test_no_face_is_ever_softer_than_the_schedule(scheduled):
    raw = torch.linspace(-12.0, 6.0, 64)
    assert torch.all(face_sigma(raw, scheduled) <= scheduled)


def test_the_endpoint_is_reached_whatever_the_face_learned():
    """Any finite parameter still ends at or below the published final sigma."""
    raw = torch.linspace(-20.0, 20.0, 256)
    assert torch.all(face_sigma(raw, 1e-4) <= 1e-4)


def test_the_gradient_never_vanishes():
    raw = torch.linspace(-12.0, 6.0, 64).requires_grad_(True)
    face_sigma(raw, 0.167).sum().backward()
    assert torch.all(raw.grad != 0)
    # Adding hardness lowers sigma, so the derivative is negative throughout.
    assert torch.all(raw.grad < 0)


def test_analytic_derivative_matches_autograd():
    """d sigma / d a = -sigma ** 2, routed through the exponential storage."""
    raw = torch.tensor([-6.0, -1.0, 0.5, 3.0], requires_grad=True)
    sigma = face_sigma(raw, 0.4)
    sigma.sum().backward()
    expected = -(sigma.detach() ** 2) * added_hardness(raw.detach())
    assert torch.allclose(raw.grad, expected, rtol=1e-6)


def test_initial_value_is_visually_neutral_but_movable():
    raw = torch.full((1,), raw_from_hardness(INITIAL_ADDED_HARDNESS))
    assert added_hardness(raw).item() == pytest.approx(INITIAL_ADDED_HARDNESS, rel=1e-6)
    # At the hardest scheduled point of interest the deviation is well under a
    # tenth of a percent, so allocation does not perturb a converged render.
    for scheduled in SCHEDULED:
        assert face_sigma(raw, scheduled).item() == pytest.approx(scheduled, rel=2e-3)


def test_hardness_is_additive_in_the_reciprocal():
    raw = torch.tensor([-2.0, 0.0, 1.5])
    for scheduled in SCHEDULED:
        combined = 1.0 / face_sigma(raw, scheduled)
        assert torch.allclose(combined, added_hardness(raw) + 1.0 / scheduled, rtol=1e-6)
