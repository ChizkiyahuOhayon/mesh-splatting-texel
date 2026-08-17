"""Tests for the budgeted per-face hardening rate."""

import pytest
import torch

from sota.hardness import DEFAULT_SPREAD, face_sigma, rate


FINAL = 1e-4
SCHEDULED = [1.0, 0.633, 0.167, 0.01, FINAL]
RAWS = torch.linspace(-6.0, 6.0, 257)


def test_zero_raw_is_the_published_schedule():
    raw = torch.zeros(5)
    assert torch.allclose(rate(raw), torch.ones(5))
    for scheduled in SCHEDULED:
        assert face_sigma(raw, scheduled, FINAL) == pytest.approx(scheduled, rel=1e-6)


def test_rates_spend_a_fixed_budget():
    """Whatever the faces learn, the mean rate is one -- softness is only traded."""
    for raws in (RAWS, torch.randn(4096) * 3.0, torch.full((512,), 5.0)):
        assert rate(raws).mean().item() == pytest.approx(1.0, rel=1e-6)


def test_rates_are_bounded_relative_to_each_other():
    spread = DEFAULT_SPREAD
    values = rate(RAWS)
    assert (values.max() / values.min()).item() <= spread ** 2 + 1e-4


def test_every_face_lands_on_the_endpoint_whatever_it_learned():
    assert torch.allclose(face_sigma(RAWS, FINAL, FINAL),
                          torch.full_like(RAWS, FINAL), rtol=1e-9)


def test_a_face_can_only_buy_softness_by_selling_it():
    """Pushing one face softer must push the rest harder, at equal total."""
    raw = torch.zeros(64)
    raw[0] = 4.0
    values = rate(raw)
    assert values[0] > 1.0
    assert torch.all(values[1:] < 1.0)
    assert values.mean().item() == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("scheduled", SCHEDULED)
def test_sigma_stays_positive(scheduled):
    assert torch.all(face_sigma(RAWS, scheduled, FINAL) > 0)


def test_hardening_is_monotone_in_the_iteration():
    raw = torch.tensor([-3.0, 0.0, 3.0])
    for schedule_value in zip(torch.linspace(1.0, FINAL, 200)[:-1],
                              torch.linspace(1.0, FINAL, 200)[1:]):
        earlier, later = (face_sigma(raw, float(v), FINAL) for v in schedule_value)
        assert torch.all(later <= earlier)


def test_gradients_survive_the_bound():
    """tanh keeps even a saturated face differentiable, unlike a clamp."""
    raw = torch.tensor([-8.0, -1.0, 0.0, 1.0, 8.0], requires_grad=True)
    face_sigma(raw, 0.4, FINAL).sum().backward()
    assert torch.all(raw.grad.abs() > 0)


def test_the_budget_is_enforced_inside_the_graph():
    """Nothing can move the total, and the gradient says so exactly.

    The rates sum to the face count by construction, so a loss that depends only
    on that total has an identically zero gradient -- the constraint is part of
    the optimisation rather than a projection applied after it.
    """
    raw = torch.randn(256, requires_grad=True)
    rate(raw).sum().backward()
    assert torch.allclose(raw.grad, torch.zeros_like(raw.grad), atol=1e-6)


def test_a_loss_that_prefers_one_face_still_moves_it():
    """Redistribution is possible even though the total is pinned."""
    raw = torch.zeros(64, requires_grad=True)
    weights = torch.zeros(64)
    weights[0] = 1.0
    (rate(raw) * weights).sum().backward()
    assert raw.grad[0] > 0
    assert torch.all(raw.grad[1:] < 0)


def test_spread_one_pins_every_face_to_the_schedule():
    assert torch.allclose(rate(RAWS, spread=1.0), torch.ones_like(RAWS))
