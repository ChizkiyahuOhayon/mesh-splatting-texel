"""Tests for the per-face hardening rate's defining properties."""

import pytest
import torch

from sota.hardness import face_sigma, rate, raw_from_rate


FINAL = 1e-4
SCHEDULED = [1.0, 0.633, 0.167, 0.01, FINAL]
RAWS = torch.linspace(-4.0, 4.0, 65)


def test_unit_rate_is_the_published_schedule():
    raw = torch.full((5,), raw_from_rate(1.0))
    for scheduled in SCHEDULED:
        assert face_sigma(raw, scheduled, FINAL) == pytest.approx(scheduled, rel=1e-6)


def test_every_face_lands_on_the_endpoint_whatever_it_learned():
    """At the last iteration the schedule is the endpoint, so every face is too."""
    assert torch.allclose(face_sigma(RAWS, FINAL, FINAL),
                          torch.full_like(RAWS, FINAL), rtol=1e-9)


@pytest.mark.parametrize("scheduled", SCHEDULED)
def test_rate_below_one_hardens_early_and_above_one_lags(scheduled):
    early = face_sigma(torch.tensor(raw_from_rate(0.5)), scheduled, FINAL)
    onclock = face_sigma(torch.tensor(raw_from_rate(1.0)), scheduled, FINAL)
    late = face_sigma(torch.tensor(raw_from_rate(2.0)), scheduled, FINAL)
    assert early <= onclock <= late
    if scheduled > FINAL:
        assert early < onclock < late


@pytest.mark.parametrize("scheduled", SCHEDULED)
def test_sigma_stays_positive(scheduled):
    assert torch.all(face_sigma(RAWS, scheduled, FINAL) > 0)


def test_hardening_is_monotone_in_the_iteration_for_any_rate():
    schedule = torch.linspace(1.0, FINAL, 200)
    for raw in (raw_from_rate(0.25), raw_from_rate(1.0), raw_from_rate(4.0)):
        values = [face_sigma(torch.tensor(raw), float(s), FINAL) for s in schedule]
        assert all(b <= a for a, b in zip(values, values[1:]))


def test_gradient_vanishes_only_at_the_endpoint():
    for scheduled in SCHEDULED[:-1]:
        raw = RAWS.clone().requires_grad_(True)
        face_sigma(raw, scheduled, FINAL).sum().backward()
        assert torch.all(raw.grad > 0)
    raw = RAWS.clone().requires_grad_(True)
    face_sigma(raw, FINAL, FINAL).sum().backward()
    assert torch.allclose(raw.grad, torch.zeros_like(raw.grad), atol=1e-12)


def test_analytic_derivative_matches_autograd():
    """d sigma / d rate is the schedule's remaining distance to the endpoint."""
    raw = torch.tensor([-2.0, 0.0, 1.5], requires_grad=True)
    face_sigma(raw, 0.4, FINAL).sum().backward()
    assert torch.allclose(raw.grad, (0.4 - FINAL) * rate(raw.detach()), rtol=1e-6)


def test_raw_round_trips_through_rate():
    for value in (0.1, 0.5, 1.0, 3.0):
        assert rate(torch.tensor(raw_from_rate(value))).item() == pytest.approx(value)
