"""Tests for the coverage identity and the hardening schedules."""

import numpy as np
import pytest

from sota.sigma_schedule import SCHEDULES, coverage, schedule, sigma_at


ENDS = dict(initial_sigma=1.0, final_sigma=1e-4, start=0, until=30000)


def monte_carlo_coverage(sigma, samples=400_000, seed=0):
    """Average `phi ** sigma` over a triangle by sampling it directly.

    `phi` is rebuilt from its definition -- distance to the nearest edge over the
    largest such distance -- rather than from the closed form under test.
    """
    rng = np.random.default_rng(seed)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.9]])
    u, v = rng.random(samples), rng.random(samples)
    flip = u + v > 1.0
    u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
    points = corners[0] + u[:, None] * (corners[1] - corners[0]) \
                        + v[:, None] * (corners[2] - corners[0])

    edges = [(corners[i], corners[(i + 1) % 3]) for i in range(3)]
    distances = []
    for tail, head in edges:
        along = head - tail
        normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
        distances.append(np.abs((points - tail) @ normal))
    nearest = np.min(np.stack(distances), axis=0)
    # The inradius is the largest distance-to-nearest-edge inside the triangle.
    sides = [np.linalg.norm(head - tail) for tail, head in edges]
    span = corners[1:] - corners[0]
    area = 0.5 * abs(span[0, 0] * span[1, 1] - span[0, 1] * span[1, 0])
    inradius = 2.0 * area / sum(sides)
    return float(np.mean((nearest / inradius) ** sigma))


@pytest.mark.parametrize("sigma", [0.05, 0.25, 1.0, 2.0])
def test_closed_form_matches_direct_sampling(sigma):
    assert coverage(sigma) == pytest.approx(monte_carlo_coverage(sigma), abs=3e-3)


def test_coverage_endpoints():
    assert coverage(1.0) == pytest.approx(1.0 / 3.0)
    assert coverage(0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("target", [1 / 3, 0.5, 0.79, 0.99, 1.0])
def test_sigma_at_inverts_coverage(target):
    assert coverage(sigma_at(target)) == pytest.approx(target, abs=1e-12)


@pytest.mark.parametrize("name", SCHEDULES)
def test_every_schedule_shares_both_endpoints(name):
    assert schedule(name, 0, **ENDS) == pytest.approx(1.0)
    assert schedule(name, 30000, **ENDS) == 1e-4
    assert schedule(name, 999999, **ENDS) == 1e-4


@pytest.mark.parametrize("name", SCHEDULES)
def test_every_schedule_hardens_monotonically(name):
    values = [schedule(name, it, **ENDS) for it in range(0, 30001, 250)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_linear_is_the_published_path():
    for iteration in (1, 7000, 24000, 29999):
        published = 1.0 - (1.0 - 1e-4) * (iteration / 30000)
        assert schedule("linear", iteration, **ENDS) == pytest.approx(published)


def test_schedules_move_coverage_out_of_the_dead_learning_rate_tail():
    """Coverage still unspent at 25k, where the vertex lr is ~2% of initial."""
    left = {name: 1.0 - coverage(schedule(name, 25000, **ENDS)) for name in SCHEDULES}
    assert left["linear"] > left["coverage"] > left["lrmatched"]
    assert left["linear"] == pytest.approx(0.209, abs=0.005)
    assert left["lrmatched"] < 0.01


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown sigma schedule"):
        schedule("cosine", 100, **ENDS)
