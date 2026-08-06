import math
import unittest

import torch

from adc_f0_decide import (
    COVARIATES,
    HOMOGENEITY_RHO,
    HOMOGENEITY_SPREAD,
    _average_ranks,
    read,
    spearman,
    spread,
    stratum_medians,
)
from adc_probe import (
    RUNGS,
    central_differences,
    face_max_per_vertex,
    rung_disagreement,
    stratified_probe_set,
)


class CentralDifferenceTest(unittest.TestCase):
    """The probe loss is exactly quadratic in the probed scalar, so the central
    difference is exact at any step size. Checking that against a closed form is the
    only way to tell a broken probe from a broken rasterizer -- the confusion that
    cost this project three failed attempts at the G0-lite gate."""

    def _probe(self, loss, start=0.3):
        box = {"x": start}
        estimates = central_differences(
            lambda value: box.__setitem__("x", value), lambda: loss(box["x"]), start
        )
        return estimates, box

    def test_a_quadratic_is_differentiated_exactly(self):
        # L(x) = 3x^2 - 2x + 7  =>  L'(0.3) = 6(0.3) - 2 = -0.2
        estimates, _ = self._probe(lambda x: 3.0 * x**2 - 2.0 * x + 7.0)
        for estimate in estimates:
            self.assertAlmostEqual(estimate, -0.2, places=9)
        self.assertLess(rung_disagreement(estimates), 1e-9)

    def test_the_probed_value_is_restored(self):
        _, box = self._probe(lambda x: x**2)
        self.assertEqual(box["x"], 0.3)

    def test_the_value_is_restored_even_when_the_loss_raises(self):
        box = {"x": 0.3}

        def exploding():
            raise RuntimeError("render failed")

        with self.assertRaises(RuntimeError):
            central_differences(
                lambda value: box.__setitem__("x", value), exploding, 0.3
            )
        self.assertEqual(box["x"], 0.3)

    def test_a_kinked_loss_makes_the_rungs_disagree(self):
        """`L1 + SSIM` has kinks where a pixel residual changes sign; this is why the
        probe loss is squared error instead. A kink inside the coarse step but
        outside the fine one is exactly what the rung check is there to catch."""
        kink = 0.3 + 0.5 * (RUNGS[0] + RUNGS[-1])
        estimates, _ = self._probe(lambda x: abs(x - kink))
        self.assertGreater(rung_disagreement(estimates), 1e-3)


class FaceToVertexReductionTest(unittest.TestCase):
    def test_a_vertex_takes_the_maximum_over_its_incident_faces(self):
        faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)
        values = torch.tensor([2.0, 5.0])
        reduced = face_max_per_vertex(values, faces, 4)
        self.assertEqual(reduced.tolist(), [2.0, 5.0, 5.0, 5.0])

    def test_vertices_touched_by_no_face_stay_at_zero(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        reduced = face_max_per_vertex(torch.tensor([4.0]), faces, 5)
        self.assertEqual(reduced[3:].tolist(), [0.0, 0.0])


class StratifiedProbeSetTest(unittest.TestCase):
    def setUp(self):
        self.values = torch.arange(100, dtype=torch.float32)
        self.eligible = torch.arange(100)

    def test_every_stratum_is_represented(self):
        picked = stratified_probe_set(self.values, self.eligible, strata=5, per_stratum=4)
        self.assertEqual(picked.numel(), 20)
        # 100 sorted values in 5 bins of 20; each pick must land in its own bin.
        self.assertEqual(sorted({int(v) // 20 for v in picked}), [0, 1, 2, 3, 4])

    def test_the_extremes_of_each_stratum_are_included(self):
        picked = stratified_probe_set(self.values, self.eligible, strata=5, per_stratum=4)
        self.assertIn(0, picked.tolist())
        self.assertIn(99, picked.tolist())

    def test_it_is_deterministic(self):
        first = stratified_probe_set(self.values, self.eligible, strata=5, per_stratum=4)
        second = stratified_probe_set(self.values, self.eligible, strata=5, per_stratum=4)
        self.assertTrue(torch.equal(first, second))

    def test_a_stratum_smaller_than_the_quota_is_not_padded(self):
        picked = stratified_probe_set(
            torch.arange(7, dtype=torch.float32), torch.arange(7), strata=5, per_stratum=4
        )
        self.assertEqual(picked.numel(), 7)
        self.assertEqual(sorted(picked.tolist()), list(range(7)))

    def test_only_eligible_members_are_probed(self):
        eligible = torch.arange(50, 100)
        picked = stratified_probe_set(self.values, eligible, strata=5, per_stratum=4)
        self.assertGreaterEqual(int(picked.min()), 50)


class SpearmanTest(unittest.TestCase):
    def test_a_monotone_relation_saturates(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=9)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=9)

    def test_it_is_rank_based_not_value_based(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 2, 3, 400]), 1.0, places=9)

    def test_a_constant_covariate_gives_zero_rather_than_nan(self):
        """`max_blending` saturates at 1.0 for many faces; a NaN here would
        propagate into the reading and into a results file written with
        allow_nan=False."""
        rho = spearman([1.0, 1.0, 1.0, 1.0], [3.0, 1.0, 4.0, 2.0])
        self.assertEqual(rho, 0.0)
        self.assertFalse(math.isnan(rho))

    def test_ties_receive_averaged_ranks(self):
        self.assertEqual(_average_ranks(torch.tensor([5.0, 5.0, 9.0])).tolist(), [1.5, 1.5, 3.0])


class StratumMediansTest(unittest.TestCase):
    def test_bins_are_ordered_by_the_covariate_not_by_input_order(self):
        values = [4.0, 1.0, 3.0, 2.0]
        ratios = [40.0, 10.0, 30.0, 20.0]
        self.assertEqual(stratum_medians(values, ratios, strata=4), [10.0, 20.0, 30.0, 40.0])


class SpreadTest(unittest.TestCase):
    def test_it_is_the_ratio_of_the_extreme_medians(self):
        self.assertAlmostEqual(spread([2.0, 3.0, 4.0]), 2.0)

    def test_a_non_positive_median_makes_the_spread_undefined(self):
        """A sign disagreement between the analytic gradient and the finite
        difference makes max/min meaningless; None keeps the record serialisable
        under allow_nan=False, which infinity would not."""
        self.assertIsNone(spread([1.0, 0.0, 2.0]))
        self.assertIsNone(spread([-1.0, 2.0]))
        self.assertIsNone(spread([]))


def _view(name="v0", medians=None, rhos=None, survival=1.0, deterministic=True):
    flat = [8.4, 8.4, 8.4, 8.4, 8.4]
    return {
        "view": name,
        "deterministic_forward": deterministic,
        "survival_fraction": survival,
        "stratum_medians": {c: (medians or {}).get(c, flat) for c in COVARIATES},
        "spearman": {c: (rhos or {}).get(c, 0.0) for c in COVARIATES},
    }


class ReadTest(unittest.TestCase):
    def test_flat_ratios_on_both_views_read_homogeneous(self):
        self.assertEqual(read([_view("a"), _view("b")])["reading"], "HOMOGENEOUS")

    def test_no_views_is_inconclusive(self):
        self.assertEqual(read([])["reading"], "INCONCLUSIVE")

    def test_low_survival_is_inconclusive(self):
        result = read([_view("a"), _view("b", survival=0.5)])
        self.assertEqual(result["reading"], "INCONCLUSIVE")

    def test_a_non_deterministic_forward_is_inconclusive(self):
        """Every central difference would be noise; no reading may be taken."""
        result = read([_view("a"), _view("b", deterministic=False)])
        self.assertEqual(result["reading"], "INCONCLUSIVE")

    def test_a_spread_beyond_the_bound_reads_heterogeneous(self):
        wide = [4.0, 5.0, 6.0, 7.0, 4.0 * HOMOGENEITY_SPREAD * 1.1]
        result = read([_view("a", {"depth": wide}), _view("b", {"depth": wide})])
        self.assertEqual(result["reading"], "HETEROGENEOUS")
        self.assertEqual(result["strongest_trend"]["covariate"], "depth")

    def test_a_gentle_monotone_trend_is_caught_by_rho_alone(self):
        """The quintile extremes can stay within the spread bound while the ratio
        still climbs monotonically with the covariate; rho is what sees that."""
        rho = HOMOGENEITY_RHO + 0.2
        result = read([_view("a", rhos={"projected_size": rho}),
                       _view("b", rhos={"projected_size": rho})])
        self.assertEqual(result["reading"], "HETEROGENEOUS")
        self.assertEqual(result["strongest_trend"]["covariate"], "projected_size")
        self.assertIsNotNone(result["strongest_trend"]["spread"])

    def test_the_strongest_trend_is_the_one_reported(self):
        result = read(
            [
                _view("a", rhos={"depth": 0.4, "max_blending": 0.9}),
                _view("b", rhos={"depth": 0.4, "max_blending": 0.9}),
            ]
        )
        self.assertEqual(result["strongest_trend"]["covariate"], "max_blending")

    def test_views_that_disagree_are_reported_not_pooled(self):
        result = read([_view("a"), _view("b", rhos={"depth": 0.9})])
        self.assertEqual(result["reading"], "VIEW_DEPENDENT")
        self.assertEqual(
            [item["view"] for item in result["per_view"] if item["offenders"]], ["b"]
        )

    def test_an_undefined_spread_reads_heterogeneous_rather_than_crashing(self):
        sign_flip = [8.4, 8.4, -1.0, 8.4, 8.4]
        result = read([_view("a", {"depth": sign_flip}), _view("b", {"depth": sign_flip})])
        self.assertEqual(result["reading"], "HETEROGENEOUS")
        self.assertIsNone(result["strongest_trend"]["spread"])


if __name__ == "__main__":
    unittest.main()
