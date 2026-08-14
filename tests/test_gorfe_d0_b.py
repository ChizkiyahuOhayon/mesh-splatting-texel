import unittest

import torch

from gorfe_d0_b import (
    PROTOCOL_CONSTANTS,
    JointTerms,
    camera_joint_terms,
    overall_reading,
    portfolio_record,
    scene_family_reading,
)


class JointUtilityTest(unittest.TestCase):
    def _terms(self, rows, *, residual=10.0, groups=2, dc=None, sh1=None):
        pixels = torch.tensor([row[0] for row in rows], dtype=torch.int64)
        group_ids = torch.tensor([row[1] for row in rows], dtype=torch.int64)
        features = torch.tensor([row[2] for row in rows], dtype=torch.float64)
        residuals = torch.zeros((2, 3), dtype=torch.float64)
        residuals[:, 0] = residual
        if dc is None:
            dc = torch.zeros((groups, 1, 3), dtype=torch.float64)
        if sh1 is None:
            sh1 = torch.zeros((groups, 3, 3), dtype=torch.float64)
        return camera_joint_terms(
            pixel_count=2,
            pixel_ids=pixels,
            group_ids=group_ids,
            features=features,
            residuals=residuals,
            dc_coefficients=dc,
            sh1_coefficients=sh1,
        )

    def test_one_group_and_disjoint_groups_have_no_interaction(self):
        coefficients = torch.zeros((2, 1, 3), dtype=torch.float64)
        coefficients[:, 0, 0] = torch.tensor([1.0, 2.0])
        one = self._terms([(0, 0, [1, 0, 0, 0])], dc=coefficients)
        disjoint = self._terms(
            [(0, 0, [1, 0, 0, 0]), (1, 1, [1, 0, 0, 0])], dc=coefficients
        )
        self.assertEqual(one.interaction_penalty, 0.0)
        self.assertEqual(disjoint.interaction_penalty, 0.0)
        self.assertEqual(one.additive_gain, one.joint_gain)

    def test_blank_design_has_exactly_zero_terms(self):
        terms = camera_joint_terms(
            pixel_count=2,
            pixel_ids=torch.empty(0, dtype=torch.int64),
            group_ids=torch.empty(0, dtype=torch.int64),
            features=torch.empty((0, 4), dtype=torch.float64),
            residuals=torch.ones((2, 3), dtype=torch.float64),
            dc_coefficients=torch.zeros((2, 1, 3), dtype=torch.float64),
            sh1_coefficients=torch.zeros((2, 3, 3), dtype=torch.float64),
        )
        self.assertEqual(terms, JointTerms(0.0, 0.0, 0.0, 0, 0))

    def test_same_pixel_overlap_can_be_destructive_or_synergistic(self):
        coefficients = torch.zeros((2, 1, 3), dtype=torch.float64)
        coefficients[:, 0, 0] = torch.tensor([1.0, 2.0])
        destructive = self._terms(
            [(0, 0, [1, 0, 0, 0]), (0, 1, [1, 0, 0, 0])], dc=coefficients
        )
        coefficients[1, 0, 0] = -2.0
        synergistic = self._terms(
            [(0, 0, [1, 0, 0, 0]), (0, 1, [1, 0, 0, 0])], dc=coefficients
        )
        self.assertEqual(destructive.linear, 60.0)
        self.assertEqual(destructive.diagonal_quadratic, 5.0)
        self.assertEqual(destructive.joint_quadratic, 9.0)
        self.assertEqual(destructive.interaction_penalty, 4.0)
        self.assertEqual(synergistic.interaction_penalty, -4.0)

    def test_duplicate_rows_reduce_before_the_diagonal_quadratic(self):
        coefficients = torch.zeros((2, 1, 3), dtype=torch.float64)
        coefficients[0, 0, 0] = 1.0
        terms = self._terms(
            [(0, 0, [0.4, 0, 0, 0]), (0, 0, [0.6, 0, 0, 0])],
            dc=coefficients,
        )
        self.assertEqual(terms.input_rows, 2)
        self.assertEqual(terms.reduced_rows, 1)
        self.assertAlmostEqual(terms.diagonal_quadratic, 1.0)

    def test_dc_and_sh1_on_one_edge_remain_distinct_groups(self):
        dc = torch.zeros((2, 1, 3), dtype=torch.float64)
        sh1 = torch.zeros((2, 3, 3), dtype=torch.float64)
        dc[0, 0, 0] = 1.0
        sh1[0, 0, 0] = 2.0
        terms = self._terms([(0, 0, [1, 1, 0, 0])], dc=dc, sh1=sh1)
        self.assertEqual(terms.diagonal_quadratic, 5.0)
        self.assertEqual(terms.joint_quadratic, 9.0)
        self.assertEqual(terms.interaction_penalty, 4.0)

    def test_row_order_and_camera_addition_are_invariant(self):
        coefficients = torch.zeros((2, 1, 3), dtype=torch.float64)
        coefficients[:, 0, 0] = torch.tensor([1.0, 2.0])
        rows = [(0, 0, [1, 0, 0, 0]), (0, 1, [1, 0, 0, 0])]
        forward = self._terms(rows, dc=coefficients)
        reverse = self._terms(list(reversed(rows)), dc=coefficients)
        self.assertEqual(forward, reverse)
        doubled = forward + reverse
        self.assertIsInstance(doubled, JointTerms)
        self.assertEqual(doubled.interaction_penalty, 8.0)

    def test_matches_an_independent_scalar_oracle(self):
        rows = [
            (0, 0, [0.5, 1.0, 0.0, 0.0]),
            (0, 0, [0.25, -0.5, 0.0, 0.0]),
            (0, 1, [1.0, 0.0, 0.5, 0.0]),
            (1, 0, [-0.5, 0.0, 0.0, 1.0]),
        ]
        dc = torch.tensor(
            [[[1.0, 0.5, -0.25]], [[-0.5, 0.25, 1.0]]], dtype=torch.float64
        )
        sh1 = torch.tensor(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, -0.5]],
                [[-0.25, 0.5, 0.0], [0.0, 0.0, 0.75], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        residuals = torch.tensor(
            [[2.0, -1.0, 0.5], [0.25, 1.5, -0.75]], dtype=torch.float64
        )
        observed = camera_joint_terms(
            pixel_count=2,
            pixel_ids=torch.tensor([row[0] for row in rows], dtype=torch.int64),
            group_ids=torch.tensor([row[1] for row in rows], dtype=torch.int64),
            features=torch.tensor([row[2] for row in rows], dtype=torch.float64),
            residuals=residuals,
            dc_coefficients=dc,
            sh1_coefficients=sh1,
        )

        reduced = {}
        for pixel, group, feature in rows:
            key = (pixel, group)
            reduced.setdefault(key, [0.0] * 4)
            reduced[key] = [left + right for left, right in zip(reduced[key], feature)]
        correction = [[0.0] * 3 for _ in range(2)]
        diagonal = 0.0
        for (pixel, group), feature in reduced.items():
            z_dc = [feature[0] * float(dc[group, 0, channel]) for channel in range(3)]
            z_sh1 = [
                sum(feature[row + 1] * float(sh1[group, row, channel]) for row in range(3))
                for channel in range(3)
            ]
            diagonal += sum(value * value for value in z_dc + z_sh1)
            correction[pixel] = [
                old + left + right
                for old, left, right in zip(correction[pixel], z_dc, z_sh1)
            ]
        linear = 2.0 * sum(
            correction[pixel][channel] * float(residuals[pixel, channel])
            for pixel in range(2)
            for channel in range(3)
        )
        joint = sum(value * value for row in correction for value in row)
        self.assertAlmostEqual(observed.linear, linear)
        self.assertAlmostEqual(observed.diagonal_quadratic, diagonal)
        self.assertAlmostEqual(observed.joint_quadratic, joint)

    def test_invalid_shapes_and_out_of_range_pixels_are_refused(self):
        coefficients = torch.zeros((2, 1, 3), dtype=torch.float64)
        with self.assertRaises(ValueError):
            self._terms([(2, 0, [1, 0, 0, 0])], dc=coefficients)
        with self.assertRaises(TypeError):
            self._terms(
                [(0, 0, [1, 0, 0, 0])], dc=coefficients.to(torch.float32)
            )

    def test_portfolio_record_checks_additive_identity_and_normalizes(self):
        terms = JointTerms(10.0, 2.0, 3.0, 7, 5)
        record = portfolio_record(
            terms,
            expected_additive_gain=8.0,
            budget=4096,
            spent=4096,
            outer_sse=100.0,
        )
        self.assertEqual(record["additive_gain"], 8.0)
        self.assertEqual(record["joint_gain"], 7.0)
        self.assertEqual(record["interaction_penalty"], 1.0)
        self.assertAlmostEqual(record["p_int"], 0.01)
        with self.assertRaisesRegex(ValueError, "sealed state"):
            portfolio_record(
                terms,
                expected_additive_gain=8.1,
                budget=4096,
                spent=4096,
                outer_sse=100.0,
            )


class MechanismReadingTest(unittest.TestCase):
    def _passing(self):
        # selector order: primary, random, rhs, same-view; budget order: small, large
        value = torch.zeros((4, 2, 4), dtype=torch.float64)
        value[0, 0] = 1.0
        value[0, 1] = 4.0
        value[1:, 0] = 0.5
        value[1:, 1] = 1.0
        return value

    def test_every_frozen_condition_is_required(self):
        passing = scene_family_reading(self._passing())
        self.assertTrue(passing["supported"])
        for selector in range(1, 4):
            failing = self._passing()
            failing[selector, 1] = 5.0
            self.assertFalse(scene_family_reading(failing)["supported"])
        failing = self._passing()
        failing[0, 1, :2] = -1.0
        self.assertFalse(scene_family_reading(failing)["supported"])

    def test_excess_growth_is_separate_from_large_budget_advantage(self):
        failing = self._passing()
        failing[0, 0] = 4.0
        reading = scene_family_reading(failing)
        self.assertTrue(reading["checks"]["large_primary_beats_random_id_cv4"])
        self.assertFalse(
            reading["checks"]["primary_excess_growth_beats_random_id_cv4"]
        )
        self.assertFalse(reading["supported"])

    def test_overall_requires_one_common_family(self):
        yes = {"supported": True}
        no = {"supported": False}
        localized = {
            "garden": {"SH1": yes, "MIXED": no},
            "room": {"SH1": no, "MIXED": yes},
        }
        self.assertEqual(overall_reading(localized)["decision"], "rejected")
        localized["room"]["SH1"] = yes
        result = overall_reading(localized)
        self.assertEqual(result["decision"], "supported")
        self.assertEqual(result["supported_common_families"], ["SH1"])

    def test_constants_are_the_frozen_protocol_values(self):
        self.assertEqual(PROTOCOL_CONSTANTS["budgets_cost_units"], [4096, 16384])
        self.assertEqual(
            PROTOCOL_CONSTANTS["selectors"],
            ["primary", "random_id", "rhs_norm", "same_view_gain"],
        )


if __name__ == "__main__":
    unittest.main()
