import json
import unittest

import torch

from gorfe_v1_evaluate_core import (
    EligibilityMasks,
    FreezeMismatchError,
    SH1_RANK_RELATIVE_TRACE_THRESHOLD,
    SceneInvalidError,
    derive_eligibility,
    evaluate_scene,
    validate_evaluation_identity,
)
from gorfe_v1_stream import CarrierStatistics, StreamStatistics


def _carrier(groups, feature_dim):
    gram = torch.eye(feature_dim, dtype=torch.float64).reshape(
        1, 1, feature_dim, feature_dim
    ).repeat(groups, 4, 1, 1)
    return CarrierStatistics(
        gram=gram,
        rhs=torch.zeros((groups, 4, feature_dim, 3), dtype=torch.float64),
        support_rss=torch.zeros((groups, 4), dtype=torch.float64),
        support_pixels=torch.full((groups, 4), 32, dtype=torch.int64),
        support_cameras=torch.full((groups, 4), 4, dtype=torch.int64),
    )


def _replace_carrier(carrier, **updates):
    values = {
        "gram": carrier.gram.clone(),
        "rhs": carrier.rhs.clone(),
        "support_rss": carrier.support_rss.clone(),
        "support_pixels": carrier.support_pixels.clone(),
        "support_cameras": carrier.support_cameras.clone(),
    }
    values.update(updates)
    return CarrierStatistics(**values)


def _replace_stream(statistics, **updates):
    values = {
        "dc": _replace_carrier(statistics.dc),
        "sh1": _replace_carrier(statistics.sh1),
        "fold_full_rss": statistics.fold_full_rss.clone(),
    }
    values.update(updates)
    return StreamStatistics(**values)


def _eligibility_fixture():
    groups = 5
    dc = _carrier(groups, 1)
    sh1 = _carrier(groups, 3)
    dc.support_pixels[1, 0] = 31
    dc.support_cameras[2, 1] = 3
    dc.gram[3, 2] = 0
    dc.gram[4, 0, 0, 0] = float("nan")

    sh1.gram[1, 0, 0, 0] = 1e-12
    sh1.support_pixels[2, 0] = 31
    sh1.gram[4, 0, 0, 0] = float("nan")
    return StreamStatistics(
        dc=dc,
        sh1=sh1,
        fold_full_rss=torch.zeros(4, dtype=torch.float64),
    )


def _evaluation_from(target_free):
    dc = _replace_carrier(target_free.dc)
    sh1 = _replace_carrier(target_free.sh1)
    for carrier in (dc, sh1):
        carrier.support_rss.fill_(100.0)
        for group in range(carrier.gram.shape[0]):
            for fold in range(4):
                carrier.rhs[group, fold, 0, 0] = 0.02 * (group + 1) * (fold + 1)
    return StreamStatistics(
        dc=dc,
        sh1=sh1,
        fold_full_rss=torch.full((4,), 1000.0, dtype=torch.float64),
    )


class EligibilityTest(unittest.TestCase):
    def test_every_locked_support_and_finiteness_condition_is_conjunctive(self):
        masks = derive_eligibility(_eligibility_fixture())
        self.assertEqual(masks.dc.tolist(), [True, False, False, False, False])
        self.assertEqual(masks.sh1.tolist(), [True, False, False, True, False])

    def test_sh1_rank_threshold_is_strict(self):
        target = StreamStatistics(
            dc=_carrier(3, 1),
            sh1=_carrier(3, 3),
            fold_full_rss=torch.zeros(4, dtype=torch.float64),
        )
        target.sh1.gram[0, :, 0, 0] = 1e-8
        target.sh1.gram[1, :, 0, 0] = 1e-12
        boundary = 2.0 * SH1_RANK_RELATIVE_TRACE_THRESHOLD / (
            3.0 - SH1_RANK_RELATIVE_TRACE_THRESHOLD
        )
        self.assertEqual(
            boundary,
            SH1_RANK_RELATIVE_TRACE_THRESHOLD * (boundary + 2.0) / 3.0,
        )
        target.sh1.gram[2, :, 0, 0] = boundary
        masks = derive_eligibility(target)
        self.assertEqual(masks.sh1.tolist(), [True, False, False])

    def test_target_free_statistics_cannot_contain_a_residual_signal(self):
        target = _eligibility_fixture()
        leaked_rhs = target.dc.rhs.clone()
        leaked_rhs[0, 0, 0, 0] = 1.0
        target = _replace_stream(
            target, dc=_replace_carrier(target.dc, rhs=leaked_rhs)
        )
        with self.assertRaisesRegex(SceneInvalidError, "rhs is not identically zero"):
            derive_eligibility(target)

    def test_finite_asymmetric_gram_invalidates_instead_of_filtering(self):
        target = StreamStatistics(
            dc=_carrier(1, 1),
            sh1=_carrier(1, 3),
            fold_full_rss=torch.zeros(4, dtype=torch.float64),
        )
        target.sh1.gram[0, 0, 0, 1] = 0.25
        with self.assertRaisesRegex(SceneInvalidError, "not exactly symmetric"):
            derive_eligibility(target)


class FreezeIdentityTest(unittest.TestCase):
    def setUp(self):
        self.target = _eligibility_fixture()
        self.evaluation = _evaluation_from(self.target)
        self.masks = derive_eligibility(self.target)

    def test_unchanged_design_and_masks_are_accepted(self):
        validate_evaluation_identity(self.target, self.evaluation, self.masks)

    def test_a_mask_cannot_drop_or_add_a_group(self):
        changed = EligibilityMasks(self.masks.dc.clone(), self.masks.sh1.clone())
        changed.dc[0] = False
        with self.assertRaisesRegex(FreezeMismatchError, "eligibility mask"):
            validate_evaluation_identity(self.target, self.evaluation, changed)

    def test_gram_identity_is_bitwise_not_merely_numerically_equal(self):
        changed_gram = self.evaluation.sh1.gram.clone()
        changed_gram[0, 0, 0, 1] = -0.0
        changed_gram[0, 0, 1, 0] = -0.0
        self.assertTrue(torch.equal(changed_gram[0], self.target.sh1.gram[0]))
        changed = _replace_stream(
            self.evaluation,
            sh1=_replace_carrier(self.evaluation.sh1, gram=changed_gram),
        )
        with self.assertRaisesRegex(FreezeMismatchError, "Gram differs bitwise"):
            validate_evaluation_identity(self.target, changed, self.masks)

    def test_pixel_and_camera_support_drift_are_both_refused(self):
        pixels = self.evaluation.dc.support_pixels.clone()
        pixels[0, 0] += 1
        changed = _replace_stream(
            self.evaluation,
            dc=_replace_carrier(self.evaluation.dc, support_pixels=pixels),
        )
        with self.assertRaisesRegex(FreezeMismatchError, "pixel support"):
            validate_evaluation_identity(self.target, changed, self.masks)

        cameras = self.evaluation.sh1.support_cameras.clone()
        cameras[0, 0] += 1
        changed = _replace_stream(
            self.evaluation,
            sh1=_replace_carrier(self.evaluation.sh1, support_cameras=cameras),
        )
        with self.assertRaisesRegex(FreezeMismatchError, "camera support"):
            validate_evaluation_identity(self.target, changed, self.masks)


class SceneIntegrationTest(unittest.TestCase):
    def test_eligible_types_are_concatenated_in_locked_order(self):
        target = _eligibility_fixture()
        evaluation = _evaluation_from(target)
        masks = derive_eligibility(target)
        endpoints = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=torch.int64
        )
        outcome = evaluate_scene(
            scene="garden",
            candidate_endpoints=endpoints,
            target_free_statistics=target,
            evaluation_statistics=evaluation,
            frozen_masks=masks,
        )
        self.assertEqual(outcome.type_ids.tolist(), [0, 1, 1])
        self.assertEqual(outcome.endpoints.tolist(), [[0, 1], [0, 1], [6, 7]])
        self.assertEqual(outcome.result["eligible_groups"], {"DC": 1, "SH1": 2})
        self.assertEqual(set(outcome.result["families"]), {"DC", "SH1", "MIXED"})
        self.assertEqual(outcome.result["passing_families"], [])
        json.dumps(outcome.result, allow_nan=False, sort_keys=True)
        for family in ("DC", "SH1", "MIXED"):
            self.assertFalse(outcome.result["families"][family]["decision"]["pass"])

    def test_a_sealed_eligible_gcv_failure_invalidates_instead_of_dropping(self):
        dc = _carrier(1, 1)
        sh1 = _carrier(1, 3)
        sh1.support_pixels.zero_()
        sh1.support_cameras.zero_()
        target = StreamStatistics(
            dc=dc,
            sh1=sh1,
            fold_full_rss=torch.zeros(4, dtype=torch.float64),
        )
        masks = derive_eligibility(target)
        self.assertEqual(masks.dc.tolist(), [True])
        self.assertEqual(masks.sh1.tolist(), [False])
        evaluation = _evaluation_from(target)
        evaluation.dc.rhs.fill_(0.1)
        evaluation.dc.support_rss.zero_()
        with self.assertRaisesRegex(SceneInvalidError, "no group was dropped"):
            evaluate_scene(
                scene="garden",
                candidate_endpoints=torch.tensor([[0, 1]], dtype=torch.int64),
                target_free_statistics=target,
                evaluation_statistics=evaluation,
                frozen_masks=masks,
            )

    def test_a_type_with_zero_eligible_groups_is_a_recorded_family_fail(self):
        dc = _carrier(1, 1)
        sh1 = _carrier(1, 3)
        sh1.support_pixels.zero_()
        sh1.support_cameras.zero_()
        target = StreamStatistics(
            dc=dc,
            sh1=sh1,
            fold_full_rss=torch.zeros(4, dtype=torch.float64),
        )
        evaluation = _evaluation_from(target)
        outcome = evaluate_scene(
            scene="room",
            candidate_endpoints=torch.tensor([[0, 1]], dtype=torch.int64),
            target_free_statistics=target,
            evaluation_statistics=evaluation,
            frozen_masks=derive_eligibility(target),
        )
        self.assertEqual(outcome.type_ids.tolist(), [0])
        self.assertEqual(outcome.result["eligible_groups"], {"DC": 1, "SH1": 0})
        self.assertFalse(
            outcome.result["families"]["SH1"]["metrics"]["metrics_applicable"]
        )
        self.assertFalse(outcome.result["families"]["SH1"]["decision"]["pass"])

    def test_wrong_dtype_is_invalid_before_any_scoring(self):
        target = _eligibility_fixture()
        target = _replace_stream(
            target,
            dc=_replace_carrier(target.dc, gram=target.dc.gram.float()),
        )
        with self.assertRaisesRegex(SceneInvalidError, "must use float64"):
            derive_eligibility(target)


if __name__ == "__main__":
    unittest.main()
