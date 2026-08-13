import inspect
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

import gorfe_v1_evaluate as cli
from gorfe_v1_prepare_core import IdentityDriftError


class FreezeCommitTest(unittest.TestCase):
    def test_scene_wall_clock_starts_before_freeze_and_checkpoint_preflight(self):
        source = inspect.getsource(cli.run_scene)
        clock = source.index("invocation_start = time.perf_counter()")
        freeze = source.index("freeze = _load_json(freeze_file)")
        checkpoint = source.index("checkpoint_state, checkpoint = load_validated_checkpoint")
        self.assertLess(clock, freeze)
        self.assertLess(clock, checkpoint)
        self.assertIn("time.perf_counter() - invocation_start", source)

    def test_only_the_tracked_freeze_may_differ_from_implementation(self):
        freeze = cli.REPOSITORY / "experiments/gorfe_v1/candidate_freeze_01.json"

        def git(*arguments):
            if arguments[:2] == ("status", "--porcelain"):
                return ""
            if arguments[:2] == ("rev-parse", "HEAD"):
                return "evaluation"
            if arguments[:2] == ("diff", "--name-only"):
                return "experiments/gorfe_v1/candidate_freeze_01.json"
            if arguments[0] == "ls-files":
                return "experiments/gorfe_v1/candidate_freeze_01.json"
            raise AssertionError(arguments)

        with mock.patch.object(cli, "_git", side_effect=git), mock.patch.object(
            cli.subprocess, "check_call", return_value=0
        ):
            self.assertEqual(
                cli.verify_tracked_freeze_commit(freeze, "implementation"),
                "evaluation",
            )

        def drifted(*arguments):
            value = git(*arguments)
            if arguments[:2] == ("diff", "--name-only"):
                return value + "\ngorfe_v1.py"
            return value

        with mock.patch.object(cli, "_git", side_effect=drifted), mock.patch.object(
            cli.subprocess, "check_call", return_value=0
        ), self.assertRaisesRegex(IdentityDriftError, "only by its freeze"):
            cli.verify_tracked_freeze_commit(freeze, "implementation")

    def test_target_loader_is_unreachable_when_freeze_verification_fails(self):
        freeze = cli.REPOSITORY / "experiments/gorfe_v1/candidate_freeze_01.json"
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            cli,
            "_load_json",
            return_value={"source_revision": "implementation"},
        ), mock.patch.object(
            cli, "verify_tracked_freeze_commit", return_value="evaluation"
        ), mock.patch.object(
            cli, "preparation_artifact_paths", return_value={}
        ), mock.patch.object(
            cli,
            "verify_candidate_freeze_payload",
            side_effect=IdentityDriftError("frozen artifact drift"),
        ), mock.patch.object(cli, "load_training_rgb") as target_loader:
            with self.assertRaisesRegex(IdentityDriftError, "frozen artifact drift"):
                cli.run_scene(
                    scene="garden",
                    dataset_root=Path(directory) / "dataset",
                    model_root=Path(directory) / "model",
                    prepare_root=Path(directory) / "prepare",
                    freeze_file=freeze,
                    output=Path(directory) / "output",
                    physical_gpu=3,
                )
        target_loader.assert_not_called()


class FrozenEndpointTest(unittest.TestCase):
    def test_verified_state_is_the_only_endpoint_source(self):
        endpoints = torch.tensor([[2, 7]], dtype=torch.int64)
        sentinel = object()
        with mock.patch.object(cli, "evaluate_scene", return_value=sentinel) as evaluate:
            observed = cli.evaluate_frozen_candidate_state(
                scene="garden",
                candidate_state={"candidate_edges": endpoints},
                target_free_statistics="target-free",
                evaluation_statistics="evaluation",
                frozen_masks="masks",
            )
        self.assertIs(observed, sentinel)
        self.assertIs(evaluate.call_args.kwargs["candidate_endpoints"], endpoints)
        with self.assertRaisesRegex(IdentityDriftError, "endpoint"):
            cli.evaluate_frozen_candidate_state(
                scene="garden",
                candidate_state={},
                target_free_statistics=None,
                evaluation_statistics=None,
                frozen_masks=None,
            )


class MetadataIdentityTest(unittest.TestCase):
    def test_pose_metadata_hash_and_size_drift_are_refused(self):
        expected = {"images": {"bytes": 3, "sha256": "abc", "path": "/old"}}
        cli._require_same_metadata(
            expected,
            {"images": {"bytes": 3, "sha256": "abc", "path": "/new"}},
        )
        with self.assertRaisesRegex(IdentityDriftError, "images"):
            cli._require_same_metadata(
                expected,
                {"images": {"bytes": 4, "sha256": "abc", "path": "/new"}},
            )


if __name__ == "__main__":
    unittest.main()
