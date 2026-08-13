from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gorfe_v1_evaluate_core import EligibilityMasks
from gorfe_v1_state import candidate_state_payload, load_candidate_state
from gorfe_v1_stream import CarrierStatistics, StreamStatistics


def _carrier(groups, feature_dim):
    return CarrierStatistics(
        gram=torch.zeros((groups, 4, feature_dim, feature_dim), dtype=torch.float64),
        rhs=torch.zeros((groups, 4, feature_dim, 3), dtype=torch.float64),
        support_rss=torch.zeros((groups, 4), dtype=torch.float64),
        support_pixels=torch.zeros((groups, 4), dtype=torch.int64),
        support_cameras=torch.zeros((groups, 4), dtype=torch.int64),
    )


class CandidateStateTest(unittest.TestCase):
    def test_round_trip_preserves_every_tensor_bitwise(self):
        groups = 2
        topology = SimpleNamespace(
            edge_count=7,
            vertex_stride=5,
            candidate_edge_indices=torch.tensor([1, 6], dtype=torch.int64),
            candidate_edges=torch.tensor([[0, 1], [3, 4]], dtype=torch.int64),
            candidate_hashes=np.array([2, 9], dtype=np.uint64),
            candidate_incident_face_counts=torch.tensor([1, 3], dtype=torch.int32),
            face_candidate_edges=torch.tensor(
                [[0, -1, 1], [-1, -1, 1], [1, -1, -1]], dtype=torch.int32
            ),
        )
        statistics = StreamStatistics(
            _carrier(groups, 1), _carrier(groups, 3), torch.zeros(4, dtype=torch.float64)
        )
        eligibility = EligibilityMasks(
            torch.tensor([True, False]), torch.tensor([False, True])
        )
        payload = candidate_state_payload(
            scene="garden", topology=topology, statistics=statistics, eligibility=eligibility
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(payload, path)
            loaded, restored, masks = load_candidate_state(path, expected_scene="garden")
        self.assertTrue(torch.equal(loaded["candidate_edges"], topology.candidate_edges))
        self.assertEqual(int(loaded["candidate_incident_face_counts"][1]), 3)
        self.assertTrue(torch.equal(restored.sh1.gram, statistics.sh1.gram))
        self.assertTrue(torch.equal(masks.dc, eligibility.dc))
        self.assertTrue(torch.equal(masks.sh1, eligibility.sh1))

    def test_face_map_and_incidence_identity_are_checked_on_load(self):
        topology = SimpleNamespace(
            edge_count=1,
            vertex_stride=3,
            candidate_edge_indices=torch.tensor([0], dtype=torch.int64),
            candidate_edges=torch.tensor([[0, 1]], dtype=torch.int64),
            candidate_hashes=np.array([7], dtype=np.uint64),
            candidate_incident_face_counts=torch.tensor([1], dtype=torch.int32),
            face_candidate_edges=torch.tensor([[0, -1, -1]], dtype=torch.int32),
        )
        statistics = StreamStatistics(
            _carrier(1, 1), _carrier(1, 3), torch.zeros(4, dtype=torch.float64)
        )
        payload = candidate_state_payload(
            scene="garden",
            topology=topology,
            statistics=statistics,
            eligibility=EligibilityMasks(torch.tensor([True]), torch.tensor([False])),
        )
        payload["face_candidate_edges"][0, 1] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "incidence"):
                load_candidate_state(path, expected_scene="garden", expected_face_count=1)

    def test_scene_drift_is_refused(self):
        topology = SimpleNamespace(
            edge_count=0,
            vertex_stride=0,
            candidate_edge_indices=torch.empty(0, dtype=torch.int64),
            candidate_edges=torch.empty((0, 2), dtype=torch.int64),
            candidate_hashes=np.empty(0, dtype=np.uint64),
            candidate_incident_face_counts=torch.empty(0, dtype=torch.int32),
            face_candidate_edges=torch.empty((0, 3), dtype=torch.int32),
        )
        statistics = StreamStatistics(
            _carrier(0, 1), _carrier(0, 3), torch.zeros(4, dtype=torch.float64)
        )
        payload = candidate_state_payload(
            scene="garden",
            topology=topology,
            statistics=statistics,
            eligibility=EligibilityMasks(torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_candidate_state(path, expected_scene="room")


if __name__ == "__main__":
    unittest.main()
