import unittest

import torch

from gorfe_v1_topology_census import census_faces


class TopologyCensusTest(unittest.TestCase):
    def test_manifold_mesh_has_no_exclusions(self):
        faces = torch.tensor(
            [[0, 1, 2], [1, 0, 3]], dtype=torch.int32
        )
        result = census_faces(faces, vertex_count=4, scan_chunk=2)
        self.assertEqual(result["incidence_histogram"], {"1": 4, "2": 1})
        self.assertEqual(result["nonmanifold_edge_count"], 0)
        self.assertEqual(result["canonical_edge_count_including_nonmanifold"], 5)

    def test_genuine_three_face_junction_is_distinguished(self):
        faces = torch.tensor(
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=torch.int64
        )
        result = census_faces(faces, vertex_count=5, scan_chunk=2)
        self.assertEqual(result["nonmanifold_edge_count"], 1)
        self.assertEqual(
            result["nonmanifold_edges_with_all_incident_faces_distinct"], 1
        )
        self.assertEqual(
            result["nonmanifold_edges_with_duplicate_incident_face"], 0
        )
        self.assertEqual(result["first_nonmanifold_edges"][0]["edge"], [0, 1])

    def test_duplicate_triangle_cause_is_reported(self):
        faces = torch.tensor(
            [[0, 1, 2], [1, 0, 3], [2, 1, 0]], dtype=torch.int32
        )
        result = census_faces(faces, vertex_count=4, scan_chunk=2)
        self.assertEqual(result["nonmanifold_edge_count"], 1)
        self.assertEqual(
            result["nonmanifold_edges_with_duplicate_incident_face"], 1
        )
        self.assertEqual(result["duplicate_incident_face_occurrences"], 1)

    def test_invalid_face_contract_is_refused(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            census_faces(
                torch.tensor([[0, 0, 1]], dtype=torch.int64), vertex_count=2
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            census_faces(
                torch.tensor([[0, 1, 2]], dtype=torch.int64), vertex_count=2
            )


if __name__ == "__main__":
    unittest.main()
