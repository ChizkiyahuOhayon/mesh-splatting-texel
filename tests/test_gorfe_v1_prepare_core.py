import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

from gorfe_v1_prepare_core import (
    IdentityDriftError,
    OfficialSplitGuard,
    TargetAccessError,
    TargetDecodeSentinel,
    artifact_identities,
    build_candidate_freeze_payload,
    build_candidate_topology,
    canonical_json_bytes,
    deterministic_fold_map,
    fold_map_sha256,
    load_validated_checkpoint,
    name_list_sha256,
    read_colmap_metadata,
    select_splitmix64_indices,
    splitmix64_priority,
    validate_checkpoint_state,
    verify_candidate_freeze_payload,
)


SEED = 15111065706836454659


def _checkpoint_state():
    vertex_count = 4
    faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int32)
    return {
        "triangles_points": torch.arange(vertex_count * 3, dtype=torch.float32).reshape(vertex_count, 3),
        "_triangle_indices": faces,
        "vertex_weight": torch.zeros((vertex_count, 1), dtype=torch.float32),
        "sigma": math.log(1e-4),
        "active_sh_degree": 3,
        "features_dc": torch.zeros((vertex_count, 1, 3), dtype=torch.float32),
        "features_rest": torch.zeros((vertex_count, 15, 3), dtype=torch.float32),
        "importance_score": torch.zeros(faces.shape[0], dtype=torch.float32),
        "image_size": torch.zeros(faces.shape[0], dtype=torch.float32),
        "pixel_count": torch.zeros(faces.shape[0], dtype=torch.int32),
        "texel_order": 0,
    }


def _write_colmap_text(root: Path, names):
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# metadata only\n1 PINHOLE 80 60 50 51 40 30\n", encoding="utf-8"
    )
    rows = ["# metadata only"]
    for image_id, name in enumerate(names, start=1):
        rows.extend(
            [
                f"{image_id} 1 0 0 0 0 0 0 1 nested/{name}.JPG",
                "",
            ]
        )
    (sparse / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_colmap_binary(root: Path, names):
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 80, 60))
        handle.write(struct.pack("<dddd", 50.0, 51.0, 40.0, 30.0))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(names)))
        for image_id, name in enumerate(names, start=1):
            handle.write(
                struct.pack(
                    "<idddddddi", image_id, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1
                )
            )
            handle.write(f"nested/{name}.JPG".encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 1))
            handle.write(struct.pack("<ddq", 2.0, 3.0, -1))


class CameraMetadataTest(unittest.TestCase):
    def test_preregistered_garden_name_hashes_use_trailing_newline(self):
        all_names = [f"DSC{number:05d}" for number in range(7956, 8141)]
        test = all_names[::8]
        train = [name for rank, name in enumerate(all_names) if rank % 8]
        self.assertEqual(
            name_list_sha256(test),
            "efad2e5e2dbfef071d2ef98a91541706d4d829ec8d2170daf9edf7a5d1aa0230",
        )
        self.assertEqual(
            name_list_sha256(train),
            "858a34eeaf9cc908e211fa85cbdcf838c1faabb8b1cdac9fa95ddd93c7e59c85",
        )

    def test_names_are_utf8_sorted_and_fold_map_is_canonical_json(self):
        names = ["z", "é", "a", "b"]
        folds = deterministic_fold_map(names, fold_count=3)
        ordered = sorted(names, key=lambda name: name.encode("utf-8"))
        self.assertEqual(folds, {name: rank % 3 for rank, name in enumerate(ordered)})
        expected = hashlib.sha256(
            json.dumps(
                folds, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(fold_map_sha256(folds), expected)

    def test_duplicate_or_delimiter_camera_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            deterministic_fold_map(["same", "same"])
        with self.assertRaisesRegex(ValueError, "delimiter"):
            name_list_sha256(["bad\nname"])

    def test_colmap_reader_returns_train_metadata_but_test_names_only(self):
        names = [f"frame{index:02d}" for index in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_colmap_text(root, list(reversed(names)))
            split = read_colmap_metadata(root)
        self.assertEqual(split.test_names, ("frame00", "frame08"))
        self.assertEqual(
            [camera.image_name for camera in split.train_cameras],
            [name for index, name in enumerate(names) if index % 8],
        )
        self.assertEqual(split.fold_sizes, (2, 2, 2, 2))
        self.assertTrue(all(camera.relative_image_path.startswith("nested/") for camera in split.train_cameras))
        self.assertFalse(hasattr(split, "test_cameras"))

    def test_metadata_read_does_not_touch_missing_image_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_colmap_text(root, [f"frame{index:02d}" for index in range(9)])
            image_root = root / "images"
            image_root.mkdir()
            with TargetDecodeSentinel([image_root]) as sentinel:
                split = read_colmap_metadata(root)
            sentinel.assert_clean()
        self.assertEqual(len(split.train_cameras), 7)

    def test_binary_metadata_skips_point_rows_without_image_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_colmap_binary(root, [f"frame{index:02d}" for index in range(9)])
            split = read_colmap_metadata(root)
        self.assertEqual(split.metadata_format, "colmap_binary")
        self.assertEqual(split.test_names, ("frame00", "frame08"))
        self.assertEqual(split.train_cameras[0].parameters, (50.0, 51.0, 40.0, 30.0))

    def test_official_test_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_colmap_text(root, [f"frame{index:02d}" for index in range(9)])
            split = read_colmap_metadata(root)
        guard = OfficialSplitGuard(split)
        guard.require_training("frame01")
        with self.assertRaisesRegex(TargetAccessError, "official test"):
            guard.require_training("frame00")
        with self.assertRaisesRegex(TargetAccessError, "unknown"):
            guard.require_training("not-a-camera")


class SplitMixAndTopologyTest(unittest.TestCase):
    @staticmethod
    def _reference(index):
        mask = (1 << 64) - 1
        z = (index + SEED + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return (z ^ (z >> 31)) & mask

    def test_splitmix_matches_scalar_reference_and_chunking(self):
        self.assertEqual([splitmix64_priority(i, SEED) for i in range(10)], [self._reference(i) for i in range(10)])
        expected = sorted(range(1000), key=lambda index: (self._reference(index), index))[:37]
        expected = sorted(expected)
        indices_a, hashes_a = select_splitmix64_indices(1000, seed=SEED, cap=37, chunk_size=53)
        indices_b, hashes_b = select_splitmix64_indices(1000, seed=SEED, cap=37, chunk_size=1000)
        self.assertEqual(indices_a.tolist(), expected)
        np.testing.assert_array_equal(indices_a, indices_b)
        np.testing.assert_array_equal(hashes_a, hashes_b)
        self.assertEqual(hashes_a.tolist(), [self._reference(index) for index in expected])

    def test_two_incident_faces_share_one_compact_candidate(self):
        faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int32)
        topology = build_candidate_topology(faces, seed=SEED, cap=10, chunk_faces=1)
        self.assertEqual(topology.edge_count, 5)
        self.assertEqual(topology.candidate_edges.tolist(), [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]])
        self.assertEqual(int(topology.face_candidate_edges[0, 0]), 0)
        self.assertEqual(int(topology.face_candidate_edges[1, 0]), 0)
        self.assertEqual(int(topology.candidate_incident_face_counts[0]), 2)
        self.assertEqual(topology.candidate_incident_face_counts.dtype, torch.int32)

    def test_compact_map_uses_minus_one_for_non_candidates(self):
        faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int64)
        topology = build_candidate_topology(faces, seed=SEED, cap=1)
        self.assertEqual(topology.candidate_edges.shape, (1, 2))
        self.assertEqual(
            int((topology.face_candidate_edges >= 0).sum()),
            int(topology.candidate_incident_face_counts.sum()),
        )
        self.assertTrue(bool((topology.face_candidate_edges >= -1).all()))

    def test_nonmanifold_star_maps_all_incident_faces_to_one_candidate(self):
        faces = torch.tensor([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=torch.int64)
        topology = build_candidate_topology(faces, seed=SEED, cap=10, chunk_faces=1)
        matches = torch.nonzero(
            (topology.candidate_edges == torch.tensor([0, 1])).all(dim=1),
            as_tuple=False,
        ).flatten()
        self.assertEqual(matches.numel(), 1)
        group = int(matches[0])
        self.assertEqual(int(topology.candidate_incident_face_counts[group]), 3)
        self.assertEqual(int((topology.face_candidate_edges == group).sum()), 3)

    def test_repeated_vertex_faces_are_refused(self):
        with self.assertRaisesRegex(ValueError, "repeats a vertex"):
            build_candidate_topology(torch.tensor([[0, 0, 1]], dtype=torch.int64), seed=SEED)

    def test_empty_topology_is_well_typed(self):
        result = build_candidate_topology(torch.empty((0, 3), dtype=torch.int32), seed=SEED)
        self.assertEqual(result.edge_count, 0)
        self.assertEqual(result.face_candidate_edges.dtype, torch.int32)
        self.assertEqual(result.candidate_incident_face_counts.dtype, torch.int32)
        self.assertEqual(result.candidate_edges.shape, (0, 2))


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_semantics_and_shapes_are_bound(self):
        details = validate_checkpoint_state(_checkpoint_state())
        self.assertEqual(details["vertex_count"], 4)
        self.assertEqual(details["face_count"], 2)
        self.assertAlmostEqual(details["activated_sigma"], 1e-4, places=15)
        self.assertEqual(details["tensor_shapes"]["features_rest"], [4, 15, 3])

    def test_checkpoint_shape_sigma_sh_texel_and_key_errors_are_refused(self):
        mutations = []
        missing = _checkpoint_state()
        missing.pop("features_dc")
        mutations.append((missing, "missing required keys"))
        bad_shape = _checkpoint_state()
        bad_shape["features_rest"] = torch.zeros((4, 14, 3))
        mutations.append((bad_shape, "invalid shape"))
        bad_sigma = _checkpoint_state()
        bad_sigma["sigma"] = math.log(2e-4)
        mutations.append((bad_sigma, "activated sigma"))
        bad_sh = _checkpoint_state()
        bad_sh["active_sh_degree"] = 2
        mutations.append((bad_sh, "active_sh_degree"))
        bad_texel = _checkpoint_state()
        bad_texel["texel_order"] = 1
        mutations.append((bad_texel, "texel_order"))
        orphan_texel = _checkpoint_state()
        orphan_texel["texels"] = torch.empty(0)
        mutations.append((orphan_texel, "must not contain texels"))
        for state, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex((TypeError, ValueError), message):
                validate_checkpoint_state(state)

    def test_explicit_iteration_path_and_sha_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            checkpoint = model / "point_cloud" / "iteration_30000" / "point_cloud_state_dict.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(_checkpoint_state(), checkpoint)
            _, inspection = load_validated_checkpoint(model)
            self.assertEqual(
                inspection.sha256,
                artifact_identities({"checkpoint": checkpoint})["checkpoint"]["sha256"],
            )
            state = _checkpoint_state()
            state["features_dc"][0, 0, 0] = 1
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(IdentityDriftError, "SHA-256"):
                load_validated_checkpoint(model, expected_sha256=inspection.sha256)
            with self.assertRaisesRegex(ValueError, "iteration 30000"):
                load_validated_checkpoint(model, iteration=29999)


class SentinelAndFreezeTest(unittest.TestCase):
    def test_target_attempt_is_sticky_even_if_decode_error_is_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory) / "images"
            image_root.mkdir()
            target = image_root / "frame.jpg"
            target.write_bytes(b"not an image")
            with TargetDecodeSentinel([image_root]) as sentinel:
                with self.assertRaises(TargetAccessError):
                    open(target, "rb")
                with self.assertRaises(TargetAccessError):
                    sentinel.assert_clean()
            self.assertEqual(sentinel.attempted_paths, (str(target.resolve()),))

    def test_non_target_metadata_remains_readable_under_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            metadata = root / "images.bin"
            metadata.write_bytes(b"metadata")
            with TargetDecodeSentinel([image_root]) as sentinel:
                self.assertEqual(metadata.read_bytes(), b"metadata")
            self.assertEqual(sentinel.manifest_record()["blocked_attempts"], 0)

    def test_freeze_helpers_detect_content_size_and_logical_name_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "candidate.pt"
            second = root / "support.pt"
            first.write_bytes(b"candidate")
            second.write_bytes(b"support")
            artifacts = {"candidate": first, "support": second}
            freeze = build_candidate_freeze_payload(
                source_revision="abc",
                protocol_sha256="protocol",
                constants_sha256="constants",
                preparation_artifacts=artifacts,
            )
            verify_candidate_freeze_payload(
                freeze,
                source_revision="abc",
                protocol_sha256="protocol",
                constants_sha256="constants",
                preparation_artifacts=artifacts,
            )
            first.write_bytes(b"changed")
            with self.assertRaisesRegex(IdentityDriftError, "artifact drift"):
                verify_candidate_freeze_payload(
                    freeze,
                    source_revision="abc",
                    protocol_sha256="protocol",
                    constants_sha256="constants",
                    preparation_artifacts=artifacts,
                )

    def test_canonical_json_refuses_nonfinite_numbers(self):
        with self.assertRaises(ValueError):
            canonical_json_bytes({"invalid": float("nan")})


if __name__ == "__main__":
    unittest.main()
