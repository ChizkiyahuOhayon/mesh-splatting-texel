"""Run the CPU-only GoRFE-D0-A audit on sealed V1 state tensors."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess

import torch

from gorfe_d0 import audit_policy_state, crosscheck_v1_portfolios
from gorfe_v1 import nested_scores
from gorfe_v1_evaluate_core import PolicyScores, validate_evaluation_identity
from gorfe_v1_io import (
    artifact_records,
    sha256_file,
    write_json_new,
    write_sha256s,
    write_text_new,
)
from gorfe_v1_state import load_candidate_state, stream_from_payload
from gorfe_v1_stream import CarrierStatistics


REPOSITORY = Path(__file__).resolve().parent
PROTOCOL = REPOSITORY / "experiments" / "gorfe_d0" / "protocol.md"
INPUT_IDENTITY = REPOSITORY / "experiments" / "gorfe_d0" / "input_identity.json"
FREEZE = REPOSITORY / "experiments" / "gorfe_v1" / "candidate_freeze_04.json"
SCENES = ("garden", "room")


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_hash(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular input is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"sealed input hash mismatch: {path}")
    return path


def _verify_ledger(path):
    checked = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if len(digest) != 64 or not separator or not name:
            raise ValueError(f"malformed SHA-256 ledger row: {line!r}")
        _require_hash(Path(name), digest)
        checked += 1
    if not checked:
        raise ValueError(f"empty SHA-256 ledger: {path}")
    return checked


def _verify_inputs(prepare_root, evaluate_root, identity):
    if (prepare_root / "DONE").read_text(encoding="utf-8") != "complete\n":
        raise ValueError("V1 preparation completion sentinel is malformed")
    if (evaluate_root / "DONE").read_text(encoding="utf-8") != "complete\n":
        raise ValueError("V1 evaluation completion sentinel is malformed")
    paths = {}
    for relative, digest in identity["preparation_artifacts"].items():
        paths[f"prepare_{relative.replace('/', '_')}"] = _require_hash(
            prepare_root / relative, digest
        )
    for relative, digest in identity["evaluation_root_artifacts"].items():
        paths[f"evaluate_{relative.replace('.', '_')}"] = _require_hash(
            evaluate_root / relative, digest
        )
    _require_hash(FREEZE, identity["candidate_freeze_sha256"])
    prepare_rows = _verify_ledger(prepare_root / "SHA256SUMS")
    evaluate_rows = _verify_ledger(evaluate_root / "SHA256SUMS")
    decision = _json(evaluate_root / "decision.json")
    if decision.get("decision") != "fail" or decision.get("phase") != "overall":
        raise ValueError("D0-A requires the sealed valid V1 failure")
    root_manifest = _json(evaluate_root / "manifest.json")
    if root_manifest.get("source_revision") != identity["v1_evaluation_revision"]:
        raise ValueError("sealed V1 evaluation revision changed")
    return paths, {"prepare_ledger_rows": prepare_rows, "evaluate_ledger_rows": evaluate_rows}


def _subset(carrier, mask):
    return CarrierStatistics(
        gram=carrier.gram[mask],
        rhs=carrier.rhs[mask],
        support_rss=carrier.support_rss[mask],
        support_pixels=carrier.support_pixels[mask],
        support_cameras=carrier.support_cameras[mask],
    )


def _policy(carrier, mask):
    value = nested_scores(_subset(carrier, mask))
    return PolicyScores(
        primary=value.primary,
        outcome=value.outcome,
        raw_residual=value.raw_residual,
        same_view_gain=value.same_view_gain,
        rhs_norm=value.rhs_norm,
        coverage=value.coverage,
    )


def _combine(first, second):
    return PolicyScores(
        **{
            name: torch.cat((getattr(first, name), getattr(second, name)), dim=0)
            for name in PolicyScores.__dataclass_fields__
        }
    )


def _load_evaluation_state(path, scene):
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema", "scene", "statistics", "eligible_type_ids", "eligible_endpoints"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("evaluation state has unexpected fields")
    if value["schema"] != "gorfe-v1-evaluation-state-v1" or value["scene"] != scene:
        raise ValueError("evaluation state identity changed")
    return value, stream_from_payload(value["statistics"])


def _scene_audit(scene, prepare_root, evaluate_root, identity):
    result_path = evaluate_root / scene / "result.json"
    _require_hash(result_path, identity["scene_result_sha256"][scene])
    sealed_result = _json(result_path)
    candidate, target_free, masks = load_candidate_state(
        prepare_root / scene / "candidate_state.pt", expected_scene=scene
    )
    saved, evaluation = _load_evaluation_state(
        evaluate_root / scene / "evaluation_state.pt", scene
    )
    validate_evaluation_identity(target_free, evaluation, masks)

    dc = _policy(evaluation.dc, masks.dc)
    sh1 = _policy(evaluation.sh1, masks.sh1)
    nested = _combine(dc, sh1)
    dc_rows = torch.nonzero(masks.dc, as_tuple=False).flatten()
    sh1_rows = torch.nonzero(masks.sh1, as_tuple=False).flatten()
    type_ids = torch.cat(
        (
            torch.zeros(dc_rows.numel(), dtype=torch.int64),
            torch.ones(sh1_rows.numel(), dtype=torch.int64),
        )
    )
    endpoints = torch.cat(
        (candidate["candidate_edges"][dc_rows], candidate["candidate_edges"][sh1_rows]), dim=0
    )
    if not torch.equal(saved["eligible_type_ids"], type_ids) or not torch.equal(
        saved["eligible_endpoints"], endpoints
    ):
        raise ValueError("saved eligible identity differs from the frozen candidate state")
    audit = audit_policy_state(
        scene=scene,
        nested=nested,
        type_ids=type_ids,
        endpoints=endpoints,
        outer_sse=evaluation.fold_full_rss,
    )
    audit["v1_portfolio_crosscheck"] = crosscheck_v1_portfolios(audit, sealed_result)
    audit["input_artifacts"] = artifact_records(
        {
            "candidate_state": prepare_root / scene / "candidate_state.pt",
            "evaluation_state": evaluate_root / scene / "evaluation_state.pt",
            "sealed_result": result_path,
        }
    )
    return audit


def run(*, prepare_root, evaluate_root, output):
    prepare_root = Path(prepare_root).resolve()
    evaluate_root = Path(evaluate_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite D0-A output: {output}")
    if torch.cuda.is_available():
        raise RuntimeError("D0-A must run with CUDA hidden")
    identity = _json(INPUT_IDENTITY)
    if identity.get("schema") != "gorfe-d0-a-input-identity-v1":
        raise ValueError("D0-A input identity schema changed")
    input_paths, ledger = _verify_inputs(prepare_root, evaluate_root, identity)
    scenes = {}
    for scene in SCENES:
        scenes[scene] = _scene_audit(scene, prepare_root, evaluate_root, identity)
        gc.collect()
    result = {
        "schema": "gorfe-d0-a-result-v1",
        "experiment": "GoRFE-D0-A",
        "decision": "diagnostic_no_pass_fail",
        "source_v1_decision": "fail",
        "scenes": scenes,
        "limitations": [
            "topology overlap is not pixel-support overlap",
            "sealed V1 states contain no cross-group Gram",
            "joint portfolio utility is not computed",
            "official test images are not accessed",
        ],
    }
    output.mkdir(parents=True)
    result_path = write_json_new(output / "result.json", result)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    manifest = {
        "schema": "gorfe-d0-a-manifest-v1",
        "experiment": "GoRFE-D0-A",
        "decision": "diagnostic_no_pass_fail",
        "source_revision": revision,
        "torch": torch.__version__,
        "cuda_visible": torch.cuda.is_available(),
        "protocol": str(PROTOCOL.relative_to(REPOSITORY)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "input_identity_sha256": sha256_file(INPUT_IDENTITY),
        "input_roots": {"prepare": str(prepare_root), "evaluate": str(evaluate_root)},
        "verified_root_artifacts": artifact_records(input_paths),
        "verified_ledger_rows": ledger,
        "artifacts": artifact_records({"result": result_path}),
    }
    manifest_path = write_json_new(output / "manifest.json", manifest)
    checksum_path = write_sha256s(output / "SHA256SUMS", (result_path, manifest_path))
    done_path = write_text_new(output / "DONE", "complete\n")
    return {
        "result": str(result_path),
        "manifest": str(manifest_path),
        "checksums": str(checksum_path),
        "done": str(done_path),
        "summary": {
            scene: {
                family: scenes[scene]["families"][family]["summary"]
                for family in ("DC", "SH1", "MIXED")
            }
            for scene in SCENES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--evaluate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(**vars(arguments)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
