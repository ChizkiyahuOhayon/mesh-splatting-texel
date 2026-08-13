# GoRFE-V1 implementation record

Status: **IMPLEMENTED; REAL-SCENE RESULT PENDING**.  No Garden or Room target
pixel was read while implementing or locally verifying this code.

## Scientific object

The executable object is fixed by `protocol.md` and
`protocol_constants.json`.  V1 evaluates whether a nested, renderer-exact
`P2 x {DC,SH1}` group value predicts signed squared-error reduction on an
unseen training-camera fold better than all six locked controls.  It is a
prospective selector gate, not a training run or a quality claim.

## Implementation map

- `gorfe_v1_prepare_core.py`, `gorfe_v1_scene.py`: COLMAP-only camera identity,
  explicit iteration-30000 checkpoint validation, canonical candidate
  topology, official-split guard, and target-decode sentinel.
- `cuda_rasterizer/gorfe.{h,cu}` and `rasterize_points.cu`: two-pass replay of
  the saved forward buffers.  The native exporter receives no target,
  dominant-face map, backward state, or `was_rendered` tensor.
- `gorfe_v1_stream.py`: complete-camera float64 reduction by
  `(final-resolution pixel, candidate edge)` before any Gram outer product.
- `gorfe_v1.py`, `gorfe_v1_evaluate_core.py`: locked GCV, strict nested
  outer-fold scores, controls, cost-aware policies, signed outcomes, scene
  predicates, and the two-scene decision.
- `gorfe_v1_prepare.py`, `gorfe_v1_evaluate.py`: separate write-once scene
  phases with identity and resource checks.
- `gorfe_v1_finalize.py`, `gorfe_v1_freeze_payload.py`,
  `gorfe_v1_freeze_verify.py`: artifact sealing and the mandatory tracked
  hash-only boundary between preparation and target access.

## Authorization boundary

Run only `run_prepare_gpu3.sh` first.  It validates both scene identities before
reserving a persistent attempt, builds one native wheel, runs the repository
and A40 native gates, and prepares Garden and Room serially without decoding
targets.  Its generated candidate-freeze JSON must be reviewed and committed as
`experiments/gorfe_v1/candidate_freeze_<attempt>.json`.  The evaluation runner
rejects any other tracked delta from the implementation revision and must not be
run before that commit.

## Verification completed locally

- 315 pure-Python repository tests under warnings-as-errors;
- Python bytecode compilation, shell syntax, and `git diff --check`;
- locked Garden/Room train/test counts, name hashes, and four-fold sizes from
  local COLMAP metadata without image decode;
- independent float64 GCV oracle checks for both feature dimensions.

The local machine has no NVIDIA CUDA compiler or A40.  Therefore native wheel
compilation, forward-replay bitwise checks, sparse/carrier equivalence, physical
GPU peak monitoring, and real-scene preparation remain mandatory server gates.
No V1 pass/fail or MeshSplatting improvement is claimed before those artifacts
exist.
