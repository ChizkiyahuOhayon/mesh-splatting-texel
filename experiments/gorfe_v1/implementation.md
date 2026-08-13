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
  explicit iteration-30000 checkpoint validation, canonical edge-star
  candidates for arbitrary positive incidence, official-split guard, and
  target-decode sentinel.
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

## Verification completed

- 321 pure-Python repository tests under warnings-as-errors;
- Python bytecode compilation, shell syntax, and `git diff --check`;
- locked Garden/Room train/test counts, name hashes, and four-fold sizes from
  local COLMAP metadata without image decode;
- a checkpoint-only Garden/Room topology census that motivated and fixed the
  complete edge-star contract before target access;
- independent float64 GCV oracle checks for both feature dimensions.

The repaired native wheel also passed every registered A40 gate in target-free
Prepare attempt `_02`: all six parent outputs were bitwise unchanged, replay
diagnostics were exact, both depth layers reduced into the same edge groups,
sparse/carrier reconstruction agreed to `7.79e-8`, and the squared-loss
gradient identity error was zero.  The subsequent topology census and edge-star
implementation changed no CUDA exporter code, but a fresh Prepare attempt must
still rebuild and rerun that gate before sealing candidates.  Real-scene
preparation remains pending.  No V1 pass/fail or MeshSplatting improvement is
claimed before the complete frozen artifacts exist.
