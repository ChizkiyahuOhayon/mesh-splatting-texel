# Response to code review v1 (2026-07-24)

External review: `comments/v1/mesh_splatting_texel_remediation_guide.pdf`. Reviewer
disclaimed compiling/running; every claim was re-verified against the actual code before
acting. Note: `verify_texel.py` **passes on the A40** (bit-exact zero-texel no-op;
gradient matches finite differences to 7 significant figures), so some theoretical
concerns are already empirically bounded.

## Verified and FIXED

| # | Finding | Verdict | Fix |
|---|---|---|---|
| P0-3 | `_prune_vertices` drops triangles (`valid_tris`) but never synced `_texels` → misaligned texels in the saved model | **REAL**. Only affects the *saved* model (runs it via the final cleanup); in-training metrics are measured before it, so E3-B/E4/E5 numbers stand. | `_prune_vertices` now applies the same face mask to texels via `_prune_texels`; added `validate_face_state()` called after every prune; `_prune_texels` asserts mask shape/dtype. |
| P0-4 | `set -eu` + `python … | tee` returns tee's status, so a crashed run still `touch DONE` | **REAL** (record integrity). | Scripts now `set -euo pipefail`; `DONE` written only if training exits 0 **and** the final checkpoint + `ITER 30000` eval line exist, else `FAILED`. |
| P1-1 | `texel_tv` is within-face variance, not total variation | **REAL** naming. | Kept the arg name for E5 reproducibility; comment/paper now state "within-face variance", not TV. |
| P1-3 | "order=0 exactly the original path" over-claims (kernel ABI still differs) | **REAL** wording. | Doc softened to "output semantics preserved; bit/perf parity is a test claim". |
| P1-4 | `capture()/restore()` (used by `--checkpoint_iterations` resume) dropped texels | **REAL**. | `capture/restore` now round-trip `texel_order`, `_texels` and the texel optimizer state; backward-compatible with 4-tuple checkpoints. |
| P1-5 | No invariant checks at the Python/C++ boundary | **REAL** (defensive). | `TORCH_CHECK` on texel dim/shape/contiguity/dtype in the forward binding (guarded by `texel_order>0`). |
| P1-7 | Texel init tied to a specific control-flow point; short/resumed runs could silently train a baseline under `--texel_order>0` | **REAL** robustness. | End-of-training assertion: requested `texel_order>0` with an uninitialised carrier now raises. |
| P0-5 (partial) | `verify_texel.py` coverage too thin for "safe to train" | **PARTLY**. Slot indexing *is* unit-tested on the Python mirror (`maclab/tests/test_ptex.py`); the kernel mirrors it. | Added T4 (topology-mutation sync regression) and T5 (malformed-carrier rejection) to `verify_texel.py`. Full pytest/CUDA suite = future. |

## Verified, PARTLY valid — reviewer over-weighted severity

- **P0-1 (perspective-correct slot).** The reviewer called screen-space barycentric slot
  selection a release blocker. Re-verified: the **baseline vertex-colour interpolation is
  itself deliberately screen-affine** (`noperspective` in the GL renderer; paper §3
  footnote 1), so the texel *value* must stay screen-affine to remain a consistent
  residual. The real sub-point — a fixed surface point can map to different *slots*
  across views under perspective — is genuine but **sub-pixel here**: ~11M triangles,
  each a few pixels, so within-triangle barycentric distortion is negligible, which is
  why the method trains cleanly and gains replicate. Documented as a known approximation;
  a perspective-correct slot (via per-vertex inverse-w) is a planned refinement for
  coarser meshes / video, requiring its own validation run. **Not** a blocker; existing
  results valid.

## Accepted, DEFERRED (honest scope, not silently ignored)

- **P0-2 (export loses texels).** Real gap between the "exports to a plain textured mesh"
  framing and `create_ply.py`/`export_web.py`, which write vertex colours only. The
  training checkpoint round-trips texels fully; the *exporter* does not yet. Scoped
  explicitly in `SETUP_SERVER.md` as the Gate-3 deliverable (per-face texel → atlas).
- **P1-2 (nearest → aliasing).** Deliberate, documented choice; a filtered/piecewise-
  linear CUDA mode is a planned methods extension (the linear variant already exists in
  the `maclab` prototype).
- **P1-6 (memory budget).** Minor: add an estimate/print in `create_texels`. Low priority.
- **P2-x (deps pinning, CI, repo hygiene), §6 (FaceAppearanceState refactor).** Sensible
  for the camera-ready release; scheduled after the numbers are frozen. The §6 refactor
  is good design but a large change; the P0-3 fix already routes face-dropping through a
  texel-syncing path, which is the safety-critical part.
