# GoRFE-Q0 attempt 04 — PASS

Date: 2026-08-11

Source revision: `6f7b03d6f9a72d75ea143bfd00f23d66e5a3f961`

Artifact directory: `${NAS_ROOT}/experiments/gorfe_q0_04`

Environment: one physical NVIDIA A40 (`CUDA_VISIBLE_DEVICES=3`), PyTorch
`2.7.1+cu126`, CUDA `12.6`.

## Decision

**PASS.** All thirteen locked Boolean checks are true. The native extension
built successfully, all 200 CPU tests passed, the CUDA result and immutable
manifest both report `pass`, and `DONE` plus the SHA-256 ledger exist.

## Numerical evidence

- active zero detail is bitwise parent, with identical SHA-256
  `5f322709cde086cb408aecee345fd108fdfb4d14979c5b0980a05e0f05caa2c0`;
- legacy `[E,3]` DC and combined `[E,4,3]` DC are bitwise equal, with SHA-256
  `8ae8c93a0e3ee16fcc8a5f4aa8518445e5e281808f4e538afc9cfbb00ab9e40f`;
- both camera-dependent renders differ, with maximum camera-direction change
  `0.01111615`;
- coefficient-gradient maximum relative errors are `0.00235837` and
  `0.00077700`, below the locked `0.005` limit;
- the active fixed-support vertex derivative is analytic `-0.71107447` versus
  finite difference `-0.71167946`, relative error `0.00060499`;
- the zero-detail parent control is analytic `-0.71040964` versus finite
  difference `-0.71108341`, relative error `0.00067377`;
- both vertex errors are below the locked `0.05` limit;
- the shared-edge orientation/continuity check passes.

The non-decision full-image diagnostic remains mismatched because the
perturbation changes exactly one parent-support pixel: analytic `-25.5611`
versus finite difference `-403.3089`. This confirms the preregistered reason for
using fixed visibility support and is not evidence against the local backward.

## Artifact integrity

| artifact | SHA-256 |
|---|---|
| `build.log` | `f8b8a39ea59b396567d404de9cadfdb9f516c4838f45ce78e6a96993de58de32` |
| native extension | `95dffe8cfe53d1442c7c8d3d9d2a371db61db78f2b11775df8b9d5c7f0abb81d` |
| `tests.log` | `5c5255a81560f53a312b8466cb20b3d9f83a3fb01135199fc219ec547787dd1e` |
| `result.json` | `5c44210c01084a175f2f459bfbf2272aebcb310c88f079ad5b989cf9f8fcd99c` |
| `smoke.log` | `5c44210c01084a175f2f459bfbf2272aebcb310c88f079ad5b989cf9f8fcd99c` |
| `manifest.json` | `77edd4ac90ca270f992f366ff35590e66df47499f0f59ce111c9b33c4f6171b9` |
| `DONE` | `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1` |

The manifest records result hash
`5c44210c01084a175f2f459bfbf2272aebcb310c88f079ad5b989cf9f8fcd99c`,
extension hash
`95dffe8cfe53d1442c7c8d3d9d2a371db61db78f2b11775df8b9d5c7f0abb81d`,
and the exact source revision above.

## Claim boundary and authorization

Q0 supports native parent preservation, legacy compatibility, shared-edge
continuity, camera dependence, and coefficient/fixed-support vertex-gradient
correctness for the P2-DC/P2-SH1 carrier. It does not establish Garden/Room
transfer, final image quality, efficiency, or novelty by itself.

This pass authorizes V0 implementation and V0/V1 protocol freezing. It does not
authorize reading Garden/Room scores until V1's executable protocol is committed,
and it does not authorize physical training before V1 passes.
