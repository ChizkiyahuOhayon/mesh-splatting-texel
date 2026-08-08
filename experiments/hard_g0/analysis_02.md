# HARD-G0 result: FAIL — early endpoint dwell severely regresses Garden

Date: 2026-08-08

Protocol: `experiments/hard_g0/protocol.md`, committed before execution

Decision: `decision_02.json`

Server records: `hard_g0_garden_{stock,early25000}_02`

Source revision: `83db7d80787effe2ef1ecbb024a1813893dce2ea`

The JSON records in this directory were transcribed verbatim from the immutable
server artifacts printed after the run. The canonical artifacts remain under
`/home/smbu/dy/nas/meshsplatting_smbu`.

## Validity

Both arms used Garden, seed 0, iteration 30,000, 24 held-out views, the same
source revision, physical GPU 1 (NVIDIA A40), PyTorch 2.7.1+cu126, and the same
protocol. Both reached the required endpoint:

| arm | final sigma | endpoint valid |
|---|---:|---:|
| stock | `9.99999999999889e-05` | yes |
| early25000 | `9.99999999999889e-05` | yes |

Stock reaches `24.7467 dB` at scaling 4, within `0.01 dB` of the established
Garden stock baseline around `24.74 dB`. The reproduction anchor is valid.

## Locked result at scaling 4

| arm | PSNR | SSIM | LPIPS-VGG |
|---|---:|---:|---:|
| stock | 24.7467 | 0.7488 | 0.2478 |
| early25000 | 23.9330 | 0.6995 | 0.2907 |
| early - stock | **-0.8136** | **-0.0493** | **+0.0430** |

The preregistered screen required `early - stock >= +0.10 dB` PSNR and
non-worse LPIPS. Both checks fail, while both identity and endpoint checks pass.
The locked decision is therefore **FAIL**.

This is not an ambiguous null: early hardening is worse on all three quality
metrics, and the PSNR regression is over fifty times Garden's previously
measured seed standard deviation (`0.0157 dB`). The one-scene screen does not
support a universal claim about hardening schedules, but it decisively rejects
the only claim it was designed to authorize.

## Direction consequence

Per `RESEARCH_PLAN_v16.md`, HARD-G0 failure does not authorize Gate A1, the
shared-clock stationarity certificate, or the certificate-gated local bridge.
No threshold repair, delayed-fast arm, or renamed schedule variant is allowed.
The next positive-method review must change the representation-level
formulation while retaining the CVPR-scale full-benchmark target; this result
is not a negative-paper pivot.

## Infrastructure incident and archival status

Attempt `_01` is excluded: it stopped at iteration 11,000 when physical GPU 0
was shared with a 27,096 MiB MapLoc process. It produced no evaluation or
decision and is an infrastructure failure, not a scientific run.

The `_02` gate decision is final, but the local archive is not yet sealed. The
following canonical server evidence still needs to be copied or summarized:

- both complete `results.json` files, including scaling 2, primitive counts,
  render time, wall time, and peak memory;
- both `DONE` sentinels;
- the stock and early per-1,000-iteration train/test traces;
- a no-Traceback/no-OOM check over both training logs;
- artifact SHA-256 checksums.

None of these missing ancillary fields can change the locked FAIL, but they are
required before marking the experiment archive complete.
