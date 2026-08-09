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

## Complete evaluation record

| arm | scaling | PSNR | SSIM | LPIPS-VGG | render ms/view |
|---|---:|---:|---:|---:|---:|
| stock | 2 | 24.4526 | 0.7345 | 0.2588 | 46.5402 |
| stock | 4 | 24.7467 | 0.7488 | 0.2478 | 71.2425 |
| early25000 | 2 | 23.7086 | 0.6866 | 0.3007 | 47.4790 |
| early25000 | 4 | 23.9330 | 0.6995 | 0.2907 | 72.1557 |

At scaling 2, early minus stock is `-0.7440 dB` PSNR, `-0.0479` SSIM,
and `+0.0419` LPIPS-VGG. The conclusion is therefore not specific to the
locked scaling-4 cell.

| arm | vertices | triangles | training seconds | peak allocated bytes |
|---|---:|---:|---:|---:|
| stock | 3,245,174 | 6,937,533 | 6,097.57 | 9,488,602,624 |
| early25000 | 3,403,425 | 7,055,075 | 5,929.68 | 9,580,032,512 |
| early - stock | +158,251 (+4.88%) | +117,542 (+1.69%) | -167.89 (-2.75%) | +91,429,888 (+0.96%) |

The early arm is slightly larger and slightly slower to render: `+2.02%` at
scaling 2 and `+1.28%` at scaling 4. Its quality loss cannot be attributed to
having fewer primitives or a smaller memory budget.

## Late training trajectory

The values below are the built-in held-out evaluations in the training logs.
They are diagnostic only; the preregistered decision uses the separate
scaling-4 evaluation above.

| iter | stock PSNR | early25000 PSNR |
|---:|---:|---:|
| 20,000 | 24.8193 | 24.6827 |
| 21,000 | **25.3101** | **25.0907** |
| 22,000 | 25.2799 | 25.0236 |
| 23,000 | 25.2214 | 24.9083 |
| 24,000 | 25.1484 | 24.7453 |
| 25,000 | 25.0925 | 24.6389 |
| 26,000 | 25.1261 | 24.5355 |
| 27,000 | 25.0894 | 24.3840 |
| 28,000 | 25.0077 | 24.2295 |
| 29,000 | 24.9095 | 24.0878 |
| 30,000 | 24.7522 | 23.9491 |

Both arms peak at 21,000 and then deteriorate, but the decline to 30,000 is
`-0.5579 dB` for stock and `-1.1417 dB` for early25000. This supports the
narrow diagnosis that earlier hard-regime exposure amplifies the existing
late-training degradation. It does not authorize checkpoint selection or a
new schedule arm under this gate.

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

The `_02` gate decision is final. Both `DONE` sentinels and both complete
`results.json` files exist. A grep over both training logs returned no
Traceback, OOM, or abnormal-command-exit marker. The canonical server artifacts
have the following verified SHA-256 values:

| artifact | SHA-256 |
|---|---|
| stock results | `df303afca3a0ae0bfdc22098985094ba476236e301f7248ca172ad3e1781eed4` |
| stock manifest | `8e99666d103355540f2bfcd20b7565f3f995ecc6326ead895a114f199def5da9` |
| stock DONE | `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1` |
| early results | `6ff60b838ea9a161335e545245e8894d1efff063e417bfd8b487e5d4accdae72` |
| early manifest | `3ab5bae83ce61bfd4115233f695a7a945aeb078ec2e7a2edfeae2b4c389ea442` |
| early DONE | `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1` |
| stock training log | `3c3e6aa0ea596943e78aa1d961a7506e40dc46c90d83a4b12a683aea72da16f0` |
| early training log | `aab5dbca10c4c38a45807b44ce7036357b471b6e4f6c208fdd4c070aa6d81b0d` |
| decision | `f8a08cd594a58cd09672658fff773c930e8c451ef91ebe9db34a040cc866ebc3` |

The first compact-summary command supplied to the server was split across
lines and raised `IndentationError`; this was a post-run reporting-command
error. The same fields were then printed with `jq` and are recorded above.

**Archive status: sealed.** The canonical full records and logs remain on the
NAS; this repository stores the immutable decision, both manifests, the full
scientific summary, and hashes binding them to those server artifacts.
