# SVSR-G1 analysis

## Garden smoke 02 — 2026-07-30

Classification: **EXPLORATORY POSITIVE-CONTROL SMOKE** (`svsr_max_views=1`). It
cannot pass the three-scene gate.

The first held-out view, `DSC07956`, completed for the SH-only checkpoint, unchanged
fixed-texel checkpoint, and the same texel checkpoint with the preregistered
projected-footprint filter.

| Variant | PSNR | SSIM | LPIPS-VGG | ΔPSNR vs fixed | ΔLPIPS vs fixed |
|---|---:|---:|---:|---:|---:|
| SH | 21.4940 | 0.7051 | 0.2705 | −0.1100 | +0.0405 |
| fixed | 21.6040 | 0.7307 | 0.2300 | — | — |
| footprint | 21.6103 | 0.7300 | 0.2305 | +0.0063 | +0.0006 |

Sanity and preregistered positive-control interpretation:

1. The SH metrics exactly reproduce the earlier G0 Garden reference on this view.
2. Fixed texels improve PSNR by 0.1100 dB and LPIPS by 0.04051 over SH.
3. The footprint variant improves PSNR by 0.1163 dB and LPIPS by 0.03993 over SH,
   retaining **105.8%** of the fixed PSNR gain and **98.6%** of the LPIPS gain. Both
   exceed the locked 70% Garden retention threshold on this exploratory view.
4. This is not a no-op: 37.81% of 1,364,470 rendered faces receive a partial-detail
   weight, and the 10th-percentile weight is 0.393. The median weight is 1.0.
5. The direct footprint-versus-fixed changes are small and mixed (+0.0063 dB PSNR,
   −0.00075 SSIM, +0.00057 LPIPS), which is acceptable for a positive control but is
   not evidence of an overall quality gain.

Decision: **continue unchanged to full-test Garden**. Do not tune the footprint
formula. Room and Stump remain the decisive negative controls for the mechanism.

## Garden full 01 — 2026-07-30

Classification: **CONFIRMATORY POSITIVE-CONTROL COMPONENT** (`svsr_max_views=0`).
This resolves the Garden condition but cannot decide G1 without Room and Stump.

| Variant | PSNR | SSIM | LPIPS-VGG | ΔPSNR vs fixed | ΔLPIPS vs fixed |
|---|---:|---:|---:|---:|---:|
| SH | 24.6951 | 0.7481 | 0.2490 | −0.3260 | +0.0406 |
| fixed | 25.0210 | 0.7739 | 0.2084 | — | — |
| footprint | 25.0242 | 0.7732 | 0.2089 | +0.0032 | +0.0005 |

Relative to SH, fixed texels gain 0.3260 dB and reduce LPIPS by 0.04061. The
footprint variant gains 0.3291 dB and reduces LPIPS by 0.04010, retaining **101.0%**
of the PSNR gain and **98.7%** of the LPIPS gain. It also retains 97.4% of the SSIM
gain. These exceed the locked 70% Garden thresholds.

Garden therefore **passes its preregistered positive-control condition**. The small
mixed footprint-versus-fixed deltas are not claimed as a new quality improvement.
Proceed unchanged to full Room and Stump; do not tune on Garden.

