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

