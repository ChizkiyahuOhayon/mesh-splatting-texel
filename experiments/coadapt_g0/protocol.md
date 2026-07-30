# COADAPT-G0: frozen-checkpoint carrier decomposition

Status: **PREREGISTERED — no COADAPT-G0 output has been observed**
Date: 2026-07-30

## Question

Did joint optimization change the Room texel checkpoint's geometry/SH base so that
the fixed-texel regression cannot be repaired by a test-time detail filter?

## Fixed input and variants

Use the exact Room SH-only and order-2 texel checkpoints from the completed
nine-scene experiment and the full held-out test split. Compare:

1. `sh_reference`: the independently trained SH-only checkpoint;
2. `fixed`: the unchanged texel checkpoint;
3. `zero`: the same texel checkpoint with its residual carrier set to zero;
4. `face_mean`: the same texel checkpoint with every cell replaced by that face's
   mean residual.

Only the in-memory texel tensor changes for variants 3 and 4. Restore the original
tensor before exit. Do not train, tune a threshold, change cameras, or change the
renderer. Write checkpoint hashes, test views, image sizes, package/GPU information,
and source revision before rendering.

## Metrics and locked decision

Report full-test PSNR, SSIM, and LPIPS-VGG. For PSNR and LPIPS, define recovery as
the fraction of the `fixed` regression relative to `sh_reference` removed by a
candidate. This gate is valid only when `fixed` regresses on both metrics, as already
observed in the confirmatory Room result.

- **PATH CO-ADAPTATION SUPPORTED:** `zero` recovers less than 25% in both PSNR and
  LPIPS. Proceed to one frozen-base, texel-only training pilot.
- **APPEARANCE HARM SUPPORTED:** `zero` recovers at least 75% in both metrics. Stop
  the texel route; freezing the base does not address the observed carrier harm.
- **MIXED:** all other outcomes. Permit one correctness audit and one unchanged
  rerun, but no threshold or per-scene tuning.

`face_mean` localizes any carrier effect into per-face DC versus within-face detail;
it does not alter the primary decision. This diagnostic cannot support an Oral-level
claim by itself.
