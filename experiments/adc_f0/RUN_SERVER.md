# ADC-F0 server runbook

No training. One existing Garden checkpoint is read, about 640 renders are taken, and
a reading is printed — roughly three minutes on one GPU.

Reuse the SAC-G1 `stock` seed-0 checkpoint: it was trained on the unmodified pipeline
and validated to reproduce `garden_baseline` to `0.010 dB`, so any ratio measured on
it is a property of the published rasterizer and not of an experimental arm.

## Run

```bash
cd /home/smbu/dy/mesh-splatting-texel
git fetch origin main && git checkout main && git pull

# The SAC-G1 stock seed-0 model. Confirm the path before running:
#   ls -d "$NAS_ROOT"/experiments/sac_g1_garden_stock_seed0_train_01
ADC_MODEL="$NAS_ROOT/experiments/sac_g1_garden_stock_seed0_train_01"
ADC_OUT="$NAS_ROOT/experiments/adc_f0_garden_01"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=0 \
python adc_forensics.py -s "$GARDEN_DATA" -m "$ADC_MODEL" --eval \
  --adc_scene garden --adc_output "$ADC_OUT" \
  2>&1 | tee "$NAS_ROOT/logs/adc_f0_garden.log"
```

The script prints the reading itself; nothing further needs running. Paste the tail of
that log.

## What the output means

```
ADC-F0 reading: HOMOGENEOUS | HETEROGENEOUS | VIEW_DEPENDENT | INCONCLUSIVE
  <view>: median ratio 8.4xxx
    kept 80/80  deterministic=True
      projected_size  spread 1.0xx  rho +0.0xx  [8.4, 8.4, 8.4, 8.4, 8.4]
                depth  spread ...
        max_blending  spread ...
```

- `median ratio` should land near the `8.44` already recorded in
  `experiments/rits_t0/results/g0_diag_garden_01.md`. It will not match exactly — that
  measurement used `L1 + SSIM` on one view and this uses squared error — but a wildly
  different value means the probe, not the rasterizer, needs explaining.
- `deterministic=True` and a high `kept` fraction are the validity guards. If either
  fails the reading is `INCONCLUSIVE` by rule and no interpretation is offered.
- The five bracketed numbers are the per-quintile median ratios. Flat across all three
  covariates is `HOMOGENEOUS`; a trend is what would make gradient-based densification
  unsafe.

## Write-once

`reserve_output_directory` refuses an existing path. A rerun uses a new suffix
(`adc_f0_garden_02`) rather than deleting the first — the project keeps failed and
superseded runs.

## Then

Either reading routes the next gate, which is why this one runs first:

- **HOMOGENEOUS** — the under-report is a global constant, Adam absorbs it, and the
  AbsGS-style gradient criterion stays available as an ADC-G0 arm.
- **HETEROGENEOUS** — the published backward carries a per-primitive bias. ADC-G0 then
  uses the error criterion only, and the bias becomes a defect worth reporting and
  repairing on its own.

Either way ADC-G0 leads with the error criterion, because it needs no rasterizer
change and cannot inherit a backward defect.
