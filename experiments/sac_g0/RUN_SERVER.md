# SAC-G0 server runbook

Run from the source checkout containing `sac_eval.py`. The rasterizer is
unchanged, so no extension rebuild is needed. Output paths are reserved
write-once; a retry uses the next suffix.

Two full Garden trainings (about 1.5 h each) followed by two fast evaluations.
The arms differ in exactly one value: `--final_scaling`, whose default is the
published `4`, so the stock arm is the unmodified pipeline.

## Train both arms

Record the wall-clock of each run; it is passed to the evaluator for the record.

```bash
SAC_STOCK_MODEL="$NAS_ROOT/experiments/sac_g0_garden_stock_train_01"
SAC_SPLAT2_MODEL="$NAS_ROOT/experiments/sac_g0_garden_splat2_train_01"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
/usr/bin/time -f "stock training seconds %e" \
python train.py -s "$GARDEN_DATA" -m "$SAC_STOCK_MODEL" --eval

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
/usr/bin/time -f "splat2 training seconds %e" \
python train.py -s "$GARDEN_DATA" -m "$SAC_SPLAT2_MODEL" --eval --final_scaling 2
```

If `train.py` needs the scene-type flag used for the existing Garden baseline
(outdoor), pass exactly the same flags to **both** arms; the arms must differ
only in `--final_scaling`.

## Evaluate each arm at both rates

```bash
SAC_STOCK="$NAS_ROOT/experiments/sac_g0_garden_stock_eval_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python sac_eval.py -s "$GARDEN_DATA" -m "$SAC_STOCK_MODEL" --eval \
  --sac_arm stock --sac_scene garden --sac_output "$SAC_STOCK" \
  --sac_training_seconds <stock seconds>

SAC_SPLAT2="$NAS_ROOT/experiments/sac_g0_garden_splat2_eval_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python sac_eval.py -s "$GARDEN_DATA" -m "$SAC_SPLAT2_MODEL" --eval \
  --sac_arm splat2 --sac_scene garden --sac_output "$SAC_SPLAT2" \
  --sac_training_seconds <splat2 seconds>
```

## Locked decision

```bash
python sac_decide.py --stock "$SAC_STOCK" --splat2 "$SAC_SPLAT2"
```

The rule is in `experiments/sac_g0/protocol.md`. All three conditions must
hold: the stock arm reproduces the recorded `garden_baseline` quality, the
`splat2@2` cell costs at most 0.35 dB and 0.020 LPIPS against `stock@4`, and
rendering at `scaling 2` is measured at least 2.5x faster than at `scaling 4`.

Read the validity condition first. If the stock arm does not reproduce the
baseline, the training platform is wrong and the comparison is not read — the
failure mode that voided RITS-T0.
