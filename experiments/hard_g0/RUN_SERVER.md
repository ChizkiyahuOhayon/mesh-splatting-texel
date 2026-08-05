# HARD-G0 server runbook

Two Garden trainings and two evaluations; about three and a half hours if the
arms share a GPU each, under two hours in parallel. No code change carries this
gate — `sigma_until` is already an exposed parameter, and the schedule clamps
after it, so both arms finish at `sigma = 1e-4`.

## Train

```bash
HARD_STOCK="$NAS_ROOT/experiments/hard_g0_garden_stock_train_01"
HARD_EARLY="$NAS_ROOT/experiments/hard_g0_garden_early_train_01"
mkdir -p "$NAS_ROOT/logs"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -f "SECONDS %e" \
python train.py -s "$GARDEN_DATA" -m "$HARD_STOCK" --eval --seed 0 \
  --cleanup_scaling 4 \
  > "$NAS_ROOT/logs/hard_g0_stock.log" 2>&1 &

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -f "SECONDS %e" \
python train.py -s "$GARDEN_DATA" -m "$HARD_EARLY" --eval --seed 0 \
  --cleanup_scaling 4 --sigma_until 25000 \
  > "$NAS_ROOT/logs/hard_g0_early.log" 2>&1 &

wait
grep -h SECONDS "$NAS_ROOT"/logs/hard_g0_*.log
```

Both arms pass `--cleanup_scaling 4` so the final pruning criterion is the one
SAC-G1 established as schedule-independent.

## Evaluate

Serially, so the render timings do not contend.

```bash
SOURCE_REVISION=$(git rev-parse HEAD)
for ARM in stock early; do
  case $ARM in
    stock) MODEL="$HARD_STOCK";;
    early) MODEL="$HARD_EARLY";;
  esac
  OUT="$NAS_ROOT/experiments/hard_g0_garden_${ARM}_eval_01"
  test -e "$OUT" && { echo "SKIP $OUT"; continue; }
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
  SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
  python sac_eval.py -s "$GARDEN_DATA" -m "$MODEL" --eval \
    --sac_arm $ARM --sac_scene garden --sac_seed 0 --sac_output "$OUT" \
    > "$NAS_ROOT/logs/hard_g0_eval_${ARM}.log" 2>&1
done
```

## Read the screen

```bash
for ARM in stock early; do
  D="$NAS_ROOT/experiments/hard_g0_garden_${ARM}_eval_01"
  python -c "
import json; r = json.load(open('$D/results.json'))
c4, c2 = r['cells']['scaling_4'], r['cells']['scaling_2']
print('$ARM  sigma', r['sigma'], ' tris', r['primitives']['triangles'])
print('   @4 psnr %.4f ssim %.4f lpips %.4f' % (c4['psnr'], c4['ssim'], c4['lpips_vgg']))
print('   @2 psnr %.4f ssim %.4f lpips %.4f' % (c2['psnr'], c2['ssim'], c2['lpips_vgg']))"
done
```

Also worth pulling, since the decline is what motivated the gate:

```bash
grep "Evaluating test" "$NAS_ROOT"/logs/hard_g0_early.log | tail -8
```

The screen passes only if both arms report `sigma = 1e-4`, `early@4` beats
`stock@4` by at least `0.10 dB`, and `early`'s LPIPS at `scaling 4` is not
worse. The rule is in `experiments/hard_g0/protocol.md`; one scene and one seed
cannot decide, only screen.
