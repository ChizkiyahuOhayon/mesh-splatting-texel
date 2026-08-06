# ADC-G0 server runbook

Three Garden trainings and three evaluations. Two GPUs run two arms at a time, so
about three and a half hours end to end.

`stock` installs nothing — `--adc_arm stock` is the default and runs the published
densification path unchanged, so it doubles as the reproduction anchor.

## Train

```bash
cd /home/smbu/dy/mesh-splatting-texel
git fetch origin main && git checkout main && git pull
mkdir -p "$NAS_ROOT/logs"

run_arm () {   # run_arm <gpu> <arm>
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=$1 nohup /usr/bin/time -f "SECONDS %e" \
  python train.py -s "$GARDEN_DATA" \
    -m "$NAS_ROOT/experiments/adc_g0_garden_$2_train_01" --eval --seed 0 \
    --cleanup_scaling 4 --adc_arm $2 \
    > "$NAS_ROOT/logs/adc_g0_$2.log" 2>&1 &
}

run_arm 0 stock
run_arm 1 rng
wait

run_arm 0 multiplicity
wait

grep -h SECONDS "$NAS_ROOT"/logs/adc_g0_*.log
```

Every arm passes `--cleanup_scaling 4` so the final pruning criterion is the
schedule-independent one SAC-G1 established.

## Manipulation check — read this before the metrics

```bash
python -c "
import json
for arm in ('rng', 'multiplicity'):
    path = '$NAS_ROOT/experiments/adc_g0_garden_%s_train_01/adc_rounds.json' % arm
    try: rounds = json.load(open(path))['rounds']
    except FileNotFoundError: print(arm, 'no rounds file (expected for rng)'); continue
    deep = sum(r['depth_2'] for r in rounds)
    print('%s: %d rounds, %d depth-2 faces total, budget %d spent %d' % (
        arm, len(rounds), deep,
        sum(r['budget'] for r in rounds), sum(r['spent'] for r in rounds)))
"
```

If `multiplicity` reports almost no depth-2 faces then it concentrated nothing and a
null result on the metrics says nothing about the hypothesis — report that rather
than a failure. `rng` writes no rounds file; it changes only the generator.

## Evaluate

Serially, so the render timings do not contend.

```bash
SOURCE_REVISION=$(git rev-parse HEAD)
for ARM in stock rng multiplicity; do
  OUT="$NAS_ROOT/experiments/adc_g0_garden_${ARM}_eval_01"
  test -e "$OUT" && { echo "SKIP $OUT"; continue; }
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
  SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
  python sac_eval.py -s "$GARDEN_DATA" \
    -m "$NAS_ROOT/experiments/adc_g0_garden_${ARM}_train_01" --eval \
    --sac_arm $ARM --sac_scene garden --sac_seed 0 --sac_output "$OUT" \
    > "$NAS_ROOT/logs/adc_g0_eval_${ARM}.log" 2>&1
done
```

## Read the screen

```bash
for ARM in stock rng multiplicity; do
  python -c "
import json
r = json.load(open('$NAS_ROOT/experiments/adc_g0_garden_${ARM}_eval_01/results.json'))
c4, c2 = r['cells']['scaling_4'], r['cells']['scaling_2']
print('${ARM}  tris %d  sigma %g' % (r['primitives']['triangles'], r['sigma']))
print('   @4 psnr %.4f ssim %.4f lpips %.4f' % (c4['psnr'], c4['ssim'], c4['lpips_vgg']))
print('   @2 psnr %.4f ssim %.4f lpips %.4f' % (c2['psnr'], c2['ssim'], c2['lpips_vgg']))"
done
```

The locked rule is in `experiments/adc_g0/protocol.md`. In short:

- **validity** — `stock@4` within `0.10 dB / 0.010 / 0.010` of `24.7372 / 0.7484 / 0.2480`,
  and `rng` and `multiplicity` triangle counts within `2%` of `stock`'s;
- **screen** — `multiplicity@4` beats **`rng@4`** (never `stock@4`) by `>= 0.15 dB`
  with LPIPS no worse.

Report `rng - stock` either way; the RNG defect is worth knowing about regardless of
how the multiplicity arm lands.

## Write-once

`reserve_output_directory` refuses an existing path. Retries use a new suffix
(`_02`) rather than deleting the first.
