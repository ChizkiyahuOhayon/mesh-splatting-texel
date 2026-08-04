# SAC-G1 server runbook

Twelve trainings — two scenes, two arms, three shared seeds — then twelve
evaluations and one decision. About ten hours of wall clock across two GPUs.

Three parameters carry the design, and each defaults to the published
behaviour, so the `stock` arm is the unmodified pipeline:

- `--seed` (default 0): `safe_state` previously hard-coded 0, so this pipeline
  has never produced seed variance.
- `--final_scaling` (default 4): the last training phase's supersampling.
- `--cleanup_scaling` (default 0 = keep the training factor): the factor used
  by the post-training pruning pass. Pinning it to 4 in both arms removes the
  confound that produced SAC-G0's 18% primitive-count difference.

## Train

Room needs `--indoor`; Garden takes no scene flag. Run two at a time, one per
GPU, and keep both GPUs on the same scene so the pair finishes together.

```bash
run_arm () {  # scene data_path indoor_flag arm seed gpu
  local SCENE=$1 DATA=$2 FLAG=$3 ARM=$4 SEED=$5 GPU=$6
  local OUT="$NAS_ROOT/experiments/sac_g1_${SCENE}_${ARM}_seed${SEED}_train"
  local EXTRA=""
  [ "$ARM" = "splat2" ] && EXTRA="--final_scaling 2 --cleanup_scaling 4"
  [ "$ARM" = "stock" ]  && EXTRA="--cleanup_scaling 4"
  test -e "$OUT" && { echo "SKIP existing $OUT"; return; }
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=$GPU nohup /usr/bin/time -f "SECONDS %e" \
  python train.py -s "$DATA" -m "$OUT" --eval --seed $SEED $FLAG $EXTRA \
    > "$NAS_ROOT/logs/sac_g1_${SCENE}_${ARM}_seed${SEED}.log" 2>&1 &
}

mkdir -p "$NAS_ROOT/logs"
for SEED in 0 1 2; do
  run_arm garden "$GARDEN_DATA" ""         stock  $SEED 0
  run_arm garden "$GARDEN_DATA" ""         splat2 $SEED 1
  wait
  run_arm room   "$ROOM_DATA"   "--indoor" stock  $SEED 0
  run_arm room   "$ROOM_DATA"   "--indoor" splat2 $SEED 1
  wait
done
```

Both arms pass `--cleanup_scaling 4`. For `stock` that is a no-op in effect,
since its training already ends at 4, but stating it makes the two commands
differ in exactly the sampling schedule and nothing else.

## Evaluate

Run these **serially**: the render timings are part of the record and must not
contend for a GPU.

```bash
for SCENE in garden room; do
  case $SCENE in
    garden) DATA="$GARDEN_DATA";;
    room)   DATA="$ROOM_DATA";;
  esac
  for ARM in stock splat2; do
    for SEED in 0 1 2; do
      MODEL="$NAS_ROOT/experiments/sac_g1_${SCENE}_${ARM}_seed${SEED}_train"
      OUT="$NAS_ROOT/experiments/sac_g1_${SCENE}_${ARM}_seed${SEED}_eval"
      test -e "$OUT" && { echo "SKIP existing $OUT"; continue; }
      TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
      TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
      SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
      python sac_eval.py -s "$DATA" -m "$MODEL" --eval \
        --sac_arm $ARM --sac_scene $SCENE --sac_seed $SEED --sac_output "$OUT"
    done
  done
done
```

## Locked decision

```bash
python sac_g1_decide.py "$NAS_ROOT"/experiments/sac_g1_*_eval
```

The rule is in `experiments/sac_g1/protocol.md`. The effect must be positive on
both scenes and exceed twice the standard error of the paired differences, with
no perceptual regression, and each scene's stock arm must reproduce its recorded
baseline within 0.15 dB. A pass authorises the nine-scene run; a failure closes
this axis.
