# APX-F0 server runbook

No training. Two existing checkpoints are read and four render passes are taken over
each — about five minutes per scene.

Reuse the SAC-G1 `stock` seed-0 models: both were trained on the unmodified pipeline
and validated against the recorded baselines, so what is measured is a property of
MeshSplatting rather than of an experimental arm.

## Run

```bash
cd ~/dy/mesh-splatting-texel
git pull && git log --oneline -1     # must show the APX-F0 commit
ls apx_f0_eval.py apx_cells.py apx_f0_decide.py

export NAS_ROOT GARDEN_DATA ROOM_DATA
```

Confirm the model paths before running:

```bash
ls -d "$NAS_ROOT"/experiments/sac_g1_garden_stock_seed0_train_01 \
      "$NAS_ROOT"/experiments/sac_g1_room_stock_seed0_train_01
```

Then, one scene at a time:

```bash
for SCENE in garden room; do
  case $SCENE in
    garden) DATA="$GARDEN_DATA";;
    room)   DATA="$ROOM_DATA";;
  esac
  OUT="$NAS_ROOT/experiments/apx_f0_${SCENE}_01"
  test -e "$OUT" && { echo "SKIP $OUT"; continue; }
  echo "=== $SCENE ==="
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
  SVSR_SOURCE_REVISION=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=1 \
  python apx_f0_eval.py -s "$DATA" \
    -m "$NAS_ROOT/experiments/sac_g1_${SCENE}_stock_seed0_train_01" --eval \
    --apx_scene $SCENE --apx_output "$OUT" \
    2>&1 | tee "$NAS_ROOT/logs/apx_f0_${SCENE}.log"
done
```

The script prints its own reading per scene. Paste both.

## What the output means

```
APX-F0 garden
  eligible faces N / M  (x.xx%)
  ceiling order 1: +0.xxxx dB
  ceiling order 2: +0.xxxx dB
  ceiling order 4: +0.xxxx dB
  concentration top10% 0.xxxx
          residual_mass  capture 0.xxxx  lift x.xxx
  ...
          ceiling: PASS|FAIL
    concentration: PASS|FAIL
   predictability: PASS|FAIL
```

- **eligible faces** is a headline. Garden carries 6.9M triangles over roughly 1M
  pixels, so most faces are sub-pixel and cannot hold a texel grid at all. A small
  percentage bounds the whole direction no matter how the other numbers land, and it
  should be read before anything else.
- **ceiling order 4** must reach `0.30 dB` — what uniform texels already achieved on
  Garden. Below that there is nothing left for adaptive allocation to win.
- **concentration** is the condition the thesis rests on. Uniform texels already
  failed the 9-scene mean at `-0.103 dB`; if gain is spread evenly, adaptive cannot
  do better than uniform and the idea is dead regardless of signal quality.
- **predictability** uses XVR-G0's thresholds verbatim (`lift >= 1.75`, `10%` over
  the best non-residual control), so a pass here against XVR-G0's fail is a statement
  about appearance and subdivision being different questions, not about two bars.

## Then

Both scenes must pass all three. **A failure closes appearance capacity as an axis,
and with density control and topology already closed, closes mechanism improvement on
this baseline** — no re-run with a different `k`, alpha threshold, view count, or
eligibility rule. A pass authorises exactly one thing: a training gate for adaptive
texel-order allocation at matched parameter count.
