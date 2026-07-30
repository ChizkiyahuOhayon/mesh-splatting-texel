# FRT-G1 server runbook

## Sync the locked source

```bash
cd "$WORKTREE"

SOURCE_REVISION=8dd6b427002925a78381fe49eb7fd7f61b52b81b
BASE_URL="https://raw.githubusercontent.com/ChizkiyahuOhayon/mesh-splatting-texel/$SOURCE_REVISION"

curl -fL "$BASE_URL/frt_freeze.py" -o frt_freeze.py
curl -fL "$BASE_URL/frt_g1_train.py" -o frt_g1_train.py
curl -fL "$BASE_URL/coadapt_decompose.py" -o coadapt_decompose.py
curl -fL "$BASE_URL/coadapt_g0_eval.py" -o coadapt_g0_eval.py
curl -fL "$BASE_URL/svsr_g1_eval.py" -o svsr_g1_eval.py
curl -fL "$BASE_URL/svsr_metadata.py" -o svsr_metadata.py

mkdir -p experiments/frt_g1
curl -fL "$BASE_URL/experiments/frt_g1/protocol.md" \
  -o experiments/frt_g1/protocol.md

SVSR_SOURCE_REVISION="$SOURCE_REVISION" python -c \
  "from frt_freeze import freeze_base_tensors; from svsr_metadata import reserve_output_directory; print('FRT imports OK')"
```

## Two-update Room implementation smoke

```bash
ROOM_DATA=/home/smbu/dy/mesh-splatting/data/mipnerf360/room
ROOM_SH=/home/smbu/dy/mesh-splatting-texel/output/main/room_baseline
NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
FRT_SMOKE="$NAS_ROOT/experiments/frt_g1_room_smoke_01"

test ! -e "$FRT_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" \
TMP="$NAS_ROOT/tmp" \
TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" \
PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" \
CUDA_VISIBLE_DEVICES=0 \
python frt_g1_train.py \
  -s "$ROOM_DATA" \
  -m "$ROOM_SH" \
  --eval \
  --frt_scene room \
  --frt_output "$FRT_SMOKE" \
  --frt_smoke
```

Validate without reading the loss as a scientific result:

```bash
python -c \
  "import json,os; m=json.load(open('$FRT_SMOKE/frt_manifest.json')); t=json.load(open('$FRT_SMOKE/training.json')); assert m['smoke'] and m['updates']==2; assert t['zero_init_max_abs']<=1e-7; assert t['base_tensors_unchanged']; assert t['optimizer_parameter_groups']==['texels']; assert os.path.isfile('$FRT_SMOKE/DONE'); print('FRT SMOKE VALID')"
```

## Locked Garden and Room training

Only after the smoke passes, use two GPUs with one identical 5,000-update schedule:

```bash
GARDEN_DATA=/home/smbu/dy/mesh-splatting/data/mipnerf360/garden
GARDEN_SH=/home/smbu/dy/mesh-splatting-texel/output/main/garden_baseline
FRT_GARDEN="$NAS_ROOT/experiments/frt_g1_garden_5000_01"
FRT_ROOM="$NAS_ROOT/experiments/frt_g1_room_5000_01"

test ! -e "$FRT_GARDEN" && test ! -e "$FRT_ROOM" && echo "full output paths OK"
```

```bash
nohup env \
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" \
  PYTHONDONTWRITEBYTECODE=1 SVSR_SOURCE_REVISION="$SOURCE_REVISION" \
  CUDA_VISIBLE_DEVICES=0 \
  python frt_g1_train.py -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
    --frt_scene garden --frt_output "$FRT_GARDEN" \
  > "$FRT_GARDEN.log" 2>&1 &

nohup env \
  TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
  TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" \
  PYTHONDONTWRITEBYTECODE=1 SVSR_SOURCE_REVISION="$SOURCE_REVISION" \
  CUDA_VISIBLE_DEVICES=1 \
  python frt_g1_train.py -s "$ROOM_DATA" -m "$ROOM_SH" --eval \
    --frt_scene room --frt_output "$FRT_ROOM" \
  > "$FRT_ROOM.log" 2>&1 &
```

Do not evaluate a held-out split until both locked training jobs finish.
