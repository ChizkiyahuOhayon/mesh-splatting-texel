# COADAPT-G0 server runbook

Run from the NAS source archive after downloading the locked revision. The root
filesystem remains read-only for outputs; temporary files and results stay on NAS.

An archive deployment must update the evaluator and all three Python dependencies
from the same revision. Mixing the earlier `a73c0f4` metadata with the COADAPT
evaluator fails because that metadata predates `reserve_output_directory`.

```bash
SOURCE_REVISION=604e66244942184669e59f5c714bd269088f5302
BASE_URL="https://raw.githubusercontent.com/ChizkiyahuOhayon/mesh-splatting-texel/$SOURCE_REVISION"

curl -fL "$BASE_URL/coadapt_decompose.py" -o coadapt_decompose.py
curl -fL "$BASE_URL/coadapt_g0_eval.py" -o coadapt_g0_eval.py
curl -fL "$BASE_URL/svsr_g1_eval.py" -o svsr_g1_eval.py
curl -fL "$BASE_URL/svsr_metadata.py" -o svsr_metadata.py

mkdir -p experiments/coadapt_g0
curl -fL "$BASE_URL/experiments/coadapt_g0/protocol.md" \
  -o experiments/coadapt_g0/protocol.md

SVSR_SOURCE_REVISION="$SOURCE_REVISION" python -c \
  "from svsr_metadata import reserve_output_directory, source_revision; print(source_revision()); print('imports OK')"
```

```bash
ROOM_DATA=/home/smbu/dy/mesh-splatting/data/mipnerf360/room
ROOM_SH=/home/smbu/dy/mesh-splatting-texel/output/main/room_baseline
ROOM_TEXEL=/home/smbu/dy/mesh-splatting-texel/output/main/room_texel2
NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu

COADAPT_SMOKE="$NAS_ROOT/experiments/coadapt_g0_room_smoke_01"
test ! -e "$COADAPT_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" \
TMP="$NAS_ROOT/tmp" \
TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" \
PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" \
CUDA_VISIBLE_DEVICES=0 \
python coadapt_g0_eval.py \
  -s "$ROOM_DATA" \
  -m "$ROOM_TEXEL" \
  --eval \
  --coadapt_sh_model "$ROOM_SH" \
  --coadapt_output "$COADAPT_SMOKE" \
  --coadapt_max_views 1
```

After the smoke completes without an exception, use a fresh path and omit the
smoke-only flag:

```bash
COADAPT_FULL="$NAS_ROOT/experiments/coadapt_g0_room_full_01"
test ! -e "$COADAPT_FULL" && echo "full output path OK"

TMPDIR="$NAS_ROOT/tmp" \
TMP="$NAS_ROOT/tmp" \
TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" \
PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" \
CUDA_VISIBLE_DEVICES=0 \
python coadapt_g0_eval.py \
  -s "$ROOM_DATA" \
  -m "$ROOM_TEXEL" \
  --eval \
  --coadapt_sh_model "$ROOM_SH" \
  --coadapt_output "$COADAPT_FULL"
```

Inspect the locked result:

```bash
python -c \
  "import json; d=json.load(open('$COADAPT_FULL/results.json')); print(json.dumps(d['summary'],indent=2)); print(json.dumps(d['recovery_fraction'],indent=2))"
```
