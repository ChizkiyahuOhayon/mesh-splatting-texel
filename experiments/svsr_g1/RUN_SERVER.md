# SVSR-G1 server runbook

No CUDA rebuild is required. All outputs and temporary files go to the NAS because
the server root filesystem is full.

## Garden one-view smoke

```bash
GARDEN_DATA=/home/smbu/dy/mesh-splatting/data/mipnerf360/garden
GARDEN_SH=/home/smbu/dy/mesh-splatting/output/mipnerf360/garden
GARDEN_TEXEL=/home/smbu/dy/mesh-splatting-texel/output/main/garden_texel2
NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
WORKTREE="$NAS_ROOT/code/mesh-splatting-texel-v10"
SVSR_OUT="$NAS_ROOT/experiments/svsr_g1_garden_smoke_02"

mkdir -p "$NAS_ROOT/code" "$NAS_ROOT/tmp" "$NAS_ROOT/torch_extensions" "$NAS_ROOT/experiments"
cd "$WORKTREE"

SVSR_SOURCE_REVISION=$(git ls-remote \
  git@github.com:ChizkiyahuOhayon/mesh-splatting-texel.git \
  refs/heads/main | cut -f1)
test -n "$SVSR_SOURCE_REVISION" && echo "source revision: $SVSR_SOURCE_REVISION"

test -d "$GARDEN_DATA/images" && echo "dataset OK"
test -f "$GARDEN_SH/cfg_args" && echo "SH model OK"
test -f "$GARDEN_TEXEL/cfg_args" && echo "texel model OK"
find "$GARDEN_SH/point_cloud" -name point_cloud_state_dict.pt -print
find "$GARDEN_TEXEL/point_cloud" -name point_cloud_state_dict.pt -print
test ! -e "$SVSR_OUT" && echo "output path OK"

TMPDIR="$NAS_ROOT/tmp" \
TMP="$NAS_ROOT/tmp" \
TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" \
PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SVSR_SOURCE_REVISION" \
CUDA_VISIBLE_DEVICES=0 \
python svsr_g1_eval.py \
  -s "$GARDEN_DATA" \
  -m "$GARDEN_TEXEL" \
  --eval \
  --svsr_scene garden \
  --svsr_sh_model "$GARDEN_SH" \
  --svsr_output "$SVSR_OUT" \
  --svsr_max_views 1

python -c \
  "import json; d=json.load(open('$SVSR_OUT/results.json')); print(json.dumps(d['summary'], indent=2))"
```

The smoke output is exploratory and cannot pass G1. After checking it, use a new
output directory and omit `--svsr_max_views 1` for the full Garden result.
