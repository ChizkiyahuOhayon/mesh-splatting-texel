#!/usr/bin/env bash
set -euo pipefail

: "${NAS_ROOT:?Set NAS_ROOT to the persistent experiment root}"
: "${GARDEN_DATA:?Set GARDEN_DATA to the Mip-NeRF 360 garden directory}"

RUN_SUFFIX="${HARD_G0_RUN_SUFFIX:-01}"
GPU_ID="${HARD_G0_GPU:-0}"
MIN_FREE_MIB=40000
TRAIN_ROOT="${NAS_ROOT}/runs"
EVAL_ROOT="${NAS_ROOT}/experiments"
LOG_ROOT="${NAS_ROOT}/logs"
TMP_ROOT="${NAS_ROOT}/tmp"
TORCH_EXTENSIONS_DIR="${NAS_ROOT}/torch_extensions"

STOCK_MODEL="${TRAIN_ROOT}/hard_g0_garden_stock_${RUN_SUFFIX}"
EARLY_MODEL="${TRAIN_ROOT}/hard_g0_garden_early25000_${RUN_SUFFIX}"
STOCK_EVAL="${EVAL_ROOT}/hard_g0_garden_stock_${RUN_SUFFIX}"
EARLY_EVAL="${EVAL_ROOT}/hard_g0_garden_early25000_${RUN_SUFFIX}"
DECISION="${EVAL_ROOT}/hard_g0_garden_decision_${RUN_SUFFIX}.json"
STOCK_LOG="${LOG_ROOT}/hard_g0_garden_stock_${RUN_SUFFIX}.log"
EARLY_LOG="${LOG_ROOT}/hard_g0_garden_early25000_${RUN_SUFFIX}.log"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "HARD_G0_GPU must be a non-negative physical GPU index." >&2
  exit 2
fi

for required in \
    "${GARDEN_DATA}/images" \
    "${GARDEN_DATA}/sparse/0/cameras.bin" \
    "${GARDEN_DATA}/sparse/0/images.bin" \
    "${GARDEN_DATA}/sparse/0/points3D.bin"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required dataset path: ${required}" >&2
    exit 2
  fi
done

for output in \
    "${STOCK_MODEL}" "${EARLY_MODEL}" \
    "${STOCK_EVAL}" "${EARLY_EVAL}" "${DECISION}" \
    "${STOCK_LOG}" "${EARLY_LOG}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing HARD-G0 artifact: ${output}" >&2
    echo "Set HARD_G0_RUN_SUFFIX to a fresh suffix for a new run." >&2
    exit 2
  fi
done

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked repository changes detected; run HARD-G0 from a clean checkout." >&2
  exit 2
fi

command -v nvidia-smi >/dev/null

require_idle_gpu() {
  local stage="$1"
  local free_mib
  local processes

  free_mib="$(nvidia-smi --id="${GPU_ID}" \
    --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "Could not read free memory for physical GPU ${GPU_ID}." >&2
    exit 2
  fi

  processes="$(nvidia-smi --id="${GPU_ID}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits)"
  if [[ -n "${processes}" || "${free_mib}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "Physical GPU ${GPU_ID} is not exclusive before ${stage}." >&2
    echo "Free memory: ${free_mib} MiB; required: ${MIN_FREE_MIB} MiB." >&2
    if [[ -n "${processes}" ]]; then
      echo "Compute processes (pid, name, used MiB):" >&2
      echo "${processes}" >&2
    fi
    exit 2
  fi

  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader
}

require_idle_gpu "stock training"

mkdir -p "${TRAIN_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}" \
         "${TMP_ROOT}" "${TORCH_EXTENSIONS_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export TORCH_EXTENSIONS_DIR
export PYTHONDONTWRITEBYTECODE=1

SOURCE_REVISION="$(git rev-parse HEAD)"
export SVSR_SOURCE_REVISION="${SOURCE_REVISION}"
echo "HARD-G0 source revision: ${SOURCE_REVISION}"
echo "Training stock arm on physical GPU ${GPU_ID}..."
/usr/bin/time -f "HARD_G0_TRAINING_SECONDS %e" \
  python train.py \
    -s "${GARDEN_DATA}" \
    -m "${STOCK_MODEL}" \
    --eval \
    --set_sigma 1.0 \
    --sigma_until 30000 \
    --iterations 30000 \
    --final_opacity_iter 24000 \
    --final_scaling 4 \
    --cleanup_scaling 4 \
    --seed 0 \
    >"${STOCK_LOG}" 2>&1

require_idle_gpu "early25000 training"
echo "Training early25000 arm on physical GPU ${GPU_ID}..."
/usr/bin/time -f "HARD_G0_TRAINING_SECONDS %e" \
  python train.py \
    -s "${GARDEN_DATA}" \
    -m "${EARLY_MODEL}" \
    --eval \
    --set_sigma 1.0 \
    --sigma_until 25000 \
    --iterations 30000 \
    --final_opacity_iter 24000 \
    --final_scaling 4 \
    --cleanup_scaling 4 \
    --seed 0 \
    >"${EARLY_LOG}" 2>&1

training_seconds() {
  awk '$1 == "HARD_G0_TRAINING_SECONDS" {value = $2} END {if (value == "") exit 1; print value}' "$1"
}

STOCK_SECONDS="$(training_seconds "${STOCK_LOG}")"
EARLY_SECONDS="$(training_seconds "${EARLY_LOG}")"

require_idle_gpu "stock evaluation"
echo "Evaluating stock arm at scaling 2 and 4..."
python sac_eval.py \
  -s "${GARDEN_DATA}" \
  -m "${STOCK_MODEL}" \
  --iteration 30000 \
  --eval \
  --sac_output "${STOCK_EVAL}" \
  --sac_protocol experiments/hard_g0/protocol.md \
  --sac_arm stock \
  --sac_scene garden \
  --sac_seed 0 \
  --sac_training_seconds "${STOCK_SECONDS}"

require_idle_gpu "early25000 evaluation"
echo "Evaluating early25000 arm at scaling 2 and 4..."
python sac_eval.py \
  -s "${GARDEN_DATA}" \
  -m "${EARLY_MODEL}" \
  --iteration 30000 \
  --eval \
  --sac_output "${EARLY_EVAL}" \
  --sac_protocol experiments/hard_g0/protocol.md \
  --sac_arm early \
  --sac_scene garden \
  --sac_seed 0 \
  --sac_training_seconds "${EARLY_SECONDS}"

python hard_g0_decide.py \
  --stock "${STOCK_EVAL}" \
  --early "${EARLY_EVAL}" \
  --output "${DECISION}"

echo "HARD-G0 completed. Decision artifact: ${DECISION}"
