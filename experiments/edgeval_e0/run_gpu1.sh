#!/usr/bin/env bash
set -euo pipefail

: "${NAS_ROOT:?Set NAS_ROOT to the persistent experiment root}"

RUN_SUFFIX="${EDGEVAL_E0_RUN_SUFFIX:-01}"
GPU_ID="${EDGEVAL_E0_GPU:-1}"
MIN_FREE_MIB=40000
OUT="${NAS_ROOT}/experiments/edgeval_e0_${RUN_SUFFIX}"
TMP_ROOT="${NAS_ROOT}/tmp"
TORCH_EXTENSIONS_DIR="${NAS_ROOT}/torch_extensions"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "EDGEVAL_E0_GPU must be a non-negative physical GPU index." >&2
  exit 2
fi
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite EdgeVal-E0 artifact directory: ${OUT}" >&2
  echo "Set EDGEVAL_E0_RUN_SUFFIX to a fresh suffix." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked repository changes detected; run EdgeVal-E0 from a clean checkout." >&2
  exit 2
fi

command -v nvidia-smi >/dev/null
FREE_MIB="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
PROCESSES="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ ! "${FREE_MIB}" =~ ^[0-9]+$ ]] || [[ -n "${PROCESSES}" ]] || [[ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]]; then
  echo "Physical GPU ${GPU_ID} is not exclusive for EdgeVal-E0." >&2
  echo "Free memory: ${FREE_MIB:-unknown} MiB; required: ${MIN_FREE_MIB} MiB." >&2
  if [[ -n "${PROCESSES}" ]]; then
    echo "Compute processes (pid, name, used MiB):" >&2
    echo "${PROCESSES}" >&2
  fi
  exit 2
fi

mkdir -p "${NAS_ROOT}/experiments" "${TMP_ROOT}" "${TORCH_EXTENSIONS_DIR}"
mkdir "${OUT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export TORCH_EXTENSIONS_DIR
export PYTHONDONTWRITEBYTECODE=1
export EDGEVAL_SOURCE_REVISION="$(git rev-parse HEAD)"

nvidia-smi --id="${GPU_ID}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | tee "${OUT}/gpu.txt"
python -c 'import torch; assert torch.cuda.is_available(), "active Python environment has no CUDA"; print("torch", torch.__version__, "cuda", torch.version.cuda)' \
  | tee "${OUT}/python_env.txt"

python -m pip install ./submodules/diff-triangle-mesh-rasterization \
  --force-reinstall --no-deps --no-build-isolation \
  2>&1 | tee "${OUT}/build.log"

python -m unittest tests.test_edgeval_core -v 2>&1 | tee "${OUT}/tests.log"

python edgeval_e0_smoke.py --output "${OUT}/result.json" \
  2>&1 | tee "${OUT}/smoke.log"

python edgeval_e0_manifest.py \
  --result "${OUT}/result.json" \
  --build-log "${OUT}/build.log" \
  --test-log "${OUT}/tests.log" \
  --smoke-log "${OUT}/smoke.log" \
  --output "${OUT}/manifest.json"

printf 'complete\n' > "${OUT}/DONE"
sha256sum "${OUT}/gpu.txt" "${OUT}/python_env.txt" "${OUT}/build.log" "${OUT}/tests.log" \
  "${OUT}/result.json" "${OUT}/smoke.log" "${OUT}/manifest.json" "${OUT}/DONE" \
  > "${OUT}/SHA256SUMS"

echo "EdgeVal-E0 completed: ${OUT}"
