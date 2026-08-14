#!/usr/bin/env bash
set -euo pipefail

NAS_ROOT="${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}"
PYTHON_BIN="${GORFE_D0_PYTHON:-/home/smbu/micromamba/envs/mesh_splatting/bin/python}"
RUN_SUFFIX="${GORFE_D0_RUN_SUFFIX:-01}"
PREPARE_ROOT="${GORFE_D0_V1_PREPARE_ROOT:-${NAS_ROOT}/experiments/gorfe_v1_prepare_04}"
EVALUATE_ROOT="${GORFE_D0_V1_EVALUATE_ROOT:-${NAS_ROOT}/experiments/gorfe_v1_evaluate_04}"
OUT="${NAS_ROOT}/experiments/gorfe_d0_a_${RUN_SUFFIX}"

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${REPOSITORY}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked repository changes detected; D0-A requires a clean commit." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "D0-A Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
for root in "${PREPARE_ROOT}" "${EVALUATE_ROOT}"; do
  if [[ ! -f "${root}/DONE" ]]; then
    echo "Sealed V1 root is unavailable: ${root}" >&2
    exit 2
  fi
done
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite D0-A output: ${OUT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" -c 'import sys,torch; print("python",sys.executable,"torch",torch.__version__,"cuda_visible",torch.cuda.is_available())'
"${PYTHON_BIN}" -m unittest tests.test_gorfe_d0 -v
"${PYTHON_BIN}" gorfe_d0_audit.py \
  --prepare-root "${PREPARE_ROOT}" \
  --evaluate-root "${EVALUATE_ROOT}" \
  --output "${OUT}"
sha256sum "${OUT}/result.json" "${OUT}/manifest.json" "${OUT}/SHA256SUMS" "${OUT}/DONE"

echo "GoRFE-D0-A completed: ${OUT}"
