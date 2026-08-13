#!/usr/bin/env bash
set -euo pipefail

: "${NAS_ROOT:?Set NAS_ROOT to the persistent experiment root}"
: "${GORFE_V1_GARDEN_DATA:?Set the explicit Garden dataset root}"
: "${GORFE_V1_ROOM_DATA:?Set the explicit Room dataset root}"
: "${GORFE_V1_GARDEN_MODEL:?Set the explicit Garden iteration-30000 model root}"
: "${GORFE_V1_ROOM_MODEL:?Set the explicit Room iteration-30000 model root}"
: "${GORFE_V1_PREPARE_ROOT:?Set the completed GoRFE-V1 preparation root}"
: "${GORFE_V1_FREEZE_FILE:?Set the tracked candidate_freeze_<attempt>.json path}"

RUN_SUFFIX="${GORFE_V1_RUN_SUFFIX:-01}"
GPU_ID="${GORFE_V1_GPU:-3}"
PYTHON_BIN="${GORFE_V1_PYTHON:-python}"
EXPECTED_TORCH="2.7.1+cu126"
EXPECTED_CUDA="12.6"
MIN_FREE_MIB=40000
OUT="${NAS_ROOT}/experiments/gorfe_v1_evaluate_${RUN_SUFFIX}"

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${REPOSITORY}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GORFE_V1_GPU must be a non-negative physical GPU index." >&2
  exit 2
fi
for path in \
  "${GORFE_V1_GARDEN_DATA}" "${GORFE_V1_ROOM_DATA}" \
  "${GORFE_V1_GARDEN_MODEL}" "${GORFE_V1_ROOM_MODEL}" \
  "${GORFE_V1_PREPARE_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Required GoRFE-V1 input directory does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${GORFE_V1_FREEZE_FILE}" ]]; then
  echo "Tracked candidate-freeze file does not exist: ${GORFE_V1_FREEZE_FILE}" >&2
  exit 2
fi
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite GoRFE-V1 evaluation root: ${OUT}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked repository changes detected; evaluation requires a clean commit." >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null; then
  echo "GoRFE-V1 Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys, torch; tv, cv = sys.argv[1:]; print("python", sys.executable, "torch", torch.__version__, "cuda", torch.version.cuda); raise SystemExit(0 if torch.__version__ == tv and torch.version.cuda == cv and torch.cuda.is_available() else 1)' "${EXPECTED_TORCH}" "${EXPECTED_CUDA}"; then
  echo "GoRFE-V1 requires torch ${EXPECTED_TORCH} with CUDA build ${EXPECTED_CUDA}." >&2
  exit 2
fi

command -v nvidia-smi >/dev/null
FREE_MIB="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
PROCESSES="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ ! "${FREE_MIB}" =~ ^[0-9]+$ ]] || [[ -n "${PROCESSES}" ]] || [[ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]]; then
  echo "Physical GPU ${GPU_ID} is not exclusive for GoRFE-V1 evaluation." >&2
  echo "Free memory: ${FREE_MIB:-unknown} MiB; required: ${MIN_FREE_MIB} MiB." >&2
  if [[ -n "${PROCESSES}" ]]; then
    echo "Compute processes (pid, name, used MiB):" >&2
    echo "${PROCESSES}" >&2
  fi
  exit 2
fi

mkdir -p "${NAS_ROOT}/experiments"
LOCAL_BUILD_ROOT="$(mktemp -d /tmp/gorfe-v1.XXXXXX)"
case "${LOCAL_BUILD_ROOT}" in
  /tmp/gorfe-v1.*) ;;
  *)
    echo "Refusing unexpected temporary directory: ${LOCAL_BUILD_ROOT}" >&2
    exit 2
    ;;
esac
finish() {
  status=$?
  if [[ "${status}" -ne 0 && -d "${OUT}" && ! -e "${OUT}/DONE" && ! -e "${OUT}/FAILED" ]]; then
    printf 'runner exit status %s\n' "${status}" > "${OUT}/FAILED"
  fi
  rm -rf -- "${LOCAL_BUILD_ROOT}"
  return "${status}"
}
trap finish EXIT
mkdir "${OUT}"
mkdir "${LOCAL_BUILD_ROOT}/tmp" "${LOCAL_BUILD_ROOT}/torch_extensions"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="${LOCAL_BUILD_ROOT}/tmp"
export TMP="${LOCAL_BUILD_ROOT}/tmp"
export TEMP="${LOCAL_BUILD_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${LOCAL_BUILD_ROOT}/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1
export GORFE_V1_SOURCE_REVISION="$(git rev-parse HEAD)"

nvidia-smi --id="${GPU_ID}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  | tee "${OUT}/gpu.txt"

"${PYTHON_BIN}" gorfe_v1_freeze_verify.py \
  --prepare-root "${GORFE_V1_PREPARE_ROOT}" \
  --freeze-file "${GORFE_V1_FREEZE_FILE}" \
  --output "${OUT}/freeze_verify.json"

shopt -s nullglob
FROZEN_WHEELS=("${GORFE_V1_PREPARE_ROOT}"/wheels/diff_triangle_rasterization-*.whl)
shopt -u nullglob
if [[ "${#FROZEN_WHEELS[@]}" -ne 1 ]]; then
  echo "Expected exactly one hash-verified frozen wheel, found ${#FROZEN_WHEELS[@]}." >&2
  exit 2
fi
"${PYTHON_BIN}" -m pip install "${FROZEN_WHEELS[0]}" --force-reinstall --no-deps \
  2>&1 | tee "${OUT}/install.log"

"${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; from diff_triangle_rasterization import _C; from gorfe_v1_io import sha256_file; frozen=Path(sys.argv[1]).resolve(); loaded=Path(_C.__file__).resolve(); fs=sha256_file(frozen); ls=sha256_file(loaded); assert fs == ls, "installed extension differs from frozen binary"; print(json.dumps({"decision":"pass","frozen_path":str(frozen),"frozen_sha256":fs,"loaded_path":str(loaded),"loaded_sha256":ls},indent=2,sort_keys=True))' \
  "${GORFE_V1_PREPARE_ROOT}/native_extension.so" | tee "${OUT}/extension_verify.json"
"${PYTHON_BIN}" -c 'import json, platform, sys, torch; print(json.dumps({"cuda_build":torch.version.cuda,"executable":sys.executable,"platform":platform.platform(),"python":sys.version,"torch":torch.__version__},indent=2,sort_keys=True))' \
  | tee "${OUT}/python_env.txt"

"${PYTHON_BIN}" -m unittest discover -s tests -t . -v 2>&1 | tee "${OUT}/tests.log"
"${PYTHON_BIN}" gorfe_v1_native_smoke.py --output "${OUT}/native_result.json" \
  2>&1 | tee "${OUT}/native_smoke.log"

"${PYTHON_BIN}" gorfe_v1_evaluate.py \
  --scene garden \
  --dataset-root "${GORFE_V1_GARDEN_DATA}" \
  --model-root "${GORFE_V1_GARDEN_MODEL}" \
  --prepare-root "${GORFE_V1_PREPARE_ROOT}" \
  --freeze-file "${GORFE_V1_FREEZE_FILE}" \
  --output "${OUT}/garden" \
  --physical-gpu "${GPU_ID}" \
  2>&1 | tee "${OUT}/garden.log"

"${PYTHON_BIN}" gorfe_v1_evaluate.py \
  --scene room \
  --dataset-root "${GORFE_V1_ROOM_DATA}" \
  --model-root "${GORFE_V1_ROOM_MODEL}" \
  --prepare-root "${GORFE_V1_PREPARE_ROOT}" \
  --freeze-file "${GORFE_V1_FREEZE_FILE}" \
  --output "${OUT}/room" \
  --physical-gpu "${GPU_ID}" \
  2>&1 | tee "${OUT}/room.log"

"${PYTHON_BIN}" gorfe_v1_decide.py \
  --garden-result "${OUT}/garden/result.json" \
  --room-result "${OUT}/room/result.json" \
  --output-root "${OUT}"
"${PYTHON_BIN}" gorfe_v1_finalize.py --root "${OUT}" --phase evaluate

echo "GoRFE-V1 evaluation completed: ${OUT}"
echo "Overall decision: ${OUT}/decision.json"
