#!/usr/bin/env bash
set -euo pipefail

: "${NAS_ROOT:?Set NAS_ROOT to the persistent experiment root}"
: "${GORFE_V1_GARDEN_DATA:?Set the explicit Garden dataset root}"
: "${GORFE_V1_ROOM_DATA:?Set the explicit Room dataset root}"
: "${GORFE_V1_GARDEN_MODEL:?Set the explicit Garden iteration-30000 model root}"
: "${GORFE_V1_ROOM_MODEL:?Set the explicit Room iteration-30000 model root}"

RUN_SUFFIX="${GORFE_V1_RUN_SUFFIX:-01}"
GPU_ID="${GORFE_V1_GPU:-3}"
PYTHON_BIN="${GORFE_V1_PYTHON:-python}"
EXPECTED_TORCH="2.7.1+cu126"
EXPECTED_CUDA="12.6"
MIN_FREE_MIB=40000
OUT="${NAS_ROOT}/experiments/gorfe_v1_prepare_${RUN_SUFFIX}"
FREEZE_OUTPUT="${GORFE_V1_FREEZE_OUTPUT:-${NAS_ROOT}/experiments/gorfe_v1_candidate_freeze_${RUN_SUFFIX}.json}"

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${REPOSITORY}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GORFE_V1_GPU must be a non-negative physical GPU index." >&2
  exit 2
fi
for path in \
  "${GORFE_V1_GARDEN_DATA}" "${GORFE_V1_ROOM_DATA}" \
  "${GORFE_V1_GARDEN_MODEL}" "${GORFE_V1_ROOM_MODEL}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Required GoRFE-V1 input directory does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite GoRFE-V1 preparation root: ${OUT}" >&2
  exit 2
fi
if [[ -e "${FREEZE_OUTPUT}" ]]; then
  echo "Refusing to overwrite candidate-freeze payload: ${FREEZE_OUTPUT}" >&2
  exit 2
fi
case "${FREEZE_OUTPUT}" in
  "${OUT}"|"${OUT}"/*)
    echo "Candidate-freeze payload must be outside the preparation root." >&2
    exit 2
    ;;
esac
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked repository changes detected; preparation requires a clean commit." >&2
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
  echo "Physical GPU ${GPU_ID} is not exclusive for GoRFE-V1 preparation." >&2
  echo "Free memory: ${FREE_MIB:-unknown} MiB; required: ${MIN_FREE_MIB} MiB." >&2
  if [[ -n "${PROCESSES}" ]]; then
    echo "Compute processes (pid, name, used MiB):" >&2
    echo "${PROCESSES}" >&2
  fi
  exit 2
fi

mkdir -p "${NAS_ROOT}/experiments" "$(dirname "${FREEZE_OUTPUT}")"
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
mkdir "${LOCAL_BUILD_ROOT}/tmp" "${LOCAL_BUILD_ROOT}/torch_extensions" "${LOCAL_BUILD_ROOT}/wheels"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="${LOCAL_BUILD_ROOT}/tmp"
export TMP="${LOCAL_BUILD_ROOT}/tmp"
export TEMP="${LOCAL_BUILD_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${LOCAL_BUILD_ROOT}/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1
export GORFE_V1_SOURCE_REVISION="$(git rev-parse HEAD)"

# Bind both dataset/checkpoint identities before a persistent attempt directory
# exists.  This mode cannot decode target images, initialize CUDA, or import the
# native rasterizer that is built below.
"${PYTHON_BIN}" gorfe_v1_prepare.py \
  --preflight-only \
  --scene garden \
  --dataset-root "${GORFE_V1_GARDEN_DATA}" \
  --model-root "${GORFE_V1_GARDEN_MODEL}" \
  --physical-gpu "${GPU_ID}" \
  > "${LOCAL_BUILD_ROOT}/garden_preflight.json"
"${PYTHON_BIN}" gorfe_v1_prepare.py \
  --preflight-only \
  --scene room \
  --dataset-root "${GORFE_V1_ROOM_DATA}" \
  --model-root "${GORFE_V1_ROOM_MODEL}" \
  --physical-gpu "${GPU_ID}" \
  > "${LOCAL_BUILD_ROOT}/room_preflight.json"

mkdir "${OUT}"
mkdir "${OUT}/wheels"
cp "${LOCAL_BUILD_ROOT}/garden_preflight.json" "${OUT}/garden_preflight.json"
cp "${LOCAL_BUILD_ROOT}/room_preflight.json" "${OUT}/room_preflight.json"

nvidia-smi --id="${GPU_ID}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  | tee "${OUT}/gpu.txt"
"${PYTHON_BIN}" -c 'import json, platform, sys, torch; print(json.dumps({"cuda_build":torch.version.cuda,"executable":sys.executable,"platform":platform.platform(),"python":sys.version,"torch":torch.__version__},indent=2,sort_keys=True))' \
  | tee "${OUT}/python_env.txt"

{
  "${PYTHON_BIN}" -m pip wheel ./submodules/diff-triangle-mesh-rasterization \
    --wheel-dir "${LOCAL_BUILD_ROOT}/wheels" --no-deps --no-build-isolation
  shopt -s nullglob
  built_wheels=("${LOCAL_BUILD_ROOT}/wheels"/diff_triangle_rasterization-*.whl)
  shopt -u nullglob
  if [[ "${#built_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one native wheel, found ${#built_wheels[@]}." >&2
    exit 2
  fi
  cp "${built_wheels[0]}" "${OUT}/wheels/$(basename "${built_wheels[0]}")"
  "${PYTHON_BIN}" -m pip install "${OUT}/wheels/$(basename "${built_wheels[0]}")" \
    --force-reinstall --no-deps
} 2>&1 | tee "${OUT}/build.log"

EXTENSION_PATH="$("${PYTHON_BIN}" -c 'from diff_triangle_rasterization import _C; print(_C.__file__)')"
if [[ ! -f "${EXTENSION_PATH}" ]]; then
  echo "Installed GoRFE-V1 extension was not found: ${EXTENSION_PATH}" >&2
  exit 2
fi
cp "${EXTENSION_PATH}" "${OUT}/native_extension.so"

"${PYTHON_BIN}" -m unittest discover -s tests -t . -v 2>&1 | tee "${OUT}/tests.log"
"${PYTHON_BIN}" gorfe_v1_native_smoke.py --output "${OUT}/native_result.json" \
  2>&1 | tee "${OUT}/native_smoke.log"

"${PYTHON_BIN}" gorfe_v1_prepare.py \
  --scene garden \
  --dataset-root "${GORFE_V1_GARDEN_DATA}" \
  --model-root "${GORFE_V1_GARDEN_MODEL}" \
  --output "${OUT}/garden" \
  --physical-gpu "${GPU_ID}" \
  2>&1 | tee "${OUT}/garden.log"

"${PYTHON_BIN}" gorfe_v1_prepare.py \
  --scene room \
  --dataset-root "${GORFE_V1_ROOM_DATA}" \
  --model-root "${GORFE_V1_ROOM_MODEL}" \
  --output "${OUT}/room" \
  --physical-gpu "${GPU_ID}" \
  2>&1 | tee "${OUT}/room.log"

"${PYTHON_BIN}" gorfe_v1_finalize.py --root "${OUT}" --phase prepare
"${PYTHON_BIN}" gorfe_v1_freeze_payload.py \
  --prepare-root "${OUT}" \
  --output "${FREEZE_OUTPUT}"

echo "GoRFE-V1 preparation completed: ${OUT}"
echo "Candidate-freeze payload (must be committed before evaluation): ${FREEZE_OUTPUT}"
