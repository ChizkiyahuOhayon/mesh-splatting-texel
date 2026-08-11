#!/usr/bin/env bash
set -euo pipefail

: "${NAS_ROOT:?Set NAS_ROOT to the persistent experiment root}"

RUN_SUFFIX="${GORFE_Q0_RUN_SUFFIX:-01}"
GPU_ID="${GORFE_Q0_GPU:-2}"
MIN_FREE_MIB=40000
OUT="${NAS_ROOT}/experiments/gorfe_q0_${RUN_SUFFIX}"
PYTHON_BIN="${GORFE_Q0_PYTHON:-python}"
EXPECTED_TORCH="2.7.1+cu126"
EXPECTED_CUDA="12.6"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GORFE_Q0_GPU must be a non-negative physical GPU index." >&2
  exit 2
fi
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite GoRFE-Q0 artifact directory: ${OUT}" >&2
  echo "Set GORFE_Q0_RUN_SUFFIX to a fresh suffix." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked repository changes detected; run GoRFE-Q0 from a clean checkout." >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null; then
  echo "GoRFE-Q0 Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys, torch; expected_torch, expected_cuda = sys.argv[1:]; print("python", sys.executable, "torch", torch.__version__, "cuda", torch.version.cuda); raise SystemExit(0 if torch.__version__ == expected_torch and torch.version.cuda == expected_cuda and torch.cuda.is_available() else 1)' "${EXPECTED_TORCH}" "${EXPECTED_CUDA}"; then
  echo "GoRFE-Q0 requires torch ${EXPECTED_TORCH} with CUDA ${EXPECTED_CUDA}." >&2
  echo "Activate the mesh_splatting environment before reserving an attempt suffix." >&2
  exit 2
fi

command -v nvidia-smi >/dev/null
FREE_MIB="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
PROCESSES="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ ! "${FREE_MIB}" =~ ^[0-9]+$ ]] || [[ -n "${PROCESSES}" ]] || [[ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]]; then
  echo "Physical GPU ${GPU_ID} is not exclusive for GoRFE-Q0." >&2
  echo "Free memory: ${FREE_MIB:-unknown} MiB; required: ${MIN_FREE_MIB} MiB." >&2
  if [[ -n "${PROCESSES}" ]]; then
    echo "Compute processes (pid, name, used MiB):" >&2
    echo "${PROCESSES}" >&2
  fi
  exit 2
fi

mkdir -p "${NAS_ROOT}/experiments"
mkdir "${OUT}"
LOCAL_BUILD_ROOT="$(mktemp -d /tmp/gorfe-q0.XXXXXX)"
case "${LOCAL_BUILD_ROOT}" in
  /tmp/gorfe-q0.*) ;;
  *)
    echo "Refusing unexpected local build path: ${LOCAL_BUILD_ROOT}" >&2
    exit 2
    ;;
esac
cleanup_local_build() {
  rm -rf -- "${LOCAL_BUILD_ROOT}"
}
trap cleanup_local_build EXIT
mkdir "${LOCAL_BUILD_ROOT}/tmp" "${LOCAL_BUILD_ROOT}/torch_extensions"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="${LOCAL_BUILD_ROOT}/tmp"
export TMP="${LOCAL_BUILD_ROOT}/tmp"
export TEMP="${LOCAL_BUILD_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${LOCAL_BUILD_ROOT}/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1
export GORFE_SOURCE_REVISION="$(git rev-parse HEAD)"

nvidia-smi --id="${GPU_ID}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  | tee "${OUT}/gpu.txt"
"${PYTHON_BIN}" -c 'import sys, torch; print("python", sys.executable, "torch", torch.__version__, "cuda", torch.version.cuda)' \
  | tee "${OUT}/python_env.txt"

"${PYTHON_BIN}" -m pip install ./submodules/diff-triangle-mesh-rasterization \
  --force-reinstall --no-deps --no-build-isolation \
  2>&1 | tee "${OUT}/build.log"

"${PYTHON_BIN}" -m unittest discover -s tests -t . -v 2>&1 | tee "${OUT}/tests.log"

"${PYTHON_BIN}" gorfe_q0_smoke.py --output "${OUT}/result.json" \
  2>&1 | tee "${OUT}/smoke.log"

"${PYTHON_BIN}" gorfe_q0_manifest.py \
  --result "${OUT}/result.json" \
  --build-log "${OUT}/build.log" \
  --test-log "${OUT}/tests.log" \
  --smoke-log "${OUT}/smoke.log" \
  --output "${OUT}/manifest.json"

printf 'complete\n' > "${OUT}/DONE"
sha256sum "${OUT}/gpu.txt" "${OUT}/python_env.txt" "${OUT}/build.log" "${OUT}/tests.log" \
  "${OUT}/result.json" "${OUT}/smoke.log" "${OUT}/manifest.json" "${OUT}/DONE" \
  > "${OUT}/SHA256SUMS"

echo "GoRFE-Q0 completed: ${OUT}"
