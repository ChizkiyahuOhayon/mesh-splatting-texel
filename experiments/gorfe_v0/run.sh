#!/usr/bin/env bash
set -euo pipefail

: "${NAS_ROOT:?Set NAS_ROOT to the persistent experiment root}"

RUN_SUFFIX="${GORFE_V0_RUN_SUFFIX:-01}"
OUT="${NAS_ROOT}/experiments/gorfe_v0_${RUN_SUFFIX}"
PYTHON_BIN="${GORFE_V0_PYTHON:-python}"
EXPECTED_TORCH="2.7.1+cu126"
EXPECTED_CUDA="12.6"

if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite GoRFE-V0 artifact directory: ${OUT}" >&2
  echo "Set GORFE_V0_RUN_SUFFIX to a fresh suffix." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked repository changes detected; run GoRFE-V0 from a clean checkout." >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null; then
  echo "GoRFE-V0 Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys, torch; expected_torch, expected_cuda = sys.argv[1:]; print("python", sys.executable, "torch", torch.__version__, "cuda build", torch.version.cuda); raise SystemExit(0 if torch.__version__ == expected_torch and torch.version.cuda == expected_cuda else 1)' "${EXPECTED_TORCH}" "${EXPECTED_CUDA}"; then
  echo "GoRFE-V0 requires torch ${EXPECTED_TORCH} with CUDA build ${EXPECTED_CUDA}." >&2
  echo "Activate the mesh_splatting environment before reserving an attempt suffix." >&2
  exit 2
fi

mkdir -p "${NAS_ROOT}/experiments"
mkdir "${OUT}"
export PYTHONDONTWRITEBYTECODE=1
export GORFE_SOURCE_REVISION="$(git rev-parse HEAD)"

"${PYTHON_BIN}" -c 'import json, platform, sys, torch; print(json.dumps({"executable": sys.executable, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_build": torch.version.cuda}, indent=2, sort_keys=True))' \
  | tee "${OUT}/python_env.txt"

"${PYTHON_BIN}" -m unittest discover -s tests -t . -v \
  2>&1 | tee "${OUT}/tests.log"

"${PYTHON_BIN}" gorfe_v0_gate.py --output "${OUT}/result.json" \
  2>&1 | tee "${OUT}/gate.log"

"${PYTHON_BIN}" gorfe_v0_manifest.py \
  --result "${OUT}/result.json" \
  --test-log "${OUT}/tests.log" \
  --gate-log "${OUT}/gate.log" \
  --python-env "${OUT}/python_env.txt" \
  --output "${OUT}/manifest.json"

printf 'complete\n' > "${OUT}/DONE"
sha256sum "${OUT}/python_env.txt" "${OUT}/tests.log" "${OUT}/result.json" \
  "${OUT}/gate.log" "${OUT}/manifest.json" "${OUT}/DONE" \
  > "${OUT}/SHA256SUMS"

echo "GoRFE-V0 completed: ${OUT}"
