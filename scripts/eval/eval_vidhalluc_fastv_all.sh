#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
VIDHALLUC_DATA_ROOT="${VIDHALLUC_DATA_ROOT:?VIDHALLUC_DATA_ROOT must be set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vidhalluc}"
SUBSET="${SUBSET:-ach_binaryqa}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"
NUM_FRAMES="${NUM_FRAMES:-16}"

MODEL_PATH="${MODEL_PATH}" \
VIDHALLUC_DATA_ROOT="${VIDHALLUC_DATA_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
SUBSET="${SUBSET}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
NUM_FRAMES="${NUM_FRAMES}" \
USE_FASTV=0 \
RUN_NAME=baseline \
bash "${SCRIPT_DIR}/eval_vidhalluc_fastv.sh"

MODEL_PATH="${MODEL_PATH}" \
VIDHALLUC_DATA_ROOT="${VIDHALLUC_DATA_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
SUBSET="${SUBSET}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
NUM_FRAMES="${NUM_FRAMES}" \
USE_FASTV=1 \
FASTV_K=3 \
FASTV_R=0.5 \
RUN_NAME=fastv_default \
bash "${SCRIPT_DIR}/eval_vidhalluc_fastv.sh"

for RATIO in 0.25 0.75; do
  MODEL_PATH="${MODEL_PATH}" \
  VIDHALLUC_DATA_ROOT="${VIDHALLUC_DATA_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  SUBSET="${SUBSET}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  NUM_FRAMES="${NUM_FRAMES}" \
  USE_FASTV=1 \
  FASTV_K=3 \
  FASTV_R="${RATIO}" \
  RUN_NAME="fastv_r_${RATIO}" \
  bash "${SCRIPT_DIR}/eval_vidhalluc_fastv.sh"
done

CKPT_NAME="$(basename "${MODEL_PATH}")"
EXP_ROOT="${OUTPUT_ROOT}/${CKPT_NAME}/${SUBSET}_${MAX_SAMPLES}"

"${PYTHON_BIN}" -m videollama2.eval.vidhalluc.compare \
  --baseline "${EXP_ROOT}/baseline" \
  --pruned "${EXP_ROOT}/fastv_default" \
  --out "${EXP_ROOT}/matched_${MAX_SAMPLES}_comparison.md"

echo "All VidHalluc experiments complete: ${EXP_ROOT}"
