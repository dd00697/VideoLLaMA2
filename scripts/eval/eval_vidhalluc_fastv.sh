#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
VIDHALLUC_DATA_ROOT="${VIDHALLUC_DATA_ROOT:?VIDHALLUC_DATA_ROOT must be set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vidhalluc}"
SUBSET="${SUBSET:-ach_binaryqa}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"
NUM_FRAMES="${NUM_FRAMES:-16}"
USE_FASTV="${USE_FASTV:-0}"
FASTV_K="${FASTV_K:-3}"
FASTV_R="${FASTV_R:-0.5}"
RESUME="${RESUME:-1}"
SAMPLE_TIMEOUT_SECONDS="${SAMPLE_TIMEOUT_SECONDS:-0}"
BAD_VIDEOS_JSON="${BAD_VIDEOS_JSON:-}"

CKPT_NAME="$(basename "${MODEL_PATH}")"
EXP_ROOT="${OUTPUT_ROOT}/${CKPT_NAME}/${SUBSET}_${MAX_SAMPLES}"

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${USE_FASTV}" == "1" ]]; then
    RUN_NAME="fastv_k_${FASTV_K}_r_${FASTV_R}"
  else
    RUN_NAME="baseline"
  fi
fi

SAVE_DIR="${EXP_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${gpu_list}"
if [[ "${#GPULIST[@]}" -eq 0 ]]; then
  GPULIST=("0")
fi
CHUNKS="${#GPULIST[@]}"

COMMON_ARGS=(
  -m videollama2.eval.vidhalluc.inference_vidhalluc
  --model-path "${MODEL_PATH}"
  --data-root "${VIDHALLUC_DATA_ROOT}"
  --subset "${SUBSET}"
  --max-samples "${MAX_SAMPLES}"
  --num-frames "${NUM_FRAMES}"
  --sample-timeout-seconds "${SAMPLE_TIMEOUT_SECONDS}"
)

if [[ -n "${BAD_VIDEOS_JSON}" ]]; then
  COMMON_ARGS+=(--bad-videos-json "${BAD_VIDEOS_JSON}")
fi

if [[ "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

if [[ "${USE_FASTV}" == "1" ]]; then
  COMMON_ARGS+=(--use-fastv --fastv-k "${FASTV_K}" --fastv-r "${FASTV_R}")
fi

if [[ "${CHUNKS}" -le 1 ]]; then
  "${PYTHON_BIN}" "${COMMON_ARGS[@]}" --save-path "${SAVE_DIR}"
else
  SHARD_ROOT="${SAVE_DIR}/_shards"
  mkdir -p "${SHARD_ROOT}"

  for IDX in $(seq 0 $((CHUNKS - 1))); do
    SHARD_DIR="${SHARD_ROOT}/shard_${IDX}"
    GPU_DEVICE="${GPULIST[$IDX]}"
    mkdir -p "${SHARD_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
      "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
      --save-path "${SHARD_DIR}" \
      --num-chunks "${CHUNKS}" \
      --chunk-idx "${IDX}" &
  done

  wait

  MERGE_ARGS=(
    -m videollama2.eval.vidhalluc.merge
    --input-glob "${SHARD_ROOT}/shard_*"
    --save-path "${SAVE_DIR}"
    --model-path "${MODEL_PATH}"
    --data-root "${VIDHALLUC_DATA_ROOT}"
    --subset "${SUBSET}"
    --max-samples "${MAX_SAMPLES}"
    --num-frames "${NUM_FRAMES}"
  )

  if [[ "${USE_FASTV}" == "1" ]]; then
    MERGE_ARGS+=(--use-fastv --fastv-k "${FASTV_K}" --fastv-r "${FASTV_R}")
  fi

  "${PYTHON_BIN}" "${MERGE_ARGS[@]}"
fi

echo "VidHalluc run complete: ${SAVE_DIR}"
