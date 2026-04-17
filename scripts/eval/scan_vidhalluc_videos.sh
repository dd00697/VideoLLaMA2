#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VIDHALLUC_DATA_ROOT="${VIDHALLUC_DATA_ROOT:?VIDHALLUC_DATA_ROOT must be set}"
SUBSET="${SUBSET:-ach_binaryqa}"
OUTPUT_JSON="${OUTPUT_JSON:-${VIDHALLUC_DATA_ROOT}/bad_videos_${SUBSET}.json}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
VALIDATE_TIMEOUT_SECONDS="${VALIDATE_TIMEOUT_SECONDS:-20}"

"${PYTHON_BIN}" -m videollama2.eval.vidhalluc.validate_videos \
  --data-root "${VIDHALLUC_DATA_ROOT}" \
  --subset "${SUBSET}" \
  --output-json "${OUTPUT_JSON}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --timeout-seconds "${VALIDATE_TIMEOUT_SECONDS}"

echo "VidHalluc bad-video scan complete: ${OUTPUT_JSON}"
