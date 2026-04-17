# VideoLLaMA2 + FastV on VidHalluc

This setup adds a VidHalluc evaluator and a repo-local FastV path for `DAMO-NLP-SG/VideoLLaMA2-7B-16F`.

For the full RunPod walkthrough on a single RTX 3090, see [RUNPOD_3090_VIDHALLUC_FASTV_STEP_BY_STEP.md](./RUNPOD_3090_VIDHALLUC_FASTV_STEP_BY_STEP.md).

## Install

```bash
cd /workspace/VideoLLaMA2
pip install -e .
pip install flash-attn==2.5.8 --no-build-isolation
```

## Model and Data

- Recommended checkpoint: `DAMO-NLP-SG/VideoLLaMA2-7B-16F`
- Set `VIDHALLUC_DATA_ROOT` to the VidHalluc root containing:
  - `ach_binaryqa.json`
  - `ach_mcq.json`
  - `sth.json`
  - `tsh.json`
  - `data/{ACH,STH,TSH}/`
- The evaluator automatically skips the three known-bad ACH videos:
  - `eXMF6Skt2To_clip_3.mp4`
  - `KVaTsulE5Z0_clip_2.mp4`
  - `KVaTsulE5Z0_clip_3.mp4`

## Single Run

Preflight scan for corrupted/stalling videos:

```bash
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
SUBSET=ach_binaryqa \
OUTPUT_JSON=/workspace/VidHalluc/bad_videos_ach_binaryqa.json \
VALIDATE_TIMEOUT_SECONDS=20 \
bash scripts/eval/scan_vidhalluc_videos.sh
```

Then pass the generated denylist into the evaluator:

Baseline:

```bash
MODEL_PATH=DAMO-NLP-SG/VideoLLaMA2-7B-16F \
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
BAD_VIDEOS_JSON=/workspace/VidHalluc/bad_videos_ach_binaryqa.json \
RUN_NAME=baseline \
USE_FASTV=0 \
bash scripts/eval/eval_vidhalluc_fastv.sh
```

Default FastV:

```bash
MODEL_PATH=DAMO-NLP-SG/VideoLLaMA2-7B-16F \
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
BAD_VIDEOS_JSON=/workspace/VidHalluc/bad_videos_ach_binaryqa.json \
RUN_NAME=fastv_default \
USE_FASTV=1 \
FASTV_K=3 \
FASTV_R=0.5 \
bash scripts/eval/eval_vidhalluc_fastv.sh
```

## All Experiments

Runs:

- baseline
- FastV default: `K=3, R=0.5`
- FastV sweep: `R=0.25`, `R=0.75`

```bash
MODEL_PATH=DAMO-NLP-SG/VideoLLaMA2-7B-16F \
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
BAD_VIDEOS_JSON=/workspace/VidHalluc/bad_videos_ach_binaryqa.json \
bash scripts/eval/eval_vidhalluc_fastv_all.sh
```

Outputs are written under:

```text
outputs/vidhalluc/<ckpt>/ach_binaryqa_3000/
```

Each run directory contains:

- `predictions.jsonl`
- `summary.json`
- `run_meta.json`

The all-in-one script also writes:

- `matched_3000_comparison.md`

## RunPod Notes

- `3090 24GB`: baseline and default FastV should fit for `7B-16F`, especially when using one GPU per shard.
- `A100`: best option if you want faster turnaround or larger parallel sweeps.
- Multi-GPU runs follow the existing chunking pattern:
  - one worker per visible GPU
  - shard outputs under `_shards/`
  - automatic merge back into the final run directory
- If a sample appears to stall on video decode, set `SAMPLE_TIMEOUT_SECONDS` to skip it after a wall-clock limit:

```bash
MODEL_PATH=DAMO-NLP-SG/VideoLLaMA2-7B-16F \
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
SAMPLE_TIMEOUT_SECONDS=120 \
bash scripts/eval/eval_vidhalluc_fastv.sh
```

- Timed-out samples are written to `predictions.jsonl` with:
  - `"skipped": true`
  - `"error": "sample timed out after ... seconds"`
- For decoder hangs that do not respect the Python timeout, run the preflight scan first and use `BAD_VIDEOS_JSON=...` so those files are skipped before inference starts.
