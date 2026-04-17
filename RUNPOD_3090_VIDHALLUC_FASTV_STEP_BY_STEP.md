# RunPod 3090 Step-By-Step: VideoLLaMA2 + FastV on VidHalluc

This is the full end-to-end guide for running the `VideoLLaMA2-7B-16F` baseline and FastV VidHalluc experiments on a **single RunPod RTX 3090 24GB** pod.

This guide assumes:

- you want to run the code from your `VideoLLaMA2` fork that already contains the FastV + VidHalluc changes
- you want the **same main subset** as the PruneVid experiment:
  - `ach_binaryqa`
  - `max_samples=3000`
- you want to run all three experiment groups:
  - baseline
  - FastV default: `K=3, R=0.5`
  - FastV sweep: `R=0.25` and `R=0.75`

The final all-in-one command is:

```bash
MODEL_PATH=/workspace/models/VideoLLaMA2-7B-16F \
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
OUTPUT_ROOT=/workspace/outputs/vidhalluc \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv_all.sh
```

But do **not** jump straight to that. Follow the steps below in order.

---

## 1. Create the RunPod pod

Recommended settings:

1. Log into RunPod.
2. Click `Deploy`.
3. Choose:
   - `Community Cloud`
   - `RTX 3090`
   - an official `PyTorch` template
4. Recommended disk:
   - `Container Disk`: `20 GB`
   - `Volume Disk`: `80 GB` minimum
   - `100 GB` is safer if you want more breathing room
5. Use **On-Demand**, not Spot, for long runs.
6. Start the pod.

Why `80-100 GB`:

- VideoLLaMA2 checkpoint: roughly `14 GB`
- VidHalluc zips + extracted videos: roughly `45-50 GB` at peak before zip cleanup
- pip/build/cache/output buffer: several extra GB

---

## 2. Open a terminal in the pod

Once the pod is running:

1. Click `Connect`
2. Open `Jupyter Lab`
3. Open a new terminal

Everything below is meant to be run inside that terminal.

---

## 3. Verify the GPU

Run:

```bash
nvidia-smi
```

You should see the `RTX 3090` listed.

Also check the mounted workspace disk:

```bash
df -h /workspace
```

---

## 4. Install basic Linux packages

Run:

```bash
apt-get update
apt-get install -y git git-lfs tmux unzip ffmpeg libsm6 libxext6 build-essential
git lfs install
```

These help with:

- cloning repos
- long-running `tmux` sessions
- video decoding
- `flash-attn` compilation

---

## 5. Clone your fork

Clone **your fork**, not the original upstream repo, because your fork contains the FastV + VidHalluc implementation.

Example:

```bash
cd /workspace
git clone https://github.com/<your-username>/VideoLLaMA2.git
cd /workspace/VideoLLaMA2
```

Replace `<your-username>` with your real GitHub username.

Confirm the new files exist:

```bash
ls videollama2/eval/vidhalluc
ls scripts/eval | grep vidhalluc
```

You should see:

- `videollama2/eval/vidhalluc/__init__.py`
- `videollama2/eval/vidhalluc/inference_vidhalluc.py`
- `videollama2/eval/vidhalluc/merge.py`
- `videollama2/eval/vidhalluc/compare.py`
- `scripts/eval/eval_vidhalluc_fastv.sh`
- `scripts/eval/eval_vidhalluc_fastv_all.sh`

---

## 6. Upgrade pip tools

Run:

```bash
python3 -m pip install --upgrade pip setuptools wheel ninja packaging
```

---

## 7. Install Python dependencies

### Recommended path

Run:

```bash
cd /workspace/VideoLLaMA2
pip install -e .
MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation
```

Notes:

- `pip install -e .` installs `videollama2` as an editable package
- `flash-attn` may take a few minutes to build
- `MAX_JOBS=4` helps avoid overloading the machine during build

### If `pip install -e .` fails

Use this fallback:

```bash
cd /workspace/VideoLLaMA2
pip install \
  "torch==2.2.0" \
  "torchvision==0.17.0" \
  "transformers==4.40.0" \
  "tokenizers==0.19.1" \
  "accelerate==0.26.1" \
  "peft==0.4.0" \
  "timm==1.0.3" \
  "numpy==1.24.4" \
  "decord==0.6.0" \
  "imageio==2.34.0" \
  "imageio-ffmpeg==0.4.9" \
  "moviepy==1.0.3" \
  "opencv-python==4.6.0.66" \
  "scikit-learn==1.2.2" \
  "huggingface_hub==0.23.4" \
  "sentencepiece==0.1.99" \
  "shortuuid" \
  "einops==0.6.1" \
  "einops-exts==0.0.4" \
  "bitsandbytes==0.43.0" \
  "pydantic>=2.0" \
  "requests"
pip install -e . --no-deps
MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation
```

Only use the fallback if the normal editable install breaks.

---

## 8. Verify the package import

Run:

```bash
cd /workspace/VideoLLaMA2
python3 -c "import videollama2; print('videollama2 import ok')"
python3 -m videollama2.eval.vidhalluc.inference_vidhalluc --help
```

If both work, the code is installed correctly.

---

## 9. Optional but recommended: Hugging Face login

The model and dataset are public, but logging in can help with rate limits.

Run:

```bash
huggingface-cli login
```

Paste your token if prompted.

You can also set a cache location:

```bash
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"
```

---

## 10. Download the model checkpoint

Create a local model directory and download the checkpoint there:

```bash
mkdir -p /workspace/models
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="DAMO-NLP-SG/VideoLLaMA2-7B-16F",
    local_dir="/workspace/models/VideoLLaMA2-7B-16F",
    local_dir_use_symlinks=False,
)
print("model download complete")
PY
```

After it finishes, verify the folder exists:

```bash
du -sh /workspace/models/VideoLLaMA2-7B-16F
ls /workspace/models/VideoLLaMA2-7B-16F
```

For all later commands in this guide, the model path is:

```bash
/workspace/models/VideoLLaMA2-7B-16F
```

---

## 11. Download the VidHalluc dataset

Run:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="chaoyuli/VidHalluc",
    repo_type="dataset",
    local_dir="/workspace/VidHalluc",
    local_dir_use_symlinks=False,
)
print("dataset download complete")
PY
```

Check that the annotation files are present:

```bash
ls /workspace/VidHalluc
```

You should see:

- `ach_binaryqa.json`
- `ach_mcq.json`
- `sth.json`
- `tsh.json`

---

## 12. Extract the dataset zips

VidHalluc ships its videos in zip files with inconsistent internal layouts. Run this exact extraction script:

```bash
python3 - <<'PY'
import glob
import os
import shutil
import zipfile

zip_paths = sorted(glob.glob('/workspace/VidHalluc/data/*_videos.zip'))
print('found zip files:', zip_paths)

for z in zip_paths:
    name = os.path.basename(z).replace('_videos.zip', '')
    dst = f'/workspace/VidHalluc/data/{name}'
    os.makedirs(dst, exist_ok=True)
    print(f'extracting {z} -> {dst}')
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dst)

for name in ('ACH', 'STH', 'TSH'):
    dst = f'/workspace/VidHalluc/data/{name}'
    mp4_dir = None
    for dirpath, _, filenames in os.walk(dst):
        if any(f.endswith('.mp4') for f in filenames):
            mp4_dir = dirpath
            break
    if mp4_dir is not None and mp4_dir != dst:
        for fname in os.listdir(mp4_dir):
            if fname.endswith('.mp4'):
                src = os.path.join(mp4_dir, fname)
                out = os.path.join(dst, fname)
                if src != out:
                    shutil.move(src, out)

    for dirpath, _, _ in os.walk(dst, topdown=False):
        if dirpath != dst:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass

    count = sum(1 for f in os.listdir(dst) if f.endswith('.mp4'))
    print(f'{name}: {count} mp4 files')
PY
```

Expected counts:

- `ACH`: about `3933`
- `STH`: about `445`
- `TSH`: about `600`

The evaluator already knows how to skip the three known bad ACH filenames, so you do **not** need to manually move them out for this setup.

---

## 13. Remove the zip files after extraction

This frees a lot of disk space.

Run:

```bash
rm -f /workspace/VidHalluc/data/*_videos.zip
df -h /workspace
```

---

## 14. Set the key environment variables

Run:

```bash
export HF_HOME=/workspace/.cache/huggingface
export MODEL_PATH=/workspace/models/VideoLLaMA2-7B-16F
export VIDHALLUC_DATA_ROOT=/workspace/VidHalluc
export OUTPUT_ROOT=/workspace/outputs/vidhalluc
export CUDA_VISIBLE_DEVICES=0
```

Create the output root:

```bash
mkdir -p "$OUTPUT_ROOT"
```

---

## 15. Verify the dataset layout one last time

Run:

```bash
python3 - <<'PY'
import os

root = "/workspace/VidHalluc"
for name in ["ach_binaryqa.json", "ach_mcq.json", "sth.json", "tsh.json"]:
    path = os.path.join(root, name)
    print(path, os.path.exists(path))

for subset in ["ACH", "STH", "TSH"]:
    d = os.path.join(root, "data", subset)
    count = len([f for f in os.listdir(d) if f.endswith(".mp4")])
    print(subset, count)
PY
```

If any of those files or directories are missing, stop and fix that before running evaluation.

---

## 16. Run a smoke test first

Do **not** start with the full 3000-example run. First do a 5-example smoke test.

Run:

```bash
cd /workspace/VideoLLaMA2
MODEL_PATH="$MODEL_PATH" \
VIDHALLUC_DATA_ROOT="$VIDHALLUC_DATA_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
MAX_SAMPLES=5 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv_all.sh
```

This should run:

- baseline
- FastV default
- FastV `R=0.25`
- FastV `R=0.75`

on just 5 samples.

---

## 17. Check the smoke test outputs

Run:

```bash
find /workspace/outputs/vidhalluc -maxdepth 4 -type f | sort
```

You should see a structure like:

```text
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/baseline/predictions.jsonl
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/baseline/summary.json
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/baseline/run_meta.json
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/fastv_default/predictions.jsonl
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/fastv_default/summary.json
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/fastv_default/run_meta.json
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/fastv_r_0.25/...
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/fastv_r_0.75/...
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/matched_5_comparison.md
```

Also inspect one summary:

```bash
cat /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/baseline/summary.json
```

And inspect the comparison file:

```bash
cat /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_5/matched_5_comparison.md
```

If the smoke test fails, do **not** move on to the full run yet.

---

## 18. Start a tmux session for the full run

The full run is long enough that you should use `tmux`.

Run:

```bash
tmux new -s vidhalluc_fastv
```

You are now inside a `tmux` session.

---

## 19. Launch the full 3000-example run

Inside the `tmux` session, run:

```bash
cd /workspace/VideoLLaMA2
MODEL_PATH="$MODEL_PATH" \
VIDHALLUC_DATA_ROOT="$VIDHALLUC_DATA_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
MAX_SAMPLES=3000 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv_all.sh 2>&1 | tee /workspace/vidhalluc_fastv_3000.log
```

This is the main experiment you want.

It will create:

- baseline on `ach_binaryqa_3000`
- FastV default on `ach_binaryqa_3000`
- FastV `R=0.25`
- FastV `R=0.75`
- a matched comparison markdown file

---

## 20. Detach from tmux and leave it running

To detach:

1. Press `Ctrl+B`
2. Release both keys
3. Press `D`

The job will keep running even if you close the browser tab.

To check later:

```bash
tmux ls
```

To reattach:

```bash
tmux attach -t vidhalluc_fastv
```

---

## 21. Monitor progress while it runs

You can monitor the log without reattaching:

```bash
tail -50 /workspace/vidhalluc_fastv_3000.log
```

You can also check current output folders:

```bash
find /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F -maxdepth 3 -type f | sort
```

And line counts:

```bash
wc -l /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/baseline/predictions.jsonl
wc -l /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/fastv_default/predictions.jsonl
wc -l /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/fastv_r_0.25/predictions.jsonl
wc -l /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/fastv_r_0.75/predictions.jsonl
```

---

## 22. Check the final outputs

When the run is finished, your main results should be here:

```text
/workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/
```

Check:

```bash
find /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000 -maxdepth 2 -type f | sort
```

Open the main comparison:

```bash
cat /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/matched_3000_comparison.md
```

Open the baseline summary:

```bash
cat /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/baseline/summary.json
```

Open the default FastV summary:

```bash
cat /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000/fastv_default/summary.json
```

---

## 23. Optional: run just one experiment instead of all of them

### Baseline only

```bash
cd /workspace/VideoLLaMA2
MODEL_PATH="$MODEL_PATH" \
VIDHALLUC_DATA_ROOT="$VIDHALLUC_DATA_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SUBSET=ach_binaryqa \
MAX_SAMPLES=3000 \
NUM_FRAMES=16 \
USE_FASTV=0 \
RUN_NAME=baseline \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv.sh
```

### FastV default only

```bash
cd /workspace/VideoLLaMA2
MODEL_PATH="$MODEL_PATH" \
VIDHALLUC_DATA_ROOT="$VIDHALLUC_DATA_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SUBSET=ach_binaryqa \
MAX_SAMPLES=3000 \
NUM_FRAMES=16 \
USE_FASTV=1 \
FASTV_K=3 \
FASTV_R=0.5 \
RUN_NAME=fastv_default \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv.sh
```

### One custom FastV ratio

```bash
cd /workspace/VideoLLaMA2
MODEL_PATH="$MODEL_PATH" \
VIDHALLUC_DATA_ROOT="$VIDHALLUC_DATA_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SUBSET=ach_binaryqa \
MAX_SAMPLES=3000 \
NUM_FRAMES=16 \
USE_FASTV=1 \
FASTV_K=3 \
FASTV_R=0.25 \
RUN_NAME=fastv_r_0.25 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv.sh
```

---

## 24. Optional: zip the outputs for download

Run:

```bash
cd /workspace
zip -r videollama2_fastv_vidhalluc_3000.zip outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_3000
```

Then download `videollama2_fastv_vidhalluc_3000.zip` from the Jupyter file browser.

---

## 25. Stop the pod when you are done

After downloading your outputs:

1. Go back to RunPod
2. Stop the pod

Do not leave it idling if you are not using it.

---

## Common failure points

### `flash-attn` build fails

Retry with:

```bash
MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation
```

If it still fails, check:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import sys; print(sys.version)"
nvcc --version || true
```

### Import works but the smoke test fails immediately

Check:

```bash
tail -100 /workspace/vidhalluc_fastv_3000.log
```

Or rerun the smoke test directly in the foreground so you can see the traceback.

### `CLIPVisionModel does not support Flash Attention 2.0 yet`

This means the vision tower is still forcing `flash_attention_2`.

Patch it:

```bash
cd /workspace/VideoLLaMA2
python3 - <<'PY'
from pathlib import Path
path = Path("videollama2/model/encoder.py")
text = path.read_text()
text = text.replace('config._attn_implementation = "flash_attention_2"', 'config._attn_implementation = "eager"')
text = text.replace("config._attn_implementation = 'flash_attention_2'", 'config._attn_implementation = "eager"')
path.write_text(text)
print("patched", path)
PY
```

Then rerun the smoke test.

### `IndexKernel.cu:92 ... index out of bounds` during FastV runs

This means the FastV path is hitting a Mistral position/cache mismatch after pruning. The symptom is usually:

- baseline works
- FastV runs produce `0.0000` accuracy
- pruned predictions are mostly failed/skipped rows rather than real answers

Patch it like this:

```bash
cd /workspace/VideoLLaMA2
python3 - <<'PY'
from pathlib import Path

p = Path("videollama2/model/videollama2_mistral.py")
text = p.read_text()
old = """                    hidden_states = hidden_states[:, keep_indices, :]
                    position_ids = keep_indices.unsqueeze(0).to(position_ids.device)
                    if base_attention_mask is not None and base_attention_mask.dim() == 2:
                        base_attention_mask = base_attention_mask[:, keep_indices]
"""
new = """                    hidden_states = hidden_states[:, keep_indices, :]
                    position_ids = torch.arange(
                        hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
                    ).unsqueeze(0)
                    if base_attention_mask is not None and base_attention_mask.dim() == 2:
                        base_attention_mask = None
"""
if old not in text:
    raise SystemExit("first patch anchor not found")
p.write_text(text.replace(old, new))

p = Path("videollama2/eval/vidhalluc/inference_vidhalluc.py")
text = p.read_text()
old = "            use_cache=True,\\n"
new = '            use_cache=not bool(getattr(model.config, "use_fastv", False)),\\n'
if old not in text:
    raise SystemExit("second patch anchor not found")
p.write_text(text.replace(old, new, 1))

print("patched FastV stability fix")
PY
```

After patching, rerun with a clean smoke-test directory, for example:

```bash
rm -rf /workspace/outputs/vidhalluc/VideoLLaMA2-7B-16F/ach_binaryqa_10
cd /workspace/VideoLLaMA2
MODEL_PATH=/workspace/models/VideoLLaMA2-7B-16F \
VIDHALLUC_DATA_ROOT=/workspace/VidHalluc \
OUTPUT_ROOT=/workspace/outputs/vidhalluc \
MAX_SAMPLES=10 \
RESUME=0 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/eval_vidhalluc_fastv_all.sh
```

### Dataset looks wrong

Re-run the extraction script and verify the final counts again.

### The run is interrupted

The inference script supports `--resume`, and the shell scripts already pass resume through. Re-running the same command should continue from existing `predictions.jsonl` output.

---

## Final checklist

Before the full run, confirm all of these are true:

- `nvidia-smi` shows an `RTX 3090`
- `videollama2` imports successfully
- `/workspace/models/VideoLLaMA2-7B-16F` exists
- `/workspace/VidHalluc/ach_binaryqa.json` exists
- `/workspace/VidHalluc/data/ACH` exists and contains mp4 files
- the 5-example smoke test succeeds
- `tmux` is installed
- you launch the full run inside `tmux`

If all of those are true, you are ready to run the full experiment.
