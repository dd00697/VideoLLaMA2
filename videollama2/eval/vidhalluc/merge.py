from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

from .inference_vidhalluc import FIXED_DECODING, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True, help="Glob of shard directories containing predictions.jsonl")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--use-fastv", action="store_true")
    parser.add_argument("--fastv-k", type=int, default=3)
    parser.add_argument("--fastv-r", type=float, default=0.5)
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    shard_dirs = sorted(glob.glob(args.input_glob))
    merged_records: List[Dict[str, Any]] = []
    shard_summaries: List[str] = []
    missing_videos: Dict[str, int] = {}
    excluded_videos: Dict[str, int] = {}
    for shard_dir in shard_dirs:
        pred_path = os.path.join(shard_dir, "predictions.jsonl")
        if not os.path.exists(pred_path):
            continue
        merged_records.extend(load_records(pred_path))
        summary_path = os.path.join(shard_dir, "summary.json")
        if os.path.exists(summary_path):
            shard_summaries.append(summary_path)
            shard_summary = load_json(summary_path)
            shard_config = shard_summary.get("config", {})
            for subset, count in (shard_config.get("missing_videos") or {}).items():
                missing_videos[subset] = missing_videos.get(subset, 0) + int(count)
            for subset, count in (shard_config.get("excluded_videos") or {}).items():
                excluded_videos[subset] = excluded_videos.get(subset, 0) + int(count)

    pred_out = os.path.join(args.save_path, "predictions.jsonl")
    with open(pred_out, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(merged_records)
    summary["config"] = {
        "model_path": args.model_path,
        "num_frames": args.num_frames,
        "decoding": FIXED_DECODING,
        "subset": args.subset,
        "max_samples": args.max_samples,
        "use_fastv": bool(args.use_fastv),
        "fastv_k": args.fastv_k,
        "fastv_r": args.fastv_r,
        "data_root": args.data_root,
        "missing_videos": missing_videos,
        "excluded_videos": excluded_videos,
        "merged_from": shard_dirs,
        "shard_summaries": shard_summaries,
    }
    with open(os.path.join(args.save_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cwd": os.getcwd(), "argv": sys.argv}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
