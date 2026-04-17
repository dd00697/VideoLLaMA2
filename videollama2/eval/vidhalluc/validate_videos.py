from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List

from tqdm import tqdm

from . import SUBSETS, enumerate_unique_videos, resolve_default_data_root


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=resolve_default_data_root(repo_root))
    parser.add_argument("--subset", default="all", choices=list(SUBSETS) + ["all"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--timeout-seconds", type=int, default=20)
    return parser.parse_args()


def resolve_subsets(subset: str) -> List[str]:
    return list(SUBSETS) if subset == "all" else [subset]


def validate_video(ffmpeg_bin: str, video_path: str, timeout_seconds: int) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        return {"ok": False, "error": "missing file"}

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"ffmpeg timeout after {timeout_seconds} seconds"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if result.returncode != 0:
        error_text = (result.stderr or "").strip() or f"ffmpeg exited with code {result.returncode}"
        return {"ok": False, "error": error_text}
    return {"ok": True, "error": None}


def main():
    args = parse_args()
    ffmpeg_bin = shutil.which(args.ffmpeg_bin)
    if ffmpeg_bin is None:
        raise FileNotFoundError(
            f"Could not find ffmpeg binary '{args.ffmpeg_bin}'. Install ffmpeg or pass --ffmpeg-bin."
        )

    subsets = resolve_subsets(args.subset)
    videos = enumerate_unique_videos(os.path.abspath(args.data_root), subsets)

    bad_videos: List[Dict[str, Any]] = []
    seen_basenames = set()

    for video in tqdm(videos, desc="Validate VidHalluc videos"):
        verdict = validate_video(ffmpeg_bin, video["video_path"], args.timeout_seconds)
        if verdict["ok"]:
            continue
        entry = dict(video)
        entry["error"] = verdict["error"]
        bad_videos.append(entry)
        seen_basenames.add(video["basename"])

    payload = {
        "generated_at_epoch": time.time(),
        "data_root": os.path.abspath(args.data_root),
        "subsets": subsets,
        "timeout_seconds": args.timeout_seconds,
        "ffmpeg_bin": ffmpeg_bin,
        "summary": {
            "total_unique_videos": len(videos),
            "bad_unique_videos": len(bad_videos),
        },
        "bad_basenames": sorted(seen_basenames),
        "bad_videos": bad_videos,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote bad-video denylist to: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
