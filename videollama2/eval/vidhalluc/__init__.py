"""VidHalluc evaluation helpers for VideoLLaMA2."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset


SUBSETS = ("ach_binaryqa", "ach_mcq", "sth", "tsh")

KNOWN_BAD_ACH_FILENAMES = {
    "eXMF6Skt2To_clip_3.mp4",
    "KVaTsulE5Z0_clip_2.mp4",
    "KVaTsulE5Z0_clip_3.mp4",
}

_SUBSET_VIDEO_DIR = {
    "ach_binaryqa": "ACH",
    "ach_mcq": "ACH",
    "sth": "STH",
    "tsh": "TSH",
}

_SUBSET_FILENAME = {
    "ach_binaryqa": "ach_binaryqa.json",
    "ach_mcq": "ach_mcq.json",
    "sth": "sth.json",
    "tsh": "tsh.json",
}

_BINARY_INSTRUCTION = "Answer the question with only 'Yes' or 'No'."
_MCQ_INSTRUCTION = (
    "Select the best answer from the options below. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
_TSH_INSTRUCTION = (
    "Which of the two actions occurs first in the video? "
    "Answer with two letters indicating the order "
    "(AB if Action A comes first, BA if Action B comes first)."
)

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_MCQ_RE = re.compile(r"\b([ABCD])\b")
_TSH_RE = re.compile(r"\b(AB|BA)\b", re.IGNORECASE)


def split_list(items: Sequence[Any], n: int) -> List[List[Any]]:
    if n <= 1:
        return [list(items)]
    chunk_size = (len(items) + n - 1) // n
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def get_chunk(items: Sequence[Any], n: int, k: int) -> List[Any]:
    chunks = split_list(items, n)
    if k < 0 or k >= len(chunks):
        return []
    return chunks[k]


def resolve_default_data_root(repo_root: str) -> str:
    env = os.environ.get("VIDHALLUC_DATA_ROOT")
    candidates: List[str] = []
    if env:
        candidates.append(env)
    candidates.extend(
        [
            os.path.join(repo_root, "eval", "vidhalluc"),
            os.path.join(repo_root, "..", "VidHalluc"),
            os.path.join(repo_root, "..", "data", "VidHalluc"),
            "/workspace/VidHalluc",
        ]
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(os.path.join(candidate, "ach_binaryqa.json")):
            return candidate
    return os.path.abspath(candidates[1])


def candidate_video_roots(data_root: str, subset: str) -> List[str]:
    name = _SUBSET_VIDEO_DIR[subset]
    return [
        os.path.join(data_root, "data", name),
        os.path.join(data_root, "data", name, name),
        os.path.join(data_root, "data", "VidHalluc", "data", name),
        os.path.join(data_root, name),
    ]


def resolve_video_path(data_root: str, subset: str, video_id: str) -> str:
    for root in candidate_video_roots(data_root, subset):
        candidate = os.path.join(root, f"{video_id}.mp4")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(candidate_video_roots(data_root, subset)[0], f"{video_id}.mp4")


class VidHallucDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subsets: Sequence[str],
        video_processor,
        max_samples: Optional[int] = None,
        num_chunks: int = 1,
        chunk_idx: int = 0,
    ):
        self.data_root = os.path.abspath(data_root)
        self.subsets = list(subsets)
        self.video_processor = video_processor
        self.data_list: List[Dict[str, Any]] = []
        self.missing_videos: Dict[str, int] = {}
        self.excluded_videos: Dict[str, int] = {}

        for subset in self.subsets:
            if subset not in SUBSETS:
                raise ValueError(f"Unknown VidHalluc subset: {subset}")
            getattr(self, f"_load_{subset}")()

        if max_samples is not None and max_samples > 0:
            self.data_list = self.data_list[:max_samples]

        if num_chunks > 1:
            self.data_list = get_chunk(self.data_list, num_chunks, chunk_idx)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = dict(self.data_list[index])
        example["video"] = self.video_processor(example["video_path"])
        return example

    def _append(
        self,
        subset: str,
        video_id: str,
        question: str,
        gold: str,
        options: Optional[Dict[str, str]] = None,
        sample_id: Optional[str] = None,
    ):
        video_path = resolve_video_path(self.data_root, subset, video_id)
        basename = os.path.basename(video_path)
        if subset in ("ach_binaryqa", "ach_mcq") and basename in KNOWN_BAD_ACH_FILENAMES:
            self.excluded_videos[subset] = self.excluded_videos.get(subset, 0) + 1
            return
        if not os.path.exists(video_path):
            self.missing_videos[subset] = self.missing_videos.get(subset, 0) + 1
            return
        if sample_id is None:
            sample_id = f"{subset}:{video_id}:{len(self.data_list)}"
        self.data_list.append(
            {
                "subset": subset,
                "sample_id": sample_id,
                "video_id": video_id,
                "video_path": video_path,
                "question": question,
                "options": options,
                "gold": gold,
            }
        )

    def _load_ach_binaryqa(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["ach_binaryqa"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_id, q_list in data.items():
            if not isinstance(q_list, list):
                continue
            for q_idx, q_entry in enumerate(q_list):
                question = q_entry.get("q", "")
                answers = q_entry.get("a", {}) or {}
                for video_id, gold in answers.items():
                    sample_id = f"ach_binaryqa:{group_id}:{q_idx}:{video_id}"
                    self._append("ach_binaryqa", video_id, question, str(gold), sample_id=sample_id)

    def _load_ach_mcq(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["ach_mcq"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_id, vids in data.items():
            if not isinstance(vids, dict):
                continue
            for video_id, entry in vids.items():
                sample_id = f"ach_mcq:{group_id}:{video_id}"
                self._append(
                    "ach_mcq",
                    video_id,
                    entry.get("Question", ""),
                    str(entry.get("Correct Answer", "")).strip(),
                    options=dict(entry.get("Choices", {}) or {}),
                    sample_id=sample_id,
                )

    def _load_sth(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["sth"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for video_id, entry in data.items():
            self._append(
                "sth",
                video_id,
                "Is there a scene change in this video?",
                str(entry.get("Scene change", "")),
                sample_id=f"sth:{video_id}",
            )

    def _load_tsh(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["tsh"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, entry in data.items():
            video_id = entry.get("video", "")
            self._append(
                "tsh",
                video_id,
                entry.get("Question", ""),
                str(entry.get("Correct Answer", "")).strip().upper(),
                sample_id=f"tsh:{idx}:{video_id}",
            )


def build_prompt(example: Dict[str, Any]) -> str:
    subset = example["subset"]
    question = example["question"]
    if subset in ("ach_binaryqa", "sth"):
        return f"{_BINARY_INSTRUCTION}\nQuestion: {question}"
    if subset == "ach_mcq":
        options = example.get("options") or {}
        option_lines = "\n".join(f"{key}. {value}" for key, value in options.items())
        return f"{_MCQ_INSTRUCTION}\nQuestion: {question}\n{option_lines}"
    if subset == "tsh":
        return f"{question}\n{_TSH_INSTRUCTION}"
    raise ValueError(f"Unknown subset: {subset}")


def parse_answer(raw: str, subset: str) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if subset in ("ach_binaryqa", "sth"):
        match = _YESNO_RE.search(text)
        return None if not match else match.group(1).capitalize()
    if subset == "ach_mcq":
        match = _MCQ_RE.search(text.upper())
        return None if not match else match.group(1)
    if subset == "tsh":
        match = _TSH_RE.search(text.upper())
        return None if not match else match.group(1).upper()
    raise ValueError(f"Unknown subset: {subset}")


def is_correct(pred: Optional[str], gold: str, subset: str) -> bool:
    if pred is None:
        return False
    if subset in ("ach_mcq", "tsh"):
        return pred.strip().upper() == str(gold).strip().upper()
    return pred.strip().lower() == str(gold).strip().lower()
