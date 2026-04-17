from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from videollama2 import model_init
from videollama2.constants import DEFAULT_VIDEO_TOKEN
from videollama2.mm_utils import KeywordsStoppingCriteria, tokenizer_multimodal_token
from videollama2.utils import disable_torch_init

from . import (
    SUBSETS,
    VidHallucDataset,
    build_prompt,
    is_correct,
    load_bad_video_basenames,
    parse_answer,
    resolve_default_data_root,
)


FIXED_DECODING = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "max_new_tokens": 32,
}


class SampleTimeoutError(TimeoutError):
    pass


@contextlib.contextmanager
def sample_time_limit(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(signum, frame):
        raise SampleTimeoutError(f"sample timed out after {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default=resolve_default_data_root(repo_root))
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--subset", default="ach_binaryqa", choices=list(SUBSETS) + ["all"])
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--sample-timeout-seconds", type=int, default=0)
    parser.add_argument("--bad-videos-json", default=None)
    parser.add_argument("--use-fastv", action="store_true")
    parser.add_argument("--fastv-k", type=int, default=3)
    parser.add_argument("--fastv-r", type=float, default=0.5)
    return parser.parse_args()


def resolve_subsets(subset: str) -> List[str]:
    return list(SUBSETS) if subset == "all" else [subset]


def load_existing_records(pred_path: str) -> Tuple[set, List[Dict[str, Any]]]:
    if not os.path.exists(pred_path):
        return set(), []
    done_ids = set()
    records: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            sample_id = record.get("sample_id")
            if sample_id is not None:
                done_ids.add(sample_id)
    return done_ids, records


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {"n": 0, "correct": 0, "skipped": 0}
    per_subset: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["subset"], []).append(record)
    for subset, items in grouped.items():
        n = len(items)
        correct = sum(1 for item in items if item.get("correct"))
        skipped = sum(1 for item in items if item.get("skipped"))
        entry = {
            "n": n,
            "correct": correct,
            "accuracy": (correct / n) if n else 0.0,
            "skipped": skipped,
        }
        if subset in ("ach_binaryqa", "sth"):
            yes = sum(1 for item in items if str(item.get("pred", "")).strip().lower() == "yes")
            entry["yes_rate"] = (yes / n) if n else None
        per_subset[subset] = entry
        totals["n"] += n
        totals["correct"] += correct
        totals["skipped"] += skipped
    totals["accuracy"] = (totals["correct"] / totals["n"]) if totals["n"] else 0.0
    return {"totals": totals, "per_subset": per_subset}


def configure_fastv(model, use_fastv: bool, fastv_k: int, fastv_r: float):
    if hasattr(model, "set_fastv_config"):
        model.set_fastv_config(use_fastv, fastv_k, fastv_r)
    else:
        model.config.use_fastv = bool(use_fastv)
        model.config.fastv_k = int(fastv_k)
        model.config.fastv_r = float(fastv_r)


def build_messages(model, instruct: str) -> List[Dict[str, str]]:
    user_message = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + instruct}]
    if model.config.model_type in ["videollama2", "videollama2_mistral", "videollama2_mixtral"]:
        system_message = [
            {
                "role": "system",
                "content": (
                    "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
                    "Always answer as helpfully as possible, while being safe. "
                    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, "
                    "or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n"
                    "If a question does not make any sense, or is not factually coherent, explain why instead of "
                    "answering something not correct. If you don't know the answer to a question, please don't share "
                    "false information.\n<</SYS>>"
                ),
            }
        ]
    else:
        system_message = []
    return system_message + user_message


def generate_response(video_tensor, instruct: str, model, tokenizer) -> str:
    tensor = [(video_tensor.half().cuda(), "video")]
    messages = build_messages(model, instruct)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).long().cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    stopping_criteria = [KeywordsStoppingCriteria([tokenizer.eos_token], tokenizer, input_ids)]

    if bool(getattr(model.config, "use_fastv", False)):
        current_ids = input_ids
        current_attention_mask = attention_mask
        generated_ids: List[torch.Tensor] = []
        eos_token_id = tokenizer.eos_token_id
        max_new_tokens = int(FIXED_DECODING["max_new_tokens"])

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_attention_mask,
                    images=tensor,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token)
                current_ids = torch.cat((current_ids, next_token), dim=1)
                current_attention_mask = torch.cat(
                    (
                        current_attention_mask,
                        torch.ones(
                            (current_attention_mask.shape[0], 1),
                            dtype=current_attention_mask.dtype,
                            device=current_attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break

        if generated_ids:
            generated = torch.cat(generated_ids, dim=1)
            return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return ""

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=tensor,
            use_cache=not bool(getattr(model.config, "use_fastv", False)),
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            **FIXED_DECODING,
        )

    full_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if output_ids.shape[1] > input_ids.shape[1]:
        generated = tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0].strip()
        return generated or full_text
    return full_text


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    disable_torch_init()
    model, processor, tokenizer = model_init(args.model_path)
    configure_fastv(model, args.use_fastv, args.fastv_k, args.fastv_r)
    model.eval()

    extra_bad_video_basenames = load_bad_video_basenames(args.bad_videos_json)

    pred_path = os.path.join(args.save_path, "predictions.jsonl")
    done_ids: set = set()
    existing_records: List[Dict[str, Any]] = []
    write_mode = "w"
    if args.resume:
        done_ids, existing_records = load_existing_records(pred_path)
        write_mode = "a"

    dataset = VidHallucDataset(
        data_root=args.data_root,
        subsets=resolve_subsets(args.subset),
        video_processor=processor["video"],
        max_samples=args.max_samples,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        extra_bad_video_basenames=extra_bad_video_basenames,
    )

    records = list(existing_records)
    with open(pred_path, write_mode, encoding="utf-8") as fout:
        for idx in tqdm(range(len(dataset)), desc="VidHalluc"):
            meta = dataset.data_list[idx]
            if meta["sample_id"] in done_ids:
                continue

            try:
                with sample_time_limit(args.sample_timeout_seconds):
                    example = dataset[idx]
                    raw_output = generate_response(example["video"], build_prompt(example), model, tokenizer)
                    pred = parse_answer(raw_output, example["subset"])
                    correct = is_correct(pred, example["gold"], example["subset"])
                    record = {
                        "subset": example["subset"],
                        "sample_id": example["sample_id"],
                        "video_id": example["video_id"],
                        "video_path": example["video_path"],
                        "question": example["question"],
                        "options": example["options"],
                        "gold": example["gold"],
                        "raw_output": raw_output,
                        "pred": pred,
                        "correct": bool(correct),
                    }
            except Exception as exc:
                record = {
                    "subset": meta["subset"],
                    "sample_id": meta["sample_id"],
                    "video_id": meta["video_id"],
                    "video_path": meta["video_path"],
                    "question": meta["question"],
                    "options": meta["options"],
                    "gold": meta["gold"],
                    "raw_output": "",
                    "pred": None,
                    "correct": False,
                    "skipped": True,
                    "error": str(exc),
                }

            records.append(record)
            done_ids.add(record["sample_id"])
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    summary = summarize(records)
    summary["config"] = {
        "model_path": args.model_path,
        "num_frames": args.num_frames,
        "decoding": FIXED_DECODING,
        "subset": args.subset,
        "subsets": resolve_subsets(args.subset),
        "max_samples": args.max_samples,
        "use_fastv": bool(args.use_fastv),
        "fastv_k": args.fastv_k,
        "fastv_r": args.fastv_r,
        "data_root": args.data_root,
        "bad_videos_json": args.bad_videos_json,
        "bad_video_basenames": extra_bad_video_basenames,
        "missing_videos": dataset.missing_videos,
        "excluded_videos": dataset.excluded_videos,
        "sample_timeout_seconds": args.sample_timeout_seconds,
        "num_chunks": args.num_chunks,
        "chunk_idx": args.chunk_idx,
    }

    with open(os.path.join(args.save_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cwd": os.getcwd(), "argv": sys.argv}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
