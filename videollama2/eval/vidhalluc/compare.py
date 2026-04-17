from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline predictions.jsonl or run directory")
    parser.add_argument("--pruned", required=True, help="FastV predictions.jsonl or run directory")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve_predictions_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "predictions.jsonl")
    return path


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(resolve_predictions_path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def accuracy(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record.get("correct")) / len(records)


def yes_rate(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if str(record.get("pred", "")).strip().lower() == "yes") / len(records)


def fmt_delta(base: float, new: float) -> str:
    delta = new - base
    return f"{delta:+.4f}"


def main():
    args = parse_args()
    baseline_records = load_records(args.baseline)
    pruned_records = load_records(args.pruned)

    baseline_by_id = {record["sample_id"]: record for record in baseline_records}
    pruned_by_id = {record["sample_id"]: record for record in pruned_records}
    matched_ids = sorted(set(baseline_by_id) & set(pruned_by_id))

    matched_baseline = [baseline_by_id[sample_id] for sample_id in matched_ids]
    matched_pruned = [pruned_by_id[sample_id] for sample_id in matched_ids]

    lines: List[str] = []
    lines.append("# VidHalluc matched comparison")
    lines.append("")
    lines.append(f"- baseline raw: {len(baseline_records)}")
    lines.append(f"- pruned raw: {len(pruned_records)}")
    lines.append(f"- matched on sample_id: {len(matched_ids)}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | baseline | pruned | delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| accuracy | {accuracy(matched_baseline):.4f} | {accuracy(matched_pruned):.4f} | "
        f"{fmt_delta(accuracy(matched_baseline), accuracy(matched_pruned))} |"
    )
    if matched_baseline and matched_baseline[0]["subset"] in ("ach_binaryqa", "sth"):
        lines.append(
            f"| yes_rate | {yes_rate(matched_baseline):.4f} | {yes_rate(matched_pruned):.4f} | "
            f"{fmt_delta(yes_rate(matched_baseline), yes_rate(matched_pruned))} |"
        )

    gold_values = sorted({str(record["gold"]) for record in matched_baseline})
    if len(gold_values) > 1:
        lines.append("")
        lines.append("## Per-gold")
        lines.append("")
        lines.append("| gold | n | baseline_acc | pruned_acc | delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for gold in gold_values:
            base_gold = [record for record in matched_baseline if str(record["gold"]) == gold]
            pruned_gold = [record for record in matched_pruned if str(record["gold"]) == gold]
            lines.append(
                f"| {gold} | {len(base_gold)} | {accuracy(base_gold):.4f} | {accuracy(pruned_gold):.4f} | "
                f"{fmt_delta(accuracy(base_gold), accuracy(pruned_gold))} |"
            )

    flips = 0
    no_to_yes = 0
    yes_to_no = 0
    for sample_id in matched_ids:
        base = baseline_by_id[sample_id]
        pruned = pruned_by_id[sample_id]
        if base.get("pred") != pruned.get("pred"):
            flips += 1
            if str(base.get("pred")).strip().lower() == "no" and str(pruned.get("pred")).strip().lower() == "yes":
                no_to_yes += 1
            if str(base.get("pred")).strip().lower() == "yes" and str(pruned.get("pred")).strip().lower() == "no":
                yes_to_no += 1

    lines.append("")
    lines.append("## Disagreements")
    lines.append("")
    lines.append(f"- Predictions that flipped: {flips} / {len(matched_ids)} ({(100.0 * flips / len(matched_ids)) if matched_ids else 0.0:.2f}%)")
    lines.append(f"- baseline No -> pruned Yes: {no_to_yes}")
    lines.append(f"- baseline Yes -> pruned No: {yes_to_no}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
