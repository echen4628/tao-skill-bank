#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure one-rank AnomalyGenNext row throughput from its timestamped log."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any


TIMESTAMP = r"(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
SAMPLE_START = re.compile(
    rf"\[{TIMESTAMP}\|.*generate_samples_from_batch\] Using sampler:"
)
SDG_DONE = re.compile(
    rf"\[{TIMESTAMP}\|.*generate\.py:612:main\] "
    r"SDG done: (?P<count>\d+) images across (?P<ranks>\d+) rank\(s\) "
    r"in (?P<wall_seconds>[0-9.]+)s generation wall-time\."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_time(value: str, *, year: int = 2000) -> datetime:
    return datetime.strptime(f"{year}-{value}", "%Y-%m-%d %H:%M:%S")


def _elapsed(start: datetime, end: datetime) -> float:
    if end < start:
        end += timedelta(days=366 if start.year % 4 == 0 else 365)
    return (end - start).total_seconds()


def analyze(
    generation_log: Path,
    generation_csv: Path,
    provenance_jsonl: Path,
    defect_spec: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    log_text = generation_log.read_text(errors="replace")
    starts = [_parse_time(match.group("timestamp")) for match in SAMPLE_START.finditer(log_text)]
    done_matches = list(SDG_DONE.finditer(log_text))
    if len(done_matches) != 1:
        raise ValueError(f"expected one SDG completion record, found {len(done_matches)}")
    done_match = done_matches[0]
    done_time = _parse_time(done_match.group("timestamp"))
    ranks = int(done_match.group("ranks"))
    if ranks != 1:
        raise ValueError("per-row timing requires one generation rank")

    provenance = _read_jsonl(provenance_jsonl)
    generated = int(done_match.group("count"))
    with generation_csv.open(newline="") as handle:
        generated_rows = list(csv.DictReader(handle))
    if generated != len(generated_rows) or generated > len(provenance):
        raise ValueError(
            "generation CSV/provenance cardinality mismatch: "
            f"generated={generated} csv={len(generated_rows)} provenance={len(provenance)}"
        )
    placement = {
        str(row["defect_type"]): str(row["spatial_dependency"])
        for row in _read_jsonl(defect_spec)
    }

    generation_end_epoch = generation_csv.stat().st_mtime
    previous_completion = generation_end_epoch - float(done_match.group("wall_seconds"))
    rows: list[dict[str, Any]] = []
    for output_order, generated_row in enumerate(generated_rows):
        input_index = int(generated_row["index"])
        provenance_row = provenance[input_index]
        output_image = generation_csv.parent / "reconstructed_image" / generated_row[
            "output_filename"
        ]
        if not output_image.is_file():
            raise FileNotFoundError(output_image)
        completion = output_image.stat().st_mtime
        anomaly_type = str(provenance_row["anomaly_type"])
        if anomaly_type not in placement:
            raise ValueError(f"missing spatial dependency for {anomaly_type}")
        rows.append(
            {
                "generation_index": provenance_row.get("generation_index", input_index),
                "dataset_id": provenance_row["dataset_id"],
                "anomaly_type": anomaly_type,
                "spatial_dependency": placement[anomaly_type],
                "mask_branch": provenance_row["mask_branch"],
                "fn_id": provenance_row["fn_id"],
                "pair_id": provenance_row["pair_id"],
                "output_filename": generated_row["output_filename"],
                "completion_epoch_seconds": completion,
                "seconds_to_output": completion - previous_completion,
                "warmup_row": output_order == 0,
            }
        )
        previous_completion = completion

    groups: dict[str, list[float]] = defaultdict(list)
    groups_without_warmup: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        mode = str(row["spatial_dependency"])
        seconds = float(row["seconds_to_output"])
        groups[mode].append(seconds)
        if not row["warmup_row"]:
            groups_without_warmup[mode].append(seconds)

    def summarize(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "mean_seconds": mean(values) if values else None,
            "median_seconds": median(values) if values else None,
            "min_seconds": min(values) if values else None,
            "max_seconds": max(values) if values else None,
        }

    summary = {
        "schema_version": 1,
        "measurement": "elapsed between reconstructed-image completion timestamps; first row uses reported generation-wall start",
        "attempted_rows": len(provenance),
        "generated_images": generated,
        "sampler_calls": len(starts),
        "generation_wall_seconds_reported": float(done_match.group("wall_seconds")),
        "post_last_image_finalize_seconds": generation_end_epoch - previous_completion,
        "rank_count": ranks,
        "all_rows": summarize([float(row["seconds_to_output"]) for row in rows]),
        "by_spatial_dependency": {
            mode: {
                "including_warmup": summarize(values),
                "excluding_first_job_row": summarize(groups_without_warmup[mode]),
            }
            for mode, values in sorted(groups.items())
        },
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-log", required=True)
    parser.add_argument("--generation-csv")
    parser.add_argument("--provenance-jsonl", required=True)
    parser.add_argument("--defect-spec", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    generation_log = Path(args.generation_log)
    provenance_jsonl = Path(args.provenance_jsonl)
    generation_csv = Path(args.generation_csv) if args.generation_csv else None
    if generation_csv is None:
        provenance_rows = _read_jsonl(provenance_jsonl)
        if not provenance_rows:
            raise ValueError("cannot infer generation CSV from empty provenance")
        dataset_id = str(provenance_rows[0]["dataset_id"])
        generation_csv = (
            generation_log.parent.parent
            / "generation"
            / dataset_id
            / "raw"
            / "texture_ft_generation_result.csv"
        )
    rows, summary = analyze(
        generation_log,
        generation_csv,
        provenance_jsonl,
        Path(args.defect_spec),
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"generation timing PASS: rows={len(rows)} "
        f"wall_seconds={summary['generation_wall_seconds_reported']} "
        f"output={output_summary}"
    )


if __name__ == "__main__":
    main()
