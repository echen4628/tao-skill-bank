#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write loose and strict DEFT OD AOI object-detection gap specs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from deft_od_aoi_policy import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--ground-truth-ann-path", required=True)
    parser.add_argument("--inference-ann-path", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kpi", required=True)
    return parser.parse_args()


def _absolute(raw: str) -> str:
    return str(Path(raw).expanduser().resolve())


def build_spec(args: argparse.Namespace, policy: dict, kind: str) -> dict:
    gap = policy["gap"]
    confidence = gap[f"{kind}_confidence"]
    return {
        "ground_truth_ann_path": _absolute(args.ground_truth_ann_path),
        "inference_ann_path": _absolute(args.inference_ann_path),
        "images_dir": _absolute(args.images_dir),
        "results_dir": str(Path(args.output_dir).expanduser().resolve() / kind),
        "kpi": f"{args.kpi}_{kind}",
        "input_format": "kitti",
        "iou_threshold": gap["match_iou"],
        "conf_threshold": confidence,
        "min_area": 0,
        "class_mapping": {},
        "weak_thresholds": {
            policy["task"]["class_name"]: {
                "recall": gap["weak_recall_threshold"],
                "ap50": gap["weak_ap50_threshold"],
            }
        },
        "default_ap50_threshold": 0.0,
        "default_recall_threshold": 0.0,
        "default_precision_threshold": gap["weak_precision_threshold"],
    }


def main() -> int:
    try:
        args = parse_args()
        policy = load_policy(args.policy)
        output = Path(args.output_dir).expanduser().resolve()
        for kind in ("loose", "strict"):
            directory = output / kind
            directory.mkdir(parents=True, exist_ok=True)
            spec_path = directory / "od_gap_spec.yaml"
            spec_path.write_text(
                yaml.safe_dump(build_spec(args, policy, kind), sort_keys=False),
                encoding="utf-8",
            )
            print(f"{kind} gap spec -> {spec_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
