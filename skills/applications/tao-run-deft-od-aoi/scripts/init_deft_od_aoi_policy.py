#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze a validated DEFT OD AOI policy JSON before launch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from deft_od_aoi_policy import build_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument(
        "--synthetic-enabled",
        choices=("true", "false"),
        required=True,
        help="Enable or disable synthetic generation.",
    )
    parser.add_argument(
        "--probes-enabled",
        choices=("true", "false"),
        default=None,
        help="Iteration-3+ LR bake-off. Default true; false skips probes.",
    )
    parser.add_argument(
        "--model-soup-enabled",
        choices=("true", "false"),
        default=None,
        help="Final greedy KPI model soup. Default true; false skips it.",
    )
    parser.add_argument(
        "--training-workers",
        type=int,
        default=3,
        help="RT-DETR training DataLoader workers. Use 0 on clusters with unstable shared-memory IPC.",
    )
    parser.add_argument(
        "--inference-workers",
        type=int,
        default=8,
        help="RT-DETR inference DataLoader workers.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        policy = build_policy(
            max_iterations=args.max_iterations,
            synthetic_enabled=(
                None
                if args.synthetic_enabled is None
                else args.synthetic_enabled == "true"
            ),
            probes_enabled=(
                None
                if args.probes_enabled is None
                else args.probes_enabled == "true"
            ),
            model_soup_enabled=(
                None
                if args.model_soup_enabled is None
                else args.model_soup_enabled == "true"
            ),
            training_workers=args.training_workers,
            inference_workers=args.inference_workers,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != policy:
                raise ValueError(
                    f"refusing to replace frozen policy with different values: {output}"
                )
            print(f"DEFT OD AOI policy already frozen and identical: {output}")
            return 0
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(
            f"DEFT OD AOI policy frozen: iterations={policy['max_iterations']} "
            f"retrieval={policy['retrieval']['mode']} "
            f"probes={policy['training']['probes_enabled']} "
            f"model_soup={policy['model_soup']['enabled']} -> {output}"
        )
        print("SigLIP role-separated retrieval is enabled; uniform mining is disabled.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
