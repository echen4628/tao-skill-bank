#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the KPI-best main checkpoint or request one DEFT OD AOI extension."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from deft_od_aoi_policy import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--status",
        action="append",
        required=True,
        help=(
            "RT-DETR JSON-lines status file. Repeat for every allocation or "
            "extension phase so selection covers the complete KPI history."
        ),
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--planned-epochs", type=int, required=True)
    parser.add_argument("--extension-applied", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metric_rows(path: Path) -> list[tuple[int, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = value.get("kpi") if isinstance(value.get("kpi"), dict) else {}
        metric = kpi.get("val_mAP50")
        if metric is None:
            metric = kpi.get("mAP50")
        epoch = value.get("epoch", kpi.get("epoch"))
        if metric is None or epoch is None or not math.isfinite(float(metric)):
            continue
        rows.append((int(epoch), float(metric)))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    if args.planned_epochs < 1:
        raise ValueError("planned_epochs must be positive")
    checkpoints = Path(args.checkpoint_dir).expanduser().resolve()
    status_values = args.status if isinstance(args.status, list) else [args.status]
    statuses = [Path(value).expanduser().resolve() for value in status_values]
    if not statuses or any(not status.is_file() for status in statuses):
        raise FileNotFoundError("one or more status files are missing")
    if not checkpoints.is_dir():
        raise FileNotFoundError("checkpoint directory is missing")
    rows = [row for status in statuses for row in metric_rows(status)]
    if not rows:
        raise ValueError("main training status has no KPI mAP50 metrics")
    best_epoch, best_metric = max(rows, key=lambda row: (row[1], -row[0]))
    selected = checkpoints / f"model_epoch_{best_epoch:03d}.pth"
    if not selected.is_file():
        raise FileNotFoundError(f"KPI-best checkpoint is missing: {selected}")
    training = policy["training"]
    cutoff = args.planned_epochs - int(training["extension_trigger_last_n_epochs"])
    needs_extension = best_epoch >= cutoff and not args.extension_applied
    report: dict[str, Any] = {
        "action": "extend" if needs_extension else "select",
        "best_epoch": best_epoch,
        "best_kpi_mAP50": best_metric,
        "selected_checkpoint": str(selected),
        "planned_epochs": args.planned_epochs,
        "extension_applied": bool(args.extension_applied),
        "status_files": [str(status) for status in statuses],
    }
    if needs_extension:
        available = sorted(checkpoints.glob("model_epoch_*.pth"))
        if not available:
            raise FileNotFoundError("no checkpoint is available for extension resume")
        report.update(
            {
                "extension_epochs": int(training["extension_epochs"]),
                "extended_num_epochs": args.planned_epochs
                + int(training["extension_epochs"]),
                "resume_checkpoint": str(available[-1].resolve()),
            }
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main() -> int:
    try:
        report = run(parse_args())
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
