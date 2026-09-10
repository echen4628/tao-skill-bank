#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select one of three completed YOLO probes using KPI validation AP50 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


METRIC = "metrics/mAP50(B)"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(manifest_path: Path, selections: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen winner: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("probe_candidates") or []
    if manifest.get("phase") != "probe" or len(candidates) != 3 or len(selections) != 3:
        raise ValueError("YOLO probe selection requires one manifest and exactly three results")
    scored = []
    for index, (candidate, path) in enumerate(zip(candidates, selections, strict=True)):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("selection_metric") != METRIC:
            raise ValueError(f"probe {index} did not report {METRIC}")
        score = float(result.get("metric_value"))
        if not math.isfinite(score):
            raise ValueError(f"probe {index} metric is not finite")
        scored.append({
            "index": index,
            "name": candidate["name"],
            "metric_value": score,
            "selected_epoch_reported": int(result["selected_epoch_reported"]),
            "selection_path": str(path.resolve()),
            "selection_sha256": _sha256(path),
        })
    winner = max(scored, key=lambda row: (row["metric_value"], -row["index"]))
    candidate = candidates[winner["index"]]
    value = {
        "schema_version": 1,
        "iteration": int(manifest["iteration"]),
        "selected_by": "kpi_validation_ap50",
        "selection_metric": METRIC,
        "test_used_for_selection": False,
        "fresh_standard_base_for_main": True,
        "winner": winner,
        "scores": scored,
        "overrides": {
            "lr0": float(candidate["lr0"]),
            "lrf": float(candidate["lrf"]),
            "weight_decay": float(candidate["weight_decay"]),
            "seed": int((manifest.get("probe_seed") or 4000 + int(manifest["iteration"]))),
        },
        "probe_manifest": str(manifest_path.resolve()),
        "probe_manifest_sha256": _sha256(manifest_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.manifest.resolve(), [p.resolve() for p in args.selection],
                            args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
