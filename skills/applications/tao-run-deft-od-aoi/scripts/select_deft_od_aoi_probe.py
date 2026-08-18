#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the KPI-best DEFT OD AOI probe and finalize the main train spec."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from deft_od_aoi_policy import load_policy, main_epoch_budget, probes_active
from write_rtdetr_specs import set_nested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--probe-manifest", required=True)
    parser.add_argument("--train-spec", required=True)
    parser.add_argument("--winner-output", required=True)
    parser.add_argument("--best-config-output", required=True)
    parser.add_argument("--history-output", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def best_metric(status_path: Path) -> float | None:
    if not status_path.is_file():
        return None
    best: float | None = None
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = value.get("kpi") if isinstance(value.get("kpi"), dict) else {}
        metric = kpi.get("val_mAP50")
        if metric is None:
            metric = kpi.get("mAP50")
        if metric is not None and math.isfinite(float(metric)):
            best = float(metric) if best is None else max(best, float(metric))
    return best


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    manifest = read_json(args.probe_manifest)
    iteration = int(manifest.get("iteration", 0))
    if not probes_active(policy, iteration):
        raise ValueError("probes are disabled; skip select_deft_od_aoi_probe.py")
    probes = manifest.get("probes")
    if not isinstance(probes, list) or len(probes) != int(policy["training"]["probe_count"]):
        raise ValueError("probe manifest does not contain the frozen probe count")
    scored = []
    for probe in probes:
        status = Path(probe["results_dir"]).expanduser().resolve() / "train" / "status.json"
        scored.append({**probe, "probe_mAP50": best_metric(status)})
    survivors = [probe for probe in scored if probe["probe_mAP50"] is not None]
    if not survivors:
        raise ValueError("no probe produced a KPI mAP50 metric")
    winner = max(survivors, key=lambda probe: (probe["probe_mAP50"], -probe["probe"]))

    train_spec_path = Path(args.train_spec).expanduser().resolve()
    train_spec = yaml.safe_load(train_spec_path.read_text(encoding="utf-8"))
    for dotted, value in winner["deltas"].items():
        set_nested(train_spec, dotted, value)
    train_size = int(manifest["train_size"])
    budget = main_epoch_budget(policy, iteration, train_size)
    train_spec["train"]["num_epochs"] = budget
    temporary = train_spec_path.with_suffix(train_spec_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(train_spec, sort_keys=False), encoding="utf-8")
    os.replace(temporary, train_spec_path)

    winner_report = {**winner, "num_epochs": budget, "results": scored}
    atomic_json(Path(args.winner_output).expanduser().resolve(), winner_report)
    atomic_json(Path(args.best_config_output).expanduser().resolve(), winner["deltas"])
    history_path = Path(args.history_output).expanduser().resolve()
    history = read_json(history_path) if history_path.is_file() else []
    if not isinstance(history, list):
        raise ValueError("history output exists but is not a JSON array")
    history.append(
        {
            "iteration": manifest["iteration"],
            "train_size": train_size,
            "winner": winner["name"],
            "probe_mAP50": winner["probe_mAP50"],
            "num_epochs": budget,
        }
    )
    atomic_json(history_path, history)
    return winner_report


def main() -> int:
    try:
        winner = run(parse_args())
        print(
            f"DEFT OD AOI probe winner={winner['name']} "
            f"mAP50={winner['probe_mAP50']} epochs={winner['num_epochs']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
