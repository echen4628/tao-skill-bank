#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write frozen nested YOLO train, KPI, and report-only test specs for DEFT AOI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--train-coco", help="Required after iteration 0.")
    parser.add_argument("--train-images", help="Required after iteration 0.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-output-dir")
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--runtime-root",
        default="${TAO_RUNTIME_ROOT}",
        help="Node-local runtime path or an environment placeholder expanded in the job.",
    )
    parser.add_argument(
        "--onelogger-enabled", choices=("true", "false"), required=True
    )
    parser.add_argument("--onelogger-callback-module")
    return parser.parse_args()


def _absolute_file(value: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _absolute_dir(value: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return str(path)


def _identity(value: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def _freeze_yaml(path: Path, value: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(value, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"refusing to replace different frozen spec: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _freeze_json(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"refusing to replace different manifest: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def build_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    policy = yaml.safe_load(Path(args.policy).expanduser().resolve().read_text())
    if not isinstance(policy, dict):
        raise ValueError("policy must be a YAML mapping")
    model = policy.get("model") or {}
    if model.get("backend") != "yolo":
        raise ValueError("write_yolo_specs requires a frozen YOLO policy")
    if not str(model.get("architecture") or "").startswith("yolo"):
        raise ValueError("model.architecture must identify YOLO")
    if not 0 <= args.iteration <= int(policy["max_iterations"]):
        raise ValueError("iteration is outside the frozen policy range")
    if args.iteration > 0 and (not args.train_coco or not args.train_images):
        raise ValueError("--train-coco and --train-images are required after iteration 0")
    train_coco = _absolute_file(args.train_coco) if args.train_coco else None
    train_images = _absolute_dir(args.train_images) if args.train_images else None
    sources = policy.get("sources") or {}
    kpi_coco = _absolute_file(str((sources.get("kpi") or {}).get("coco") or ""))
    kpi_images = _absolute_dir(str((sources.get("kpi") or {}).get("images") or ""))
    test_coco = _absolute_file(str((sources.get("test") or {}).get("coco") or ""))
    test_images = _absolute_dir(str((sources.get("test") or {}).get("images") or ""))
    checkpoint = _absolute_file(str(policy.get("base_checkpoint") or ""))
    yolo = policy.get("yolo") or {}
    training = yolo.get("training") or {}
    inference = yolo.get("evaluation") or {}
    results = _identity(args.results_root) / f"iteration_{args.iteration}"
    runtime = args.runtime_root.rstrip("/")
    onelogger_enabled = args.onelogger_enabled == "true"
    callback = (args.onelogger_callback_module or "").strip()
    if onelogger_enabled and not callback:
        raise ValueError("enabled OneLogger requires --onelogger-callback-module")
    logging = {
        "onelogger_enabled": onelogger_enabled,
        **({"callback_module": callback} if callback else {}),
    }
    train = {
        "model": {
            "architecture": model["architecture"],
            "checkpoint": checkpoint,
        },
        "dataset": {
            "train_coco": train_coco,
            "train_images": train_images,
            "eval_coco": kpi_coco,
            "eval_images": kpi_images,
        },
        "train": {
            "num_gpus": training["num_gpus"],
            "epochs": int(training["epochs"]),
            "patience": training["patience"],
            "imgsz": training["imgsz"],
            "batch": training["batch_size"],
            "nbs": training["nbs"],
            "devices": training["devices"],
            "workers": training["workers"],
            "stage_workers": training["stage_workers"],
            "optimizer": training["optimizer"],
            "lr0": training["lr0"],
            "lrf": training["lrf"],
            "momentum": training["momentum"],
            "weight_decay": training["weight_decay"],
            "warmup_epochs": training["warmup_epochs"],
            "amp": training["amp"],
            "deterministic": training["deterministic"],
            "seed": training["seed"],
            "close_mosaic": training["close_mosaic"],
            "resume": False,
        },
        "runtime": {"scratch_root": f"{runtime}/train"},
        "logging": logging,
        "results_dir": str(results / "train"),
    }

    selected_checkpoint = (
        checkpoint if args.iteration == 0 else str(results / "train" / "selected.pt")
    )

    def evaluation_spec(
        coco: str, images: str, split: str, report_only: bool
    ) -> dict[str, Any]:
        return {
            "model": {
                "architecture": model["architecture"],
                "checkpoint": selected_checkpoint,
            },
            "dataset": {"eval_coco": coco, "eval_images": images},
            "evaluation": {
                "split_name": split,
                "imgsz": inference["imgsz"],
                "batch": inference["batch_size"],
                "device": "0",
                "workers": inference["workers"],
                "stage_workers": inference["stage_workers"],
                "confidence": inference["confidence"],
                "iou": inference["iou"],
                "max_det": inference["max_det"],
                "report_only": report_only,
            },
            "runtime": {"scratch_root": f"{runtime}/{split}"},
            "results_dir": str(results / split),
        }

    specs = {
        "kpi_evaluate.yaml": evaluation_spec(kpi_coco, kpi_images, "kpi", False),
        "test_evaluate.yaml": evaluation_spec(test_coco, test_images, "test", True),
    }
    if args.iteration > 0:
        specs = {"train.yaml": train, **specs}
    return specs


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    published = _identity(args.published_output_dir or str(output))
    for name, spec in specs.items():
        _freeze_yaml(output / name, spec)
    actions = {
        name.removesuffix(".yaml"): str(published / name) for name in specs
    }
    ordering = list(actions)
    manifest = {
        "schema_version": 1,
        "detector_backend": "yolo",
        "iteration": args.iteration,
        "leaf_skill": "tao-train-yolo",
        "actions": actions,
        "ordering": ordering,
        "test_is_report_only": True,
        "unsupported": ["probes", "late_best_extension", "model_soup"],
    }
    _freeze_json(output / "yolo_spec_manifest.json", manifest)
    return manifest


def main() -> int:
    try:
        manifest = run(parse_args())
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
