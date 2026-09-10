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
    parser.add_argument("--phase", choices=("baseline", "probe", "train", "measure"), required=True)
    parser.add_argument("--train-coco", help="Required after iteration 0.")
    parser.add_argument("--train-images", help="Required after iteration 0.")
    parser.add_argument("--selected-checkpoint", help="Required for post-train measurement.")
    parser.add_argument("--probe-winner", help="Frozen winning_recipe.json for probe-selected main training.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-output-dir")
    parser.add_argument(
        "--runtime-root",
        default="${TAO_RUNTIME_ROOT}",
        help="Node-local runtime path or an environment placeholder expanded in the job.",
    )
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
    if args.phase == "baseline" and args.iteration != 0:
        raise ValueError("baseline phase requires iteration 0")
    if args.phase in {"probe", "train", "measure"} and args.iteration < 1:
        raise ValueError("probe/train/measure phases require a positive iteration")
    if args.phase in {"probe", "train"} and (not args.train_coco or not args.train_images):
        raise ValueError("--train-coco and --train-images are required after iteration 0")
    if args.phase == "measure" and not args.selected_checkpoint:
        raise ValueError("measure phase requires --selected-checkpoint")
    train_coco = _absolute_file(args.train_coco) if args.train_coco else None
    train_images = _absolute_dir(args.train_images) if args.train_images else None
    sources = policy.get("sources") or {}
    kpi_coco = _absolute_file(str((sources.get("kpi") or {}).get("coco") or ""))
    kpi_images = _absolute_dir(str((sources.get("kpi") or {}).get("images") or ""))
    test_coco = _absolute_file(str((sources.get("test") or {}).get("coco") or ""))
    test_images = _absolute_dir(str((sources.get("test") or {}).get("images") or ""))
    checkpoint = _absolute_file(str(policy.get("base_checkpoint") or ""))
    yolo = policy.get("yolo") or {}
    probes = yolo.get("probes") or {}
    training = yolo.get("training") or {}
    inference = yolo.get("evaluation") or {}
    runtime = args.runtime_root.rstrip("/")
    probe_active = bool(probes.get("enabled")) and args.iteration >= int(
        probes.get("start_iteration", 1)
    )
    winner: dict[str, Any] | None = None
    if args.phase == "probe" and not probe_active:
        raise ValueError("YOLO probes are disabled for this iteration")
    if args.phase == "train" and probe_active:
        if not getattr(args, "probe_winner", None):
            raise ValueError("probe-enabled main training requires --probe-winner")
        winner_path = Path(args.probe_winner).expanduser().resolve()
        if not winner_path.is_file():
            raise FileNotFoundError(winner_path)
        winner = json.loads(winner_path.read_text(encoding="utf-8"))
        if int(winner.get("iteration", -1)) != args.iteration:
            raise ValueError("probe winner iteration does not match requested iteration")
        if winner.get("selected_by") != "kpi_validation_ap50":
            raise ValueError("probe winner must be selected only by KPI validation AP50")

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
        "results_dir": "{results_dir}",
    }

    if winner is not None:
        overrides = winner.get("overrides") or {}
        allowed = {"optimizer", "lr0", "lrf", "momentum", "weight_decay", "seed"}
        if not isinstance(overrides, dict) or not overrides or set(overrides) - allowed:
            raise ValueError("probe winner contains invalid or empty training overrides")
        train["train"].update(overrides)

    selected_checkpoint = (
        _absolute_file(args.selected_checkpoint)
        if args.phase == "measure"
        else checkpoint
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
            "results_dir": "{results_dir}",
        }

    if args.phase == "probe":
        candidates = probes.get("candidates") or []
        if len(candidates) != 3:
            raise ValueError("YOLO probe policy requires exactly three candidates")
        names: set[str] = set()
        specs = {}
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError("each YOLO probe candidate must be a mapping")
            name = str(candidate.get("name") or "").strip()
            if not name or name in names:
                raise ValueError("YOLO probe candidate names must be non-empty and unique")
            names.add(name)
            probe = json.loads(json.dumps(train))
            probe["train"].update(
                epochs=int(probes["epochs"]),
                patience=int(probes["epochs"]),
                close_mosaic=int(probes.get("close_mosaic", 0)),
                seed=int(probes.get("seed_base", 4000)) + args.iteration,
                lr0=float(candidate["lr0"]),
                lrf=float(candidate["lrf"]),
                weight_decay=float(candidate["weight_decay"]),
            )
            specs[f"probe_{index}_{name}.yaml"] = probe
    elif args.phase == "train":
        specs = {"train.yaml": train}
    else:
        specs = {
            "kpi_evaluate.yaml": evaluation_spec(kpi_coco, kpi_images, "kpi", False),
            "test_evaluate.yaml": evaluation_spec(test_coco, test_images, "test", True),
        }
    return specs


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args)
    policy = yaml.safe_load(Path(args.policy).expanduser().resolve().read_text())
    yolo = policy.get("yolo") or {}
    probes = yolo.get("probes") or {}
    training = yolo.get("training") or {}
    winner = (
        json.loads(Path(args.probe_winner).expanduser().resolve().read_text(encoding="utf-8"))
        if getattr(args, "probe_winner", None) else None
    )
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
        "phase": args.phase,
        "leaf_skill": "tao-train-yolo",
        "actions": actions,
        "ordering": ordering,
        "test_is_report_only": True,
        "probe_selection": winner,
        "probe_candidates": (probes.get("candidates") if args.phase == "probe" else None),
        "probe_seed": (
            int(probes.get("seed_base", 4000)) + args.iteration
            if args.phase == "probe" else None
        ),
        "probe_fixed_overrides": (
            {
                "optimizer": str(training["optimizer"]),
                "momentum": float(training["momentum"]),
            }
            if args.phase == "probe" else None
        ),
        "unsupported": ["late_best_extension", "model_soup"],
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
