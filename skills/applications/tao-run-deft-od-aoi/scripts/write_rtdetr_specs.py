#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write nested DEFT OD AOI RT-DETR train, probe, and inference specs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import yaml

from deft_od_aoi_policy import load_policy, main_epoch_budget, probes_active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-coco", required=True)
    parser.add_argument("--train-images-dir", required=True)
    parser.add_argument("--kpi-coco", required=True)
    parser.add_argument("--kpi-images-dir", required=True)
    parser.add_argument("--test-images-dir", required=True)
    parser.add_argument("--classmap", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--incumbent-config", default=None)
    parser.add_argument("--history", default=None)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def absolute_file(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)


def absolute_dir(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return str(resolved)


def set_nested(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    node = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = replacement


def dump_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def dump_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def base_train_spec(
    policy: dict[str, Any],
    *,
    train_coco: str,
    train_images: str,
    kpi_coco: str,
    kpi_images: str,
    checkpoint: str,
    results_dir: str,
) -> dict[str, Any]:
    model = policy["model"]
    training = policy["training"]
    return {
        "results_dir": results_dir,
        "wandb": {"enable": False},
        "model": {
            "backbone": model["backbone"],
            "train_backbone": True,
            "num_feature_levels": model["num_feature_levels"],
            "return_interm_indices": model["return_interm_indices"],
        },
        "dataset": {
            "train_data_sources": [
                {"image_dir": train_images, "json_file": train_coco}
            ],
            "val_data_sources": {"image_dir": kpi_images, "json_file": kpi_coco},
            "num_classes": 2,
            "eval_class_ids": [1],
            "batch_size": training["batch_size"],
            "workers": training["workers"],
            "remap_mscoco_category": False,
        },
        "train": {
            "num_gpus": training["num_gpus"],
            "gpu_ids": list(range(int(training["num_gpus"]))),
            "num_epochs": training["fixed_epochs"],
            "checkpoint_interval": training["checkpoint_interval"],
            "validation_interval": training["validation_interval"],
            "pretrained_model_path": checkpoint,
            "optim": {
                "lr": training["base_learning_rate"],
                "lr_backbone": training["backbone_learning_rate"],
            },
        },
    }


def inference_spec(
    policy: dict[str, Any],
    *,
    images_dir: str,
    classmap: str,
    checkpoint: str,
    results_dir: str,
) -> dict[str, Any]:
    model = policy["model"]
    inference = policy["inference"]
    return {
        "results_dir": results_dir,
        "wandb": {"enable": False},
        "model": {
            "backbone": model["backbone"],
            "train_backbone": True,
            "num_feature_levels": model["num_feature_levels"],
            "return_interm_indices": model["return_interm_indices"],
        },
        "dataset": {
            "infer_data_sources": {
                "image_dir": [images_dir],
                "classmap": classmap,
            },
            "num_classes": 2,
            "batch_size": inference["batch_size"],
            "workers": inference["workers"],
            "remap_mscoco_category": False,
        },
        "inference": {
            "num_gpus": inference["num_gpus"],
            "gpu_ids": list(range(int(inference["num_gpus"]))),
            "checkpoint": checkpoint,
            "conf_threshold": inference["confidence_threshold"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    if not 1 <= args.iteration <= int(policy["max_iterations"]):
        raise ValueError("iteration is outside the frozen policy range")
    train_coco = absolute_file(args.train_coco)
    train_images = absolute_dir(args.train_images_dir)
    kpi_coco = absolute_file(args.kpi_coco)
    kpi_images = absolute_dir(args.kpi_images_dir)
    test_images = absolute_dir(args.test_images_dir)
    classmap = absolute_file(args.classmap)
    checkpoint = absolute_file(args.base_checkpoint)
    train_size = len(read_json(train_coco).get("images", []))
    if train_size < 1:
        raise ValueError("assembled training COCO has no images")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = Path(args.results_root).expanduser().resolve()
    train_spec = base_train_spec(
        policy,
        train_coco=train_coco,
        train_images=train_images,
        kpi_coco=kpi_coco,
        kpi_images=kpi_images,
        checkpoint=checkpoint,
        results_dir=str(results / "main"),
    )
    training = policy["training"]
    run_probes = probes_active(policy, args.iteration)
    if not run_probes:
        train_spec["train"]["num_epochs"] = main_epoch_budget(
            policy, args.iteration, train_size
        )
    dump_yaml(output / "train.yaml", train_spec)
    for name, images in (("kpi", kpi_images), ("test", test_images)):
        dump_yaml(
            output / f"{name}_infer.yaml",
            inference_spec(
                policy,
                images_dir=images,
                classmap=classmap,
                checkpoint="DEFT_OD_AOI_SELECTED_CHECKPOINT",
                results_dir=str(results / "inference" / name),
            ),
        )

    manifest: dict[str, Any] = {
        "iteration": args.iteration,
        "train_size": train_size,
        "probes": [],
    }
    if run_probes:
        incumbent = read_json(args.incumbent_config) if args.incumbent_config else {}
        if not isinstance(incumbent, dict):
            raise ValueError("incumbent config must be a JSON object")
        history = read_json(args.history) if args.history else []
        if not isinstance(history, list):
            raise ValueError("history must be a JSON array")
        previous_size = int(history[-1]["train_size"]) if history else train_size
        growth = train_size / max(1, previous_size)
        current = {
            "train.optim.lr": float(training["base_learning_rate"]),
            "train.optim.lr_backbone": float(training["backbone_learning_rate"]),
            "train.optim.weight_decay": 1.0e-4,
            "train.optim.lr_decay": 0.1,
            "train.optim.lr_step_size": 30,
        }
        current.update(incumbent)
        probe_policy = training["probe_policy"]
        scaled = dict(current)
        if growth > float(probe_policy["growth_high"]):
            scaled["train.optim.lr"] *= float(probe_policy["high_growth_lr_factor"])
            scaled["train.optim.lr_step_size"] = min(
                int(probe_policy["high_growth_step_maximum"]),
                int(current["train.optim.lr_step_size"])
                + int(probe_policy["high_growth_step_increment"]),
            )
        elif growth < float(probe_policy["growth_low"]):
            scaled["train.optim.lr"] *= float(probe_policy["low_growth_lr_factor"])
        rng = random.Random(int(probe_policy["jitter_seed_base"]) + args.iteration)
        jitter = dict(current)
        jitter["train.optim.lr"] *= rng.uniform(*probe_policy["jitter_lr_factor"])
        jitter["train.optim.weight_decay"] *= rng.uniform(
            *probe_policy["jitter_weight_decay_factor"]
        )
        jitter["train.optim.lr_decay"] *= rng.uniform(
            *probe_policy["jitter_lr_decay_factor"]
        )
        low, high = probe_policy["jitter_lr_decay_bounds"]
        jitter["train.optim.lr_decay"] = min(
            float(high), max(float(low), jitter["train.optim.lr_decay"])
        )
        candidates = (("incumbent", current), ("scaled", scaled), ("jitter", jitter))
        manifest["growth"] = growth
        for index, (name, deltas) in enumerate(candidates):
            spec = copy.deepcopy(train_spec)
            for dotted, value in deltas.items():
                set_nested(spec, dotted, value)
            spec["train"]["num_epochs"] = training["probe_epochs"]
            spec["train"]["checkpoint_interval"] = training["probe_epochs"]
            probe_results = results / "probes" / f"p{index}"
            spec["results_dir"] = str(probe_results)
            dump_yaml(output / f"probe{index}.yaml", spec)
            manifest["probes"].append(
                {
                    "probe": index,
                    "name": name,
                    "deltas": deltas,
                    "results_dir": str(probe_results),
                }
            )
    dump_json(output / "probe_manifest.json", manifest)
    return manifest


def main() -> int:
    try:
        report = run(parse_args())
        print(
            f"DEFT OD AOI RT-DETR specs: iter={report['iteration']} "
            f"train_size={report['train_size']} probes={len(report['probes'])}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
