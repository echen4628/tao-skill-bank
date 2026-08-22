#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze selected-checkpoint RT-DETR inference and evaluation specs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"spec is not a nested mapping: {path}")
    return value


def write_yaml(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-spec", required=True)
    parser.add_argument("--kpi-infer-spec", required=True)
    parser.add_argument("--test-infer-spec", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--kpi-coco", required=True)
    parser.add_argument("--test-coco", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-output-dir")
    parser.add_argument("--results-root", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    train = load_yaml(args.train_spec)
    inference_sources = {
        "kpi": load_yaml(args.kpi_infer_spec),
        "test": load_yaml(args.test_infer_spec),
    }
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if selection.get("action") != "select":
        raise ValueError("measurement specs require a final select decision")
    checkpoint = str(Path(selection["selected_checkpoint"]).resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    coco_paths = {
        "kpi": str(Path(args.kpi_coco).resolve()),
        "test": str(Path(args.test_coco).resolve()),
    }
    if not all(Path(value).is_file() for value in coco_paths.values()):
        raise FileNotFoundError("KPI or test COCO is missing")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Published identities may deliberately use a stable shared-filesystem
    # alias. Make the path absolute without resolving symlinks/aliases.
    published_output = Path(
        os.path.abspath(os.path.expanduser(args.published_output_dir or str(output)))
    )
    results = Path(args.results_root).resolve()
    manifest = {
        "selected_checkpoint": checkpoint,
        "best_epoch": selection["best_epoch"],
        "best_kpi_mAP50": selection["best_kpi_mAP50"],
        "specs": {},
    }

    for split, infer_source in inference_sources.items():
        infer = copy.deepcopy(infer_source)
        workers = infer["dataset"].get("workers")
        if not isinstance(workers, int) or workers < 0:
            raise ValueError(f"{split} inference workers must be non-negative")
        infer["inference"]["checkpoint"] = checkpoint
        infer["results_dir"] = str(results / "inference" / split)
        infer_path = output / f"{split}_infer_selected.yaml"
        write_yaml(infer_path, infer)

        eval_spec = {
            "results_dir": str(results / "evaluation" / split),
            "wandb": {"enable": False},
            "model": copy.deepcopy(train["model"]),
            "dataset": {
                "test_data_sources": {
                    "image_dir": infer["dataset"]["infer_data_sources"]["image_dir"][0],
                    "json_file": coco_paths[split],
                },
                "num_classes": train["dataset"]["num_classes"],
                "eval_class_ids": copy.deepcopy(train["dataset"]["eval_class_ids"]),
                "batch_size": infer["dataset"]["batch_size"],
                "workers": workers,
                "remap_mscoco_category": False,
            },
            "evaluate": {
                "num_gpus": 1,
                "gpu_ids": [0],
                "checkpoint": checkpoint,
                "conf_threshold": 0.0,
            },
        }
        eval_path = output / f"{split}_evaluate_selected.yaml"
        write_yaml(eval_path, eval_spec)
        manifest["specs"][split] = {
            "inference": str(published_output / infer_path.name),
            "evaluate": str(published_output / eval_path.name),
            "workers": workers,
        }

    manifest_path = output / "measurement_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> int:
    try:
        report = run(parse_args())
        print(
            "PREPARED_RTDETR_MEASUREMENT_SPECS",
            report["best_epoch"],
            report["best_kpi_mAP50"],
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
