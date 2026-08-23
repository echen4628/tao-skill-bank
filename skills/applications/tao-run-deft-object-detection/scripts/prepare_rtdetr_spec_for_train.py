#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append one mined COCO source and freeze an RT-DETR training class contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from stage_mined_coco import category_contract


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _absolute_existing(raw: str, kind: str) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {path}")
    path = path.resolve()
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    if kind == "dir" and not path.is_dir():
        raise NotADirectoryError(f"directory does not exist: {path}")
    return str(path)


def _contract(coco_path: Path) -> tuple[list[int], list[str]]:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    _categories, ids, names = category_contract(coco)
    return ids, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-spec", required=True)
    parser.add_argument("--output-spec", required=True)
    parser.add_argument("--tmm-image-dir", required=True)
    parser.add_argument("--tmm-coco-file", required=True)
    parser.add_argument("--val-image-dir", required=True)
    parser.add_argument("--val-json-file", required=True)
    parser.add_argument("--pretrained-model-path", required=True)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    try:
        previous = Path(args.previous_spec).expanduser().resolve()
        if not previous.is_file():
            raise FileNotFoundError(f"previous-spec does not exist: {previous}")
        spec = _load_yaml(previous)
        image_dir = _absolute_existing(args.tmm_image_dir, "dir")
        coco_file = _absolute_existing(args.tmm_coco_file, "file")
        val_image_dir = _absolute_existing(args.val_image_dir, "dir")
        val_json_file = _absolute_existing(args.val_json_file, "file")
        checkpoint = _absolute_existing(args.pretrained_model_path, "file")

        ids, names = _contract(Path(coco_file))
        val_ids, val_names = _contract(Path(val_json_file))
        if (ids, names) != (val_ids, val_names):
            raise ValueError(
                "training and validation COCO class contracts differ: "
                f"train={list(zip(ids, names))}, val={list(zip(val_ids, val_names))}"
            )

        dataset = spec.setdefault("dataset", {})
        sources = dataset.get("train_data_sources") or []
        if not isinstance(sources, list):
            raise ValueError("dataset.train_data_sources must be a list")
        entry = {"image_dir": image_dir, "json_file": coco_file}
        if entry not in sources:
            sources.append(entry)
        dataset["train_data_sources"] = sources
        dataset["val_data_sources"] = {"image_dir": val_image_dir, "json_file": val_json_file}
        dataset["num_classes"] = max(ids) + 1
        dataset["eval_class_ids"] = ids
        dataset["remap_mscoco_category"] = False

        train = spec.setdefault("train", {})
        train["pretrained_model_path"] = checkpoint
        train["resume_training_checkpoint_path"] = ""
        if args.num_epochs is not None:
            if args.num_epochs < 1:
                raise ValueError("--num-epochs must be at least 1")
            train["num_epochs"] = args.num_epochs
            for field in ("checkpoint_interval", "validation_interval"):
                if isinstance(train.get(field), int) and train[field] > args.num_epochs:
                    train[field] = args.num_epochs
        if args.learning_rate is not None:
            if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
                raise ValueError("--learning-rate must be finite and greater than zero")
            train.setdefault("optim", {})["lr"] = args.learning_rate
        if args.num_gpus is not None:
            if args.num_gpus < 1:
                raise ValueError("--num-gpus must be at least 1")
            train["num_gpus"] = args.num_gpus
            train["gpu_ids"] = list(range(args.num_gpus))

        output = Path(args.output_spec).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        print(
            f"wrote {output}: sources={len(sources)} categories={list(zip(ids, names))} "
            f"num_classes={dataset['num_classes']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
