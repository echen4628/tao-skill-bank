#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one train or evaluate action from the tao-train-yolo YAML contract."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from yolo_common import (
    METRIC_KEY,
    atomic_json,
    common_coco_score,
    merge_results,
    predictions_to_kitti,
    select_row,
    sha256_file,
    stage_coco,
    write_dataset_yaml,
    write_results,
)


TOP_LEVEL_KEYS = {"model", "dataset", "train", "evaluation", "runtime", "logging", "results_dir"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _mapping(config: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = config.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _path(value: Any, label: str, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"{label} contains an unresolved environment variable")
    path = Path(expanded).expanduser().resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _directory(value: Any, label: str) -> Path:
    path = _path(value, label, must_exist=False)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_config(path: Path, action: str) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    def reject_dotted_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if "." in str(key):
                    raise ValueError("flat dotted keys are forbidden")
                reject_dotted_keys(child)
        elif isinstance(value, list):
            for child in value:
                reject_dotted_keys(child)

    reject_dotted_keys(config)
    model = _mapping(config, "model")
    architecture = str(model.get("architecture", "")).strip()
    if not architecture.startswith("yolo"):
        raise ValueError("model.architecture must identify a YOLO architecture")
    _path(model.get("checkpoint"), "model.checkpoint")
    dataset = _mapping(config, "dataset")
    _path(dataset.get("eval_coco"), "dataset.eval_coco")
    _directory(dataset.get("eval_images"), "dataset.eval_images")
    if action == "train":
        _path(dataset.get("train_coco"), "dataset.train_coco")
        _directory(dataset.get("train_images"), "dataset.train_images")
        train = _mapping(config, "train")
        for key in ("num_gpus", "epochs", "patience", "imgsz", "batch", "nbs", "stage_workers"):
            if int(train.get(key, 0)) < 1:
                raise ValueError(f"train.{key} must be positive")
        for key in ("lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"):
            if key not in train:
                raise ValueError(f"train.{key} is required")
        for key in ("amp", "deterministic"):
            if not isinstance(train.get(key), bool):
                raise ValueError(f"train.{key} must be boolean")
        for key in ("devices", "optimizer", "seed", "close_mosaic"):
            if key not in train:
                raise ValueError(f"train.{key} is required")
        devices = train["devices"]
        if isinstance(devices, str):
            device_count = len([value for value in devices.split(",") if value.strip()])
        elif isinstance(devices, list):
            device_count = len(devices)
        else:
            raise ValueError("train.devices must be a comma-separated string or list")
        if device_count != int(train["num_gpus"]):
            raise ValueError("train.num_gpus must match train.devices")
        if int(train.get("workers", -1)) < 0:
            raise ValueError("train.workers cannot be negative")
        resume = train.get("resume")
        if not isinstance(resume, bool):
            raise ValueError("train.resume must be boolean")
        if resume:
            _path(train.get("prior_results_csv"), "train.prior_results_csv")
            _path(train.get("prior_best_checkpoint"), "train.prior_best_checkpoint")
    else:
        evaluation = _mapping(config, "evaluation")
        if evaluation.get("split_name") not in {"kpi", "test"}:
            raise ValueError("evaluation.split_name must be kpi or test")
        if evaluation["split_name"] == "test" and evaluation.get("report_only") is not True:
            raise ValueError("sealed test evaluation must be report-only")
        for key in ("stage_workers", "imgsz", "batch", "max_det"):
            if int(evaluation.get(key, 0)) < 1:
                raise ValueError(f"evaluation.{key} must be positive")
        if int(evaluation.get("workers", -1)) < 0:
            raise ValueError("evaluation.workers cannot be negative")
        confidence = float(evaluation.get("confidence", -1))
        if not 0 <= confidence <= 1:
            raise ValueError("evaluation.confidence must be within [0, 1]")
    runtime = _mapping(config, "runtime")
    scratch = _path(runtime.get("scratch_root"), "runtime.scratch_root", must_exist=False)
    if os.environ.get("SLURM_JOB_ID") and not scratch.is_relative_to(Path("/raid/scratch")):
        raise ValueError("SLURM runtime.scratch_root must be under /raid/scratch")
    _path(config.get("results_dir"), "results_dir", must_exist=False)
    logging = _mapping(config, "logging", required=False)
    enabled = logging.get("onelogger_enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("logging.onelogger_enabled must be boolean")
    if enabled and not str(logging.get("callback_module", "")).strip():
        raise ValueError("enabled OneLogger requires logging.callback_module")
    return config


def _register_logging(model: object, config: dict[str, Any]) -> None:
    logging = _mapping(config, "logging", required=False)
    if not logging.get("onelogger_enabled"):
        return
    module = importlib.import_module(str(logging["callback_module"]))
    register = getattr(module, "register", None)
    if not callable(register):
        raise ValueError("OneLogger callback module must export register(model)")
    register(model)


def _roots(config: dict[str, Any]) -> tuple[Path, Path, dict[str, int]]:
    runtime = _mapping(config, "runtime")
    scratch = _path(runtime["scratch_root"], "runtime.scratch_root", must_exist=False)
    output = _path(config["results_dir"], "results_dir", must_exist=False)
    scratch.mkdir(parents=True, exist_ok=False)
    # The platform binds results_dir when it opens the job record, before the
    # workload is submitted, and may place immutable launch inputs beneath its
    # staged/ directory.  Accept that control-plane-owned directory while still
    # refusing to mix a new action with artifacts from an earlier attempt.
    output.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in output.iterdir() if path.name != "staged")
    if unexpected:
        raise FileExistsError(
            f"results_dir already contains workload artifacts: {', '.join(unexpected)}"
        )
    now = int(time.time())
    timings = {
        "allocation_start": int(runtime.get("allocation_start", now)),
        "runner_start": now,
    }
    return scratch, output, timings


def _public_stage(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "source_coco"}


def run_train(config: dict[str, Any]) -> None:
    from ultralytics import YOLO

    model_config = _mapping(config, "model")
    dataset = _mapping(config, "dataset")
    train = _mapping(config, "train")
    checkpoint = _path(model_config["checkpoint"], "model.checkpoint")
    train_coco = _path(dataset["train_coco"], "dataset.train_coco")
    train_images = _directory(dataset["train_images"], "dataset.train_images")
    eval_coco = _path(dataset["eval_coco"], "dataset.eval_coco")
    eval_images = _directory(dataset["eval_images"], "dataset.eval_images")
    scratch, output, timings = _roots(config)
    status = output / "status.json"
    atomic_json(status, {"state": "RUNNING", "stage": "staging", "updated_at": int(time.time())})
    data_root = scratch / "dataset"
    train_report = stage_coco(
        train_coco,
        train_images,
        data_root / "train",
        preserve_ids=False,
        workers=int(train["stage_workers"]),
    )
    eval_report = stage_coco(
        eval_coco,
        eval_images,
        data_root / "eval",
        preserve_ids=True,
        workers=int(train["stage_workers"]),
    )
    dataset_yaml = data_root / "dataset.yaml"
    write_dataset_yaml(
        dataset_yaml,
        train_images=data_root / "train" / "images",
        eval_images=data_root / "eval" / "images",
    )
    timings["staging_complete"] = int(time.time())
    atomic_json(output / "timings.json", timings)
    manifest = {
        "architecture": model_config["architecture"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train": _public_stage(train_report),
        "kpi": _public_stage(eval_report),
        "recipe": train,
        "onelogger": _mapping(config, "logging", required=False),
    }
    atomic_json(output / "training_manifest.json", manifest)
    model = YOLO(str(checkpoint))
    _register_logging(model, config)
    observed_optimizer: dict[str, str] = {}

    def capture_optimizer(trainer: object) -> None:
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            observed_optimizer["class"] = type(optimizer).__name__

    model.add_callback("on_pretrain_routine_end", capture_optimizer)
    atomic_json(status, {"state": "RUNNING", "stage": "training", "updated_at": int(time.time())})
    timings["workload_start"] = int(time.time())
    atomic_json(output / "timings.json", timings)
    common = {
        "data": str(dataset_yaml),
        "device": train["devices"],
        "workers": int(train["workers"]),
        "save_period": 1,
        "patience": int(train["patience"]),
        "val": True,
        "plots": False,
        "verbose": True,
    }
    if train["resume"]:
        model.train(resume=True, **common)
    else:
        model.train(
            epochs=int(train["epochs"]),
            imgsz=int(train["imgsz"]),
            batch=int(train["batch"]),
            nbs=int(train["nbs"]),
            optimizer=str(train["optimizer"]),
            lr0=float(train["lr0"]),
            lrf=float(train["lrf"]),
            momentum=float(train["momentum"]),
            weight_decay=float(train["weight_decay"]),
            warmup_epochs=float(train["warmup_epochs"]),
            amp=bool(train["amp"]),
            deterministic=bool(train["deterministic"]),
            seed=int(train["seed"]),
            close_mosaic=int(train["close_mosaic"]),
            save=True,
            project=str(scratch / "trainer"),
            name="train",
            exist_ok=False,
            **common,
        )
    train_dir = Path(model.trainer.save_dir)
    prior_csv = _path(train["prior_results_csv"], "train.prior_results_csv") if train["resume"] else None
    rows, prior_last_epoch = merge_results(prior_csv, train_dir / "results.csv")
    write_results(output / "results.csv", rows)
    selected_index, selected_row = select_row(rows)
    selected_epoch = int(float(selected_row.get("epoch", selected_index + 1)))
    if train["resume"] and selected_epoch <= prior_last_epoch:
        selected_source = _path(train["prior_best_checkpoint"], "train.prior_best_checkpoint")
    else:
        periodic = train_dir / "weights" / f"epoch{selected_epoch - 1}.pt"
        selected_source = periodic if periodic.is_file() else train_dir / "weights" / "last.pt"
    terminal_epoch = max(int(float(row.get("epoch", index + 1))) for index, row in enumerate(rows))
    terminal_source = train_dir / "weights" / f"epoch{terminal_epoch - 1}.pt"
    last_source = train_dir / "weights" / "last.pt"
    for source in (selected_source, terminal_source, last_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    selected_output = output / "selected.pt"
    terminal_output = output / "terminal_resume.pt"
    last_output = output / "last.pt"
    shutil.copyfile(selected_source, selected_output)
    shutil.copyfile(terminal_source, terminal_output)
    shutil.copyfile(last_source, last_output)
    atomic_json(
        output / "selection.json",
        {
            "selection_metric": METRIC_KEY,
            "metric_value": float(selected_row[METRIC_KEY]),
            "selected_epoch_reported": selected_epoch,
            "selected_epoch_zero_based": selected_epoch - 1,
            "completed_epochs": len(rows),
            "selected_checkpoint": str(selected_output),
            "selected_sha256": sha256_file(selected_output),
            "terminal_resume_checkpoint": str(terminal_output),
            "terminal_resume_sha256": sha256_file(terminal_output),
            "last_checkpoint": str(last_output),
            "last_sha256": sha256_file(last_output),
            "optimizer_observed_class": observed_optimizer.get("class", "unknown"),
            "row": selected_row,
        },
    )
    timings["workload_end"] = int(time.time())
    atomic_json(output / "timings.json", timings)
    atomic_json(status, {"state": "COMPLETE", "stage": "complete", "updated_at": int(time.time())})


def run_evaluate(config: dict[str, Any]) -> None:
    from ultralytics import YOLO

    model_config = _mapping(config, "model")
    dataset = _mapping(config, "dataset")
    evaluation = _mapping(config, "evaluation")
    checkpoint = _path(model_config["checkpoint"], "model.checkpoint")
    eval_coco = _path(dataset["eval_coco"], "dataset.eval_coco")
    eval_images = _directory(dataset["eval_images"], "dataset.eval_images")
    scratch, output, timings = _roots(config)
    status = output / "status.json"
    atomic_json(status, {"state": "RUNNING", "stage": "staging", "updated_at": int(time.time())})
    split_root = scratch / "dataset" / str(evaluation["split_name"])
    stage_report = stage_coco(
        eval_coco,
        eval_images,
        split_root,
        preserve_ids=True,
        workers=int(evaluation["stage_workers"]),
    )
    dataset_yaml = scratch / "dataset" / "dataset.yaml"
    write_dataset_yaml(dataset_yaml, train_images=None, eval_images=split_root / "images")
    timings["staging_complete"] = int(time.time())
    timings["workload_start"] = int(time.time())
    atomic_json(output / "timings.json", timings)
    model = YOLO(str(checkpoint))
    result = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=int(evaluation["imgsz"]),
        batch=int(evaluation["batch"]),
        device=evaluation["device"],
        workers=int(evaluation["workers"]),
        conf=float(evaluation["confidence"]),
        iou=float(evaluation["iou"]),
        max_det=int(evaluation["max_det"]),
        save_json=True,
        plots=False,
        project=str(scratch / "evaluation"),
        name=str(evaluation["split_name"]),
        exist_ok=False,
        verbose=True,
    )
    predictions_source = Path(result.save_dir) / "predictions.json"
    if not predictions_source.is_file():
        raise FileNotFoundError(predictions_source)
    predictions = output / "predictions.json"
    shutil.copyfile(predictions_source, predictions)
    native = {key: float(value) for key, value in result.results_dict.items()}
    atomic_json(output / "ultralytics_metrics.json", native)
    common = common_coco_score(eval_coco, predictions, output / "common_coco_metrics.json")
    kitti = predictions_to_kitti(
        eval_coco,
        predictions,
        output / "kitti_labels",
        float(evaluation["confidence"]),
    )
    atomic_json(
        output / "measurement_manifest.json",
        {
            "architecture": model_config["architecture"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "split": evaluation["split_name"],
            "report_only": bool(evaluation["report_only"]),
            "inference": {
                "confidence": float(evaluation["confidence"]),
                "iou": float(evaluation["iou"]),
                "max_det": int(evaluation["max_det"]),
            },
            "stage": _public_stage(stage_report),
            "native_metrics": native,
            "common_coco_metrics": common,
            "predictions": str(predictions),
            "kitti_labels": str(output / "kitti_labels"),
            "kitti_conversion": kitti,
        },
    )
    timings["workload_end"] = int(time.time())
    atomic_json(output / "timings.json", timings)
    atomic_json(status, {"state": "COMPLETE", "stage": "complete", "updated_at": int(time.time())})


def main() -> int:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve(), args.action)
    if args.action == "train":
        run_train(config)
    else:
        run_evaluate(config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
