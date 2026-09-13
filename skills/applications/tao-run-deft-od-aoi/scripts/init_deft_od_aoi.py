#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate normalized DEFT OD AOI roles and freeze a detector run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deft_od_aoi_policy import build_policy


DEFAULTS = Path(__file__).resolve().parents[1] / "assets" / "default_policy.yaml"
# Normalized handoff roles: KPI/test are held out, ``real`` is the
# defective-real mining pool, and ``clean`` is the verified-clean mining pool.
ROLES = ("kpi", "test", "real", "clean")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def _image_path(images: Path, row: dict[str, Any]) -> Path:
    source = str(row.get("source_path") or "").strip()
    path = Path(source) if source else images / str(row.get("file_name") or "")
    return path.expanduser().resolve()


def _role(name: str, value: dict[str, Any]) -> dict[str, Any]:
    images = Path(str(value.get("images") or "")).expanduser().resolve()
    coco_path = Path(str(value.get("coco") or "")).expanduser().resolve()
    if not images.is_dir() or not coco_path.is_file():
        raise ValueError(f"{name} images/COCO are missing")
    coco = json.loads(coco_path.read_text())
    categories = coco.get("categories", [])
    if not categories or {str(row.get("name")) for row in categories} != {"defect"}:
        raise ValueError(f"{name} must declare only the defect category")
    category_ids = {int(row["id"]) for row in categories}
    image_rows = coco.get("images", [])
    image_ids = {int(row["id"]) for row in image_rows}
    if not image_rows or len(image_ids) != len(image_rows):
        raise ValueError(f"{name} has no images or duplicate image ids")
    counts = {image_id: 0 for image_id in image_ids}
    for annotation in coco.get("annotations", []):
        image_id, category = int(annotation["image_id"]), int(annotation["category_id"])
        if image_id not in counts or category not in category_ids:
            raise ValueError(f"{name} annotation references unknown image/category")
        x, y, width, height = map(float, annotation["bbox"])
        if min(x, y) < 0 or width <= 0 or height <= 0:
            raise ValueError(f"{name} contains an invalid bbox")
        counts[image_id] += 1
    paths = [_image_path(images, row) for row in image_rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{name} references missing images; first={missing[0]}")
    if name == "clean" and any(counts.values()):
        raise ValueError("clean role must have zero annotations")
    if name == "real" and any(count == 0 for count in counts.values()):
        raise ValueError("defective-real role contains a boxless image")
    return {"images": str(images), "coco": str(coco_path), "coco_sha256": _sha(coco_path),
            "image_count": len(paths), "annotation_count": sum(counts.values()),
            "identities": {str(path) for path in paths}}


def _routing_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Translate the public application policy into the proven routing contract."""
    routed = build_policy(
        max_iterations=int(policy["max_iterations"]),
        synthetic_enabled=bool((policy.get("synthesis") or {}).get("enabled")),
        probes_enabled=bool((policy.get("yolo") or {}).get("probes", {}).get("enabled")),
        model_soup_enabled=False,
        training_workers=int((policy.get("training") or {}).get("workers", 4)),
        inference_workers=int((policy.get("yolo") or {}).get("evaluation", {}).get("workers", 8)),
    )
    routed["gap"].update({
        key: policy["gap"][key]
        for key in ("loose_confidence", "strict_confidence", "match_iou",
                    "background_iou_upper", "near_miss_iou_upper")
    })
    routed["routing"].update(policy["routing"])
    routed["retrieval"].update(policy["retrieval"])
    routed["admission"].update(policy["admission"])
    routed["synthetic"]["cumulative_fraction_of_defective"] = float(
        (policy.get("synthesis") or {}).get("cumulative_fraction_of_real_defects", 0.25)
    )
    return routed


def initialize(
    config_path: Path,
    output: Path,
    *,
    published_output: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    user = yaml.safe_load(config_path.read_text())
    if not isinstance(user, dict):
        raise ValueError("config must be a YAML mapping")
    policy = _merge(yaml.safe_load(DEFAULTS.read_text()), user)
    if "near_miss_real_cap" in (user.get("routing") or {}):
        raise ValueError(
            "routing.near_miss_real_cap is obsolete; use "
            "routing.near_miss_real_cap_per_pocket"
        )
    if not isinstance(policy.get("max_iterations"), int) or policy["max_iterations"] < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not str(policy.get("platform") or "").strip():
        raise ValueError("platform must be selected before initialization")
    checkpoint = Path(str(policy.get("base_checkpoint") or "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError("base_checkpoint must be a trainable detector file")
    mode = str(policy.get("reproduction_mode") or "")
    if mode not in {"true_fresh", "warm_seeded_historical"}:
        raise ValueError(
            "reproduction_mode must be true_fresh or warm_seeded_historical"
        )
    seed_raw = policy.get("routing_seed_checkpoint") or checkpoint
    routing_seed = Path(str(seed_raw)).expanduser().resolve()
    if not routing_seed.is_file():
        raise ValueError("routing_seed_checkpoint must be a detector file")
    if mode == "true_fresh" and _sha(routing_seed) != _sha(checkpoint):
        raise ValueError(
            "true_fresh requires routing_seed_checkpoint to match base_checkpoint"
        )
    model = policy.get("model") or {}
    backend = str(model.get("backend") or "rtdetr")
    architecture = str(model.get("architecture") or "")
    if backend not in {"rtdetr", "yolo"}:
        raise ValueError("model.backend must be rtdetr or yolo")
    if backend == "rtdetr" and architecture != "rtdetr":
        raise ValueError("the RT-DETR backend requires model.architecture: rtdetr")
    if backend == "yolo" and not architecture.startswith("yolo"):
        raise ValueError("the YOLO backend requires a YOLO model.architecture")
    if backend == "yolo":
        yolo = policy.get("yolo") or {}
        probes = yolo.get("probes") or {}
        training = yolo.get("training") or {}
        evaluation = yolo.get("evaluation") or {}
        for key in ("num_gpus", "epochs", "patience", "imgsz", "batch_size", "nbs", "stage_workers"):
            if int(training.get(key, 0)) < 1:
                raise ValueError(f"yolo.training.{key} must be positive")
        for key in ("imgsz", "batch_size", "stage_workers", "max_det"):
            if int(evaluation.get(key, 0)) < 1:
                raise ValueError(f"yolo.evaluation.{key} must be positive")
        if probes.get("enabled"):
            if int(probes.get("start_iteration", 0)) < 1 or int(probes.get("epochs", 0)) < 1:
                raise ValueError("enabled yolo.probes needs positive start_iteration and epochs")
            candidates = probes.get("candidates") or []
            if len(candidates) != 3 or len({str(row.get("name") or "") for row in candidates}) != 3:
                raise ValueError("enabled yolo.probes requires exactly three unique candidates")
            for candidate in candidates:
                for key in ("lr0", "lrf", "weight_decay"):
                    if float(candidate.get(key, -1)) < 0:
                        raise ValueError(f"yolo probe candidate {key} must be non-negative")
    role_reports = {name: _role(name, policy["sources"][name]) for name in ROLES}
    owners: dict[str, str] = {}
    for name, report in role_reports.items():
        for identity in report.pop("identities"):
            if identity in owners:
                raise ValueError(f"image overlaps {owners[identity]} and {name}: {identity}")
            owners[identity] = name
    gap = policy["gap"]
    if not (0 <= gap["inference_confidence"] <= gap["loose_confidence"]
            <= gap["strict_confidence"] <= 1):
        raise ValueError("inference/loose/strict confidence thresholds are inconsistent")
    if not (0 <= gap["background_iou_upper"] < gap["near_miss_iou_upper"] <= 1):
        raise ValueError("background and near-miss IoU thresholds are inconsistent")
    if policy["class_name"] != "defect":
        raise ValueError("DEFT OD AOI has one foreground class named defect")
    synthesis = policy.get("synthesis", {})
    if synthesis.get("enabled"):
        if not Path(str(synthesis.get("pool_dataset_root") or "")).is_dir():
            raise ValueError("enabled synthesis needs pool_dataset_root")
        if not Path(str(synthesis.get("defect_spec") or "")).is_file():
            raise ValueError("enabled synthesis needs defect_spec")
        if not synthesis.get("routes"):
            raise ValueError("enabled synthesis needs at least one dataset route")
        for name, route in synthesis["routes"].items():
            ready = Path(str(route.get("checkpoint") or "")).is_file() and Path(
                str(route.get("recipe") or "")).is_file()
            finetune = route.get("finetune") or {}
            if not ready and any(not str(finetune.get(key) or "").strip() for key in
                                 ("dataset_root", "validation_testcase", "base_checkpoint",
                                  "vae_path", "nn_backbone", "result_handoff")):
                raise ValueError(f"synthesis route {name} needs checkpoint/recipe or finetune inputs")
    output.mkdir(parents=True)
    policy["base_checkpoint"] = str(checkpoint)
    policy["routing_seed_checkpoint"] = str(routing_seed)
    policy["sources"] = {name: {"images": role_reports[name]["images"],
                                 "coco": role_reports[name]["coco"]} for name in ROLES}
    frozen = output / "deft_od_aoi_policy.yaml"
    frozen.write_text(yaml.safe_dump(policy, sort_keys=False))
    routing_policy = output / "deft_od_aoi_routing_policy.json"
    _json(routing_policy, _routing_policy(policy))
    classmap = output / "inference_classmap.txt"
    classmap.write_text("background\ndefect\n")
    synthesis_enabled = bool(policy.get("synthesis", {}).get("enabled"))
    bootstrap_required = synthesis_enabled and any(
        not (Path(str(route.get("checkpoint") or "")).is_file()
             and Path(str(route.get("recipe") or "")).is_file())
        for route in policy.get("synthesis", {}).get("routes", {}).values()
    )
    published = (published_output or output).expanduser().resolve()
    state = {"schema_version": 1, "status": "READY",
             "mode": f"{backend}_with_synthesis" if synthesis_enabled else f"{backend}_real_only",
             "detector_backend": backend,
             "detector_architecture": architecture,
             "detector_skill": "tao-train-yolo" if backend == "yolo" else "tao-train-rtdetr",
             "synthesis_enabled": synthesis_enabled,
             "synthesis_bootstrap_required": bootstrap_required,
             "current_iteration": 0,
             "next_stage": "synthesis_bootstrap" if bootstrap_required else "candidate_cache",
             "max_iterations": policy["max_iterations"], "platform": policy["platform"],
             "reproduction_mode": mode,
             "training_base_checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
             "routing_seed_checkpoint": {"path": str(routing_seed), "sha256": _sha(routing_seed)},
             # Retained for existing leaf helpers; training always consumes this value.
             "base_checkpoint": str(checkpoint),
             "policy": str(published / frozen.name),
             "policy_sha256": _sha(frozen),
             "classmap": str(published / classmap.name),
             "routing_policy": str(published / routing_policy.name),
             "routing_policy_sha256": _sha(routing_policy),
             "roles": role_reports, "iterations": {}}
    _json(output / "deft_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--published-output-dir",
        type=Path,
        help="durable destination recorded in state when output is staged elsewhere",
    )
    args = parser.parse_args()
    print(json.dumps(initialize(
        args.config.resolve(),
        args.output_dir.resolve(),
        published_output=(args.published_output_dir.resolve()
                          if args.published_output_dir else None),
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
