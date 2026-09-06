#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data, checkpoint, and serialization helpers for the YOLO leaf skill."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_KEY = "metrics/mAP50(B)"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary_coco(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    categories = document.get("categories", [])
    if categories != [{"id": 1, "name": "defect"}] and not (
        len(categories) == 1
        and int(categories[0].get("id", -1)) == 1
        and categories[0].get("name") == "defect"
    ):
        raise ValueError(f"{path}: require exactly category id 1 named defect")
    images = document.get("images", [])
    image_ids = [int(row["id"]) for row in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{path}: image ids are not unique")
    known = set(image_ids)
    for annotation in document.get("annotations", []):
        if int(annotation["image_id"]) not in known:
            raise ValueError(f"{path}: annotation references an unknown image")
        if int(annotation["category_id"]) != 1:
            raise ValueError(f"{path}: annotation is outside category 1")
        _, _, width, height = [float(value) for value in annotation["bbox"]]
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}: annotation has a non-positive box")
    return document


def _source(image: dict[str, Any], annotation_path: Path, image_root: Path) -> Path:
    raw = image.get("source_path") or image.get("file_name")
    if not raw:
        raise ValueError(f"image {image.get('id')} has no source path")
    source = Path(str(raw)).expanduser()
    if not source.is_absolute():
        source = image_root / source
    if not source.is_file():
        raise FileNotFoundError(source)
    return source.resolve()


def _write_yolo_label(
    path: Path, annotations: list[dict[str, Any]], width: int, height: int
) -> int:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions {width}x{height}")
    lines: list[str] = []
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)):
            continue
        x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + box_width))
        y2 = max(0.0, min(float(height), y + box_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"box collapses after clipping for {path}")
        lines.append(
            "0 "
            f"{((x1 + x2) / 2) / width:.8f} "
            f"{((y1 + y2) / 2) / height:.8f} "
            f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def stage_coco(
    annotation_path: Path,
    image_root: Path,
    split_root: Path,
    *,
    preserve_ids: bool,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("stage workers must be positive")
    annotation_path = annotation_path.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    document = load_binary_coco(annotation_path)
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    copy_jobs: list[tuple[Path, Path]] = []
    mapping: list[dict[str, Any]] = []
    boxes = 0
    clean_images = 0
    for row_index, image in enumerate(document.get("images", []), start=1):
        original_id = int(image["id"])
        staged_id = original_id if preserve_ids else row_index
        source = _source(image, annotation_path, image_root)
        suffix = source.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        staged_name = f"{staged_id:012d}{suffix}"
        copy_jobs.append((source, images_dir / staged_name))
        annotations = annotations_by_image.get(original_id, [])
        clean_images += not annotations
        boxes += _write_yolo_label(
            labels_dir / f"{Path(staged_name).stem}.txt",
            annotations,
            int(image.get("width", 0)),
            int(image.get("height", 0)),
        )
        mapping.append(
            {
                "original_image_id": original_id,
                "staged_image_id": staged_id,
                "original_file_name": str(image.get("file_name", "")),
                "staged_file_name": staged_name,
            }
        )

    def copy_one(pair: tuple[Path, Path]) -> int:
        shutil.copyfile(*pair)
        return pair[1].stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        copied_bytes = sum(pool.map(copy_one, copy_jobs))
    atomic_json(split_root / "image_mapping.json", mapping)
    report = {
        "source_coco": str(annotation_path),
        "images": len(mapping),
        "boxes": boxes,
        "clean_images": clean_images,
        "copied_bytes": copied_bytes,
        "preserved_image_ids": preserve_ids,
    }
    atomic_json(split_root / "stage_report.json", report)
    return report


def write_dataset_yaml(
    path: Path, *, train_images: Path | None, eval_images: Path
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(path.parent),
                "train": str(train_images or eval_images),
                "val": str(eval_images),
                "names": {0: "defect"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("training curve is empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_results(
    prior_path: Path | None, current_path: Path
) -> tuple[list[dict[str, str]], int]:
    current = read_results(current_path)
    if prior_path is None:
        combined = current
        prior_last_epoch = 0
    else:
        prior = read_results(prior_path)
        prior_last_epoch = max((int(float(row["epoch"])) for row in prior), default=0)
        by_epoch = {int(float(row["epoch"])): row for row in current}
        # Completed rows from the durable prior curve remain authoritative if
        # Ultralytics repeats them while restoring a run.
        by_epoch.update({int(float(row["epoch"])): row for row in prior})
        combined = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    reported = [int(float(row["epoch"])) for row in combined]
    if reported != list(range(1, len(combined) + 1)):
        raise ValueError(f"combined training curve has an epoch gap: {reported}")
    return combined, prior_last_epoch


def select_row(rows: list[dict[str, str]]) -> tuple[int, dict[str, str]]:
    if not rows or METRIC_KEY not in rows[0]:
        raise ValueError(f"training curve has no {METRIC_KEY}")
    return max(
        enumerate(rows),
        key=lambda item: (float(item[1][METRIC_KEY]), -item[0]),
    )


def predictions_to_kitti(
    coco_path: Path, predictions_path: Path, output_dir: Path, minimum_score: float
) -> dict[str, Any]:
    document = load_binary_coco(coco_path)
    images = {int(row["id"]): row for row in document["images"]}
    stems = {image_id: Path(str(row["file_name"])).stem for image_id, row in images.items()}
    if len(stems.values()) != len(set(stems.values())):
        raise ValueError("evaluation image stems must be unique")
    grouped: dict[int, list[tuple[float, int, str]]] = {
        image_id: [] for image_id in images
    }
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    dropped = 0
    for order, prediction in enumerate(predictions):
        image_id = int(prediction["image_id"])
        if image_id not in images:
            raise ValueError(f"prediction references unknown image {image_id}")
        if int(prediction.get("category_id", 1)) != 1:
            raise ValueError("prediction is outside category 1")
        score = float(prediction["score"])
        if score < minimum_score:
            continue
        x, y, width, height = [float(value) for value in prediction["bbox"]]
        if width <= 0 or height <= 0:
            dropped += 1
            continue
        line = (
            "defect 0.0 0 0.0 "
            f"{x:.6f} {y:.6f} {x + width:.6f} {y + height:.6f} "
            f"0 0 0 0 0 0 0 {score:.9f}"
        )
        grouped[image_id].append((score, order, line))
    output_dir.mkdir(parents=True, exist_ok=False)
    kept = 0
    for image_id, stem in stems.items():
        ordered = sorted(grouped[image_id], key=lambda item: (-item[0], item[1]))
        kept += len(ordered)
        (output_dir / f"{stem}.txt").write_text(
            "\n".join(row[2] for row in ordered) + ("\n" if ordered else ""),
            encoding="utf-8",
        )
    return {
        "images": len(images),
        "input_predictions": len(predictions),
        "kept_predictions": kept,
        "dropped_invalid_boxes": dropped,
        "minimum_score": minimum_score,
    }


def common_coco_score(
    ground_truth: Path, predictions: Path, output: Path
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        faster = False
    except ModuleNotFoundError:
        from faster_coco_eval import COCO, COCOeval_faster

        faster = True
    normalized = []
    for prediction in json.loads(predictions.read_text(encoding="utf-8")):
        normalized.append({**prediction, "image_id": int(prediction["image_id"]), "category_id": 1})
    normalized_path = output.parent / "predictions.json"
    atomic_json(normalized_path, normalized)
    coco = COCO(str(ground_truth))
    if not normalized:
        metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR300": 0.0, "detections": 0}
    else:
        detections = coco.loadRes(str(normalized_path))
        evaluator = (
            COCOeval_faster(coco, detections, iouType="bbox")
            if faster
            else COCOeval(coco, detections, "bbox")
        )
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        precision = evaluator.eval["precision"][:, :, :, 0, -1]
        recall = evaluator.eval["recall"][:, :, 0, -1]

        def mean_valid(values: Any) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else 0.0

        metrics = {
            "AP": mean_valid(precision),
            "AP50": mean_valid(precision[0]),
            "AP75": mean_valid(precision[5]),
            "AR300": mean_valid(recall),
            "detections": len(normalized),
        }
    atomic_json(output, metrics)
    return metrics
