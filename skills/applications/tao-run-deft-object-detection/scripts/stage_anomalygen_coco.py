#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage validated AnomalyGenNext COCO for one DEFT OD training iteration.

The generator deliberately leaves its output outside the training pool. This
script is the explicit application boundary that admits a validated run into
training: it checks the immutable generation summary, copies images into the
iteration tree, assigns collision-proof
basenames, and projects every annotation onto the detector class approved
during preflight.

It emits COCO, not ODVG. Run TAO Data Services ``annotations convert`` on the
output before appending the resulting ODVG source to Grounding DINO.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def stage(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.validation_summary).expanduser().resolve()
    source_coco = Path(args.source_coco).expanduser().resolve()
    output_images = Path(args.output_images_dir).expanduser().resolve()
    output_coco = Path(args.output_coco).expanduser().resolve()
    target_class = args.target_class.strip()
    if not target_class:
        raise ValueError("--target-class must be non-empty")

    summary = _load_object(summary_path)
    if summary.get("status") != "COMPLETE":
        raise ValueError("AnomalyGenNext validation summary is not COMPLETE")
    if summary.get("training_pool_mutated") is not False:
        raise ValueError(
            "generation summary must still report training_pool_mutated=false before staging"
        )
    source_tag = str(summary.get("source_tag", ""))
    if not source_tag.strip():
        raise ValueError("generation summary must contain a non-empty source_tag")
    recorded_coco = summary.get("od_coco")
    if recorded_coco and Path(str(recorded_coco)).expanduser().resolve() != source_coco:
        raise ValueError(
            f"--source-coco differs from validation_summary.od_coco: {source_coco} != "
            f"{Path(str(recorded_coco)).expanduser().resolve()}"
        )

    coco = _load_object(source_coco)
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO must contain images and annotations lists")
    image_ids = {int(row["id"]) for row in images}
    if len(image_ids) != len(images):
        raise ValueError("COCO image ids must be unique")

    output_images.mkdir(parents=True, exist_ok=True)
    staged_images: list[dict[str, Any]] = []
    for row in sorted(images, key=lambda item: int(item["id"])):
        image_id = int(row["id"])
        source = Path(str(row["file_name"])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower() or ".png"
        name = f"synthetic_{image_id:08d}{suffix}"
        destination = output_images / name
        shutil.copy2(source, destination)
        staged = dict(row)
        staged["file_name"] = name
        staged_images.append(staged)

    staged_annotations: list[dict[str, Any]] = []
    per_image = {image_id: 0 for image_id in image_ids}
    for row in annotations:
        image_id = int(row["image_id"])
        if image_id not in image_ids:
            raise ValueError(f"annotation references unknown image id {image_id}")
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotation has invalid bbox: {row}")
        _, _, width, height = map(float, bbox)
        if width <= 0 or height <= 0:
            raise ValueError(f"annotation has non-positive bbox: {row}")
        staged = dict(row)
        staged["category_id"] = 1
        staged_annotations.append(staged)
        per_image[image_id] += 1
    missing = sorted(image_id for image_id, count in per_image.items() if count == 0)
    if missing:
        raise ValueError(f"generated images have no annotations: {missing[:10]}")

    staged_coco = {
        "images": staged_images,
        "annotations": staged_annotations,
        "categories": [{"id": 1, "name": target_class}],
    }
    output_coco.parent.mkdir(parents=True, exist_ok=True)
    output_coco.write_text(json.dumps(staged_coco, indent=2) + "\n", encoding="utf-8")

    report = {
        "status": "COMPLETE",
        "source_tag": source_tag,
        "target_class": target_class,
        "source_validation_summary": str(summary_path),
        "source_coco": str(source_coco),
        "output_coco": str(output_coco),
        "output_images_dir": str(output_images),
        "images_staged": len(staged_images),
        "annotations_staged": len(staged_annotations),
        "training_pool_mutated": True,
    }
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--source-coco", required=True)
    parser.add_argument("--output-images-dir", required=True)
    parser.add_argument("--output-coco", required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        report = stage(parse_args())
        print(
            "staged AnomalyGenNext training data "
            f"images={report['images_staged']} annotations={report['annotations_staged']} "
            f"class={report['target_class']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
