#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage mined images and their source-pool COCO records for RT-DETR."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def _image_column(frame: pd.DataFrame) -> str:
    for name in ("filepath", "source_filepath"):
        if name in frame.columns:
            return name
    raise ValueError("mined parquet requires a filepath or source_filepath column")


def category_contract(coco: dict) -> tuple[list[dict], list[int], list[str]]:
    categories = coco.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("source COCO has no categories")
    ordered = sorted(categories, key=lambda row: row.get("id"))
    ids = [row.get("id") for row in ordered]
    names = [str(row.get("name", "")).strip() for row in ordered]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ValueError("source COCO category ids must be integers")
    if any(not name or "\n" in name or "\r" in name for name in names):
        raise ValueError("source COCO category names must be non-empty and single-line")
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("source COCO category ids and names must be unique")
    if ids not in (list(range(len(ids))), list(range(1, len(ids) + 1))):
        raise ValueError(f"RT-DETR requires dense category ids 0..N-1 or 1..N; got {ids}")
    return ordered, ids, names


def stage(args: argparse.Namespace) -> dict:
    mined_path = Path(args.mined_parquet).expanduser().resolve()
    source_path = Path(args.source_coco).expanduser().resolve()
    output_images = Path(args.output_images_dir).expanduser().resolve()
    output_coco = Path(args.output_coco).expanduser().resolve()
    output_classmap = Path(args.output_classmap).expanduser().resolve()

    frame = pd.read_parquet(mined_path)
    column = _image_column(frame)
    selected_paths = list(dict.fromkeys(
        Path(str(value)).expanduser().resolve() for value in frame[column]
    ))
    if not selected_paths:
        raise ValueError("mined parquet selects zero images")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    categories, category_ids, class_names = category_contract(source)
    images = source.get("images")
    annotations = source.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("source COCO requires images and annotations lists")

    by_basename: dict[str, dict] = {}
    duplicates: set[str] = set()
    for image in images:
        name = Path(str(image.get("file_name", ""))).name
        if not name:
            continue
        if name in by_basename:
            duplicates.add(name)
        by_basename[name] = image
    if duplicates:
        raise ValueError(
            "source COCO has duplicate basenames and cannot be staged flat: "
            f"{sorted(duplicates)[:8]}"
        )

    anns_by_image: dict[object, list[dict]] = {}
    for annotation in annotations:
        anns_by_image.setdefault(annotation.get("image_id"), []).append(annotation)

    output_images.mkdir(parents=True, exist_ok=True)
    output_coco.parent.mkdir(parents=True, exist_ok=True)
    output_classmap.parent.mkdir(parents=True, exist_ok=True)
    staged_images: list[dict] = []
    staged_annotations: list[dict] = []
    missing_images: list[str] = []
    missing_annotations: list[str] = []

    for source_file in selected_paths:
        if not source_file.is_file():
            missing_images.append(str(source_file))
            continue
        source_image = by_basename.get(source_file.name)
        if source_image is None:
            missing_annotations.append(source_file.name)
            continue
        source_annotations = anns_by_image.get(source_image.get("id"), [])
        if not source_annotations:
            missing_annotations.append(source_file.name)
            continue
        image_id = len(staged_images)
        shutil.copy2(source_file, output_images / source_file.name)
        staged_images.append({**source_image, "id": image_id, "file_name": source_file.name})
        for annotation in source_annotations:
            if annotation.get("category_id") not in category_ids:
                raise ValueError(
                    f"annotation {annotation.get('id')} refers to unknown category_id "
                    f"{annotation.get('category_id')}"
                )
            staged_annotations.append({
                **annotation,
                "id": len(staged_annotations),
                "image_id": image_id,
            })

    if not staged_annotations:
        raise ValueError("no mined image resolved to a COCO annotation")

    payload = {k: v for k, v in source.items() if k not in {"images", "annotations", "categories"}}
    payload.update({"images": staged_images, "annotations": staged_annotations, "categories": categories})
    output_coco.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_classmap.write_text("".join(f"{name}\n" for name in class_names), encoding="utf-8")

    report = {
        "mined_unique": len(selected_paths),
        "images_staged": len(staged_images),
        "annotations_written": len(staged_annotations),
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "category_ids": category_ids,
        "class_names": class_names,
        "coco_path": str(output_coco),
        "classmap_path": str(output_classmap),
        "images_dir": str(output_images),
    }
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-parquet", required=True)
    parser.add_argument("--source-coco", required=True)
    parser.add_argument("--output-images-dir", required=True)
    parser.add_argument("--output-coco", required=True)
    parser.add_argument("--output-classmap", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if not 0 <= args.min_success_rate <= 1:
            raise ValueError("--min-success-rate must be within [0, 1]")
        report = stage(args)
        total = report["mined_unique"]
        staged = report["images_staged"]
        rate = staged / total if total else 0.0
        if rate < args.min_success_rate:
            raise ValueError(f"staging success rate {rate:.1%} is below {args.min_success_rate:.1%}")
        print(
            f"staged RT-DETR COCO images={staged}/{total} "
            f"annotations={report['annotations_written']} classes={report['class_names']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
