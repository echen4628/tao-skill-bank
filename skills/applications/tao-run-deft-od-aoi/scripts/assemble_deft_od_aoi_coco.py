#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble cumulative admitted DEFT OD AOI data into one binary COCO dataset."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", action="append", required=True)
    parser.add_argument(
        "--synthetic-source",
        action="append",
        default=[],
        metavar="COCO_JSON::IMAGES_DIR",
    )
    parser.add_argument("--output-coco", required=True)
    parser.add_argument("--output-images-dir", required=True)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def source_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def route_records(paths: list[str]) -> list[dict[str, Any]]:
    output = []
    for raw_path in paths:
        records = read_json(raw_path)
        if not isinstance(records, list):
            raise ValueError(f"route manifest must be an array: {raw_path}")
        for record in records:
            if not isinstance(record, dict) or not record.get("source_path"):
                raise ValueError(f"invalid route record in {raw_path}")
            boxes = record.get("boxes")
            if not isinstance(boxes, list):
                raise ValueError(f"route record lacks boxes in {raw_path}")
            if not boxes and not record.get("allow_empty_annotations"):
                raise ValueError("an empty route record must be an admitted clean negative")
            output.append(record)
    return output


def synthetic_records(specs: list[str]) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        if "::" not in spec:
            raise ValueError("synthetic-source must be COCO_JSON::IMAGES_DIR")
        coco_raw, images_raw = spec.split("::", 1)
        coco = read_json(coco_raw)
        images_dir = Path(images_raw).expanduser().resolve()
        categories = coco.get("categories")
        if not isinstance(categories, list) or len(categories) != 1:
            raise ValueError("synthetic COCO must have exactly one category")
        category_id = categories[0].get("id")
        annotations: dict[Any, list[list[float]]] = {}
        for annotation in coco.get("annotations", []):
            if annotation.get("category_id") != category_id:
                raise ValueError("synthetic COCO has a non-defect category")
            annotations.setdefault(annotation.get("image_id"), []).append(
                [float(value) for value in annotation["bbox"]]
            )
        for image in coco.get("images", []):
            boxes = annotations.get(image.get("id"), [])
            if not boxes:
                raise ValueError("synthetic COCO image has no defect annotation")
            raw_source = image.get("source_path") or images_dir / str(image["file_name"])
            source = source_key(raw_source)
            output.append(
                {
                    "source_path": source,
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "boxes": boxes,
                    "kind": "synthetic_defect",
                }
            )
    return output


def materialize(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"training image is missing: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if (
            destination.is_file()
            and mode == "copy"
            and filecmp.cmp(source, destination, shallow=False)
        ):
            return
        raise FileExistsError(f"refusing to replace conflicting output image: {destination}")
    if mode == "symlink":
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = route_records(args.route_manifest) + synthetic_records(args.synthetic_source)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = source_key(record["source_path"])
        if key in unique and unique[key].get("boxes") != record.get("boxes"):
            raise ValueError(f"conflicting boxes for duplicate source image: {key}")
        unique.setdefault(key, record)

    images_dir = Path(args.output_images_dir).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    kind_counts: dict[str, int] = {}
    annotation_id = 1
    for image_id, (key, record) in enumerate(sorted(unique.items()), start=1):
        source = Path(key)
        suffix = source.suffix.lower() or ".png"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        file_name = f"deft_od_aoi_{digest}{suffix}"
        materialize(source, images_dir / file_name, args.link_mode)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(record["width"]),
                "height": int(record["height"]),
                "source_path": key,
                "deft_od_aoi_kind": record.get("kind", "unknown"),
            }
        )
        kind = str(record.get("kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        for bbox in record.get("boxes", []):
            x, y, width, height = [float(value) for value in bbox]
            if width <= 0 or height <= 0:
                raise ValueError(f"non-positive training box for {key}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "defect", "supercategory": "defect"}],
    }
    output_coco = Path(args.output_coco).expanduser().resolve()
    output_coco.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_coco.with_suffix(output_coco.suffix + ".tmp")
    temporary.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_coco)
    report = {
        "images": len(images),
        "annotations": len(annotations),
        "by_kind": dict(sorted(kind_counts.items())),
        "link_mode": args.link_mode,
        "output_coco": str(output_coco),
        "output_images_dir": str(images_dir),
    }
    report_path = output_coco.with_name("assembly_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    try:
        print(json.dumps(run(parse_args()), indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
