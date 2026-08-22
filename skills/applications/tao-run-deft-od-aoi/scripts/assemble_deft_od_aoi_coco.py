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
    parser.add_argument(
        "--previous-assembled-coco",
        help="Immediately previous cumulative assembly; omit only for iteration 1.",
    )
    parser.add_argument("--route-manifest", action="append", required=True)
    parser.add_argument(
        "--current-routing-report",
        required=True,
        help="Current committed routing report containing cumulative real/clean totals.",
    )
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


def previous_assembled_records(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    coco = read_json(path)
    categories = coco.get("categories")
    if (
        not isinstance(categories, list)
        or len(categories) != 1
        or categories[0].get("id") != 1
        or categories[0].get("name") != "defect"
    ):
        raise ValueError("previous assembled COCO must have category id 1 named defect")

    boxes_by_image: dict[Any, list[list[float]]] = {}
    for annotation in coco.get("annotations", []):
        if annotation.get("category_id") != 1:
            raise ValueError("previous assembled COCO has a non-defect category")
        boxes_by_image.setdefault(annotation.get("image_id"), []).append(
            [float(value) for value in annotation["bbox"]]
        )

    output = []
    seen_ids: set[Any] = set()
    for image in coco.get("images", []):
        image_id = image.get("id")
        if image_id in seen_ids:
            raise ValueError(f"previous assembled COCO has duplicate image id: {image_id}")
        seen_ids.add(image_id)
        source = image.get("source_path")
        if not source:
            raise ValueError("previous assembled COCO image lacks source_path")
        kind = image.get("deft_od_aoi_kind")
        if kind not in {"real_defect", "clean_negative", "synthetic_defect"}:
            raise ValueError(f"previous assembled COCO has invalid kind: {kind}")
        boxes = boxes_by_image.get(image_id, [])
        if kind == "clean_negative" and boxes:
            raise ValueError("previous clean-negative image has defect boxes")
        if kind != "clean_negative" and not boxes:
            raise ValueError(f"previous {kind} image has no defect boxes")
        output.append(
            {
                "source_path": source_key(source),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "boxes": boxes,
                "kind": kind,
            }
        )
    return output


def expected_cumulative_route_counts(path: str | Path) -> dict[str, int]:
    report = read_json(path)
    output = {}
    for report_key, kind in (
        ("cumulative_real_defectives", "real_defect"),
        ("cumulative_clean_negatives", "clean_negative"),
    ):
        value = report.get(report_key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"routing report lacks valid {report_key}")
        output[kind] = value
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
    previous_coco = getattr(args, "previous_assembled_coco", None)
    output_coco = Path(args.output_coco).expanduser().resolve()
    if previous_coco and Path(previous_coco).expanduser().resolve() == output_coco:
        raise ValueError("previous assembled COCO and output COCO must be different files")

    previous_records = previous_assembled_records(previous_coco)
    records = (
        previous_records
        + route_records(args.route_manifest)
        + synthetic_records(args.synthetic_source)
    )
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = source_key(record["source_path"])
        if key in unique and (
            unique[key].get("boxes") != record.get("boxes")
            or unique[key].get("kind") != record.get("kind")
        ):
            raise ValueError(f"conflicting record for duplicate source image: {key}")
        unique.setdefault(key, record)

    previous_keys = {source_key(record["source_path"]) for record in previous_records}
    missing_previous = previous_keys - set(unique)
    if missing_previous:
        raise ValueError(
            f"cumulative assembly dropped {len(missing_previous)} previous images"
        )

    kind_counts: dict[str, int] = {}
    for record in unique.values():
        kind = str(record.get("kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    expected_route_counts = expected_cumulative_route_counts(
        args.current_routing_report
    )
    for kind, expected in expected_route_counts.items():
        actual = kind_counts.get(kind, 0)
        if actual != expected:
            raise ValueError(
                f"cumulative {kind} count mismatch: expected {expected}, got {actual}; "
                "include the immediately previous assembled COCO"
            )

    images_dir = Path(args.output_images_dir).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
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
    output_coco.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_coco.with_suffix(output_coco.suffix + ".tmp")
    temporary.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_coco)
    report = {
        "images": len(images),
        "annotations": len(annotations),
        "by_kind": dict(sorted(kind_counts.items())),
        "expected_cumulative_route_counts": expected_route_counts,
        "previous_assembled_coco": (
            str(Path(previous_coco).expanduser().absolute()) if previous_coco else None
        ),
        "retained_previous_images": len(previous_keys),
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
