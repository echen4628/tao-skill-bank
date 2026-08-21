#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize global SigLIP candidate crops from canonical source/clean COCO."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps, ImageStat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-coco", required=True)
    parser.add_argument("--source-images-dir", required=True)
    parser.add_argument("--clean-coco", required=True)
    parser.add_argument("--clean-images-dir", required=True)
    parser.add_argument("--output-crops-dir", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--defect-context-scale", type=float, default=1.5)
    parser.add_argument("--clean-grids", default="1,2")
    parser.add_argument("--output-size", type=int, default=224)
    return parser.parse_args()


def _read_coco(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO {path} lacks array {key!r}")
    return value


def _source_path(image: dict[str, Any], images_dir: Path) -> Path:
    # Prefer a staged runtime view when the caller supplied one. Keep the
    # durable source_path in COCO for identity/provenance, but never force a
    # SLURM allocation to read the hot image from shared storage.
    staged = images_dir / Path(str(image.get("file_name") or "")).name
    if staged.is_file():
        return staged.absolute()
    raw = str(image.get("source_path") or image.get("file_name") or "").strip()
    if not raw:
        raise ValueError("COCO image needs source_path or file_name")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = images_dir / path
    path = path.absolute()
    if not path.is_file():
        raise FileNotFoundError(f"COCO image is missing: {path}")
    return path


def _identity_path(image: dict[str, Any], images_dir: Path) -> Path:
    raw = str(image.get("source_path") or image.get("file_name") or "").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = images_dir / path
    return path.absolute()


def _metadata(image: dict[str, Any], *, defect: bool) -> dict[str, str]:
    nested = image.get("deft_od_aoi")
    nested = nested if isinstance(nested, dict) else {}

    def value(key: str) -> str:
        raw = nested.get(key, image.get(key, ""))
        return str(raw).strip() if raw is not None else ""

    output = {
        "benchmark": value("benchmark"),
        "texture": value("texture"),
        "defect_type": value("defect_type") if defect else "",
        "generator_type": value("generator_type") if defect else "",
    }
    required = ["benchmark", "texture"] + (["defect_type"] if defect else [])
    missing = [key for key in required if not output[key]]
    if missing:
        raise ValueError(f"image {image.get('file_name')!r} lacks metadata {missing}")
    return output


def _context_crop(image: Image.Image, bbox: list[float], scale: float) -> Image.Image:
    x, y, width, height = [float(value) for value in bbox]
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive bbox {bbox}")
    side = max(1, int(math.ceil(max(width, height) * scale)))
    center_x, center_y = x + width / 2.0, y + height / 2.0
    left = int(math.floor(center_x - side / 2.0))
    top = int(math.floor(center_y - side / 2.0))
    right, bottom = left + side, top + side
    clipped = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"bbox context lies outside image: {bbox}")
    visible = image.crop(clipped).convert("RGB")
    mean = tuple(int(round(value)) for value in ImageStat.Stat(visible).mean[:3])
    canvas = Image.new("RGB", (side, side), mean)
    canvas.paste(visible, (clipped[0] - left, clipped[1] - top))
    return canvas


def _grid_boxes(width: int, height: int, grids: list[int]) -> list[tuple[int, int, int, int, int, int, int]]:
    boxes = []
    for grid in grids:
        for row in range(grid):
            for column in range(grid):
                left = round(column * width / grid)
                right = round((column + 1) * width / grid)
                top = round(row * height / grid)
                bottom = round((row + 1) * height / grid)
                boxes.append((grid, row, column, left, top, right, bottom))
    return boxes


def _save_crop(image: Image.Image, path: Path, output_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resized = image.convert("RGB").resize(
        (output_size, output_size), Image.Resampling.BICUBIC
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    resized.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def _base_row(
    *,
    role: str,
    candidate_id: str,
    crop_path: Path,
    parent: Path,
    image: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "filepath": str(crop_path),
        "candidate_id": candidate_id,
        "role": role,
        "parent_filepath": str(parent),
        "source_image_id": image.get("id"),
        "benchmark": metadata["benchmark"],
        "texture": metadata["texture"],
        "defect_type": metadata["defect_type"],
        "generator_type": metadata["generator_type"],
    }


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.defect_context_scale < 1.0:
        raise ValueError("--defect-context-scale must be >= 1")
    if args.output_size <= 0:
        raise ValueError("--output-size must be positive")
    grids = sorted({int(value) for value in str(args.clean_grids).split(",") if value})
    if not grids or any(value <= 0 for value in grids):
        raise ValueError("--clean-grids must contain positive integers")
    source_coco = _read_coco(Path(args.source_coco).expanduser().resolve())
    clean_coco = _read_coco(Path(args.clean_coco).expanduser().resolve())
    if clean_coco["annotations"]:
        raise ValueError("clean COCO must have zero annotations")
    source_images = {image.get("id"): image for image in source_coco["images"]}
    by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source_coco["annotations"]:
        by_image[annotation.get("image_id")].append(annotation)
    if any(not by_image.get(image_id) for image_id in source_images):
        raise ValueError("source COCO contains a boxless image")

    source_dir = Path(args.source_images_dir).expanduser().resolve()
    clean_dir = Path(args.clean_images_dir).expanduser().resolve()
    crops_root = Path(args.output_crops_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []

    for image_id in sorted(source_images, key=str):
        image_row = source_images[image_id]
        parent = _source_path(image_row, source_dir)
        metadata = _metadata(image_row, defect=True)
        with Image.open(parent) as opened:
            # COCO coordinates follow the displayed orientation. JPEG pixel
            # storage may be landscape with an EXIF rotation into portrait.
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for annotation in sorted(by_image[image_id], key=lambda value: str(value.get("id"))):
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"annotation {annotation.get('id')} has invalid bbox")
                annotation_id = annotation.get("id")
                candidate_id = f"defect:{image_id}:{annotation_id}"
                crop_path = crops_root / "defect" / f"image_{image_id}_ann_{annotation_id}.png"
                crop = _context_crop(image, bbox, args.defect_context_scale)
                _save_crop(crop, crop_path, args.output_size)
                row = _base_row(
                    role="defect",
                    candidate_id=candidate_id,
                    crop_path=crop_path,
                    parent=_identity_path(image_row, source_dir),
                    image=image_row,
                    metadata=metadata,
                )
                row.update(
                    {
                        "source_annotation_id": annotation_id,
                        "bbox_xywh": [float(value) for value in bbox],
                        "patch_xyxy": None,
                    }
                )
                rows.append(row)

    for image_row in sorted(clean_coco["images"], key=lambda value: str(value.get("id"))):
        image_id = image_row.get("id")
        parent = _source_path(image_row, clean_dir)
        metadata = _metadata(image_row, defect=False)
        with Image.open(parent) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for grid, row_index, column, left, top, right, bottom in _grid_boxes(
                image.width, image.height, grids
            ):
                candidate_id = f"clean:{image_id}:g{grid}:r{row_index}:c{column}"
                crop_path = (
                    crops_root
                    / "clean"
                    / f"image_{image_id}_g{grid}_r{row_index}_c{column}.png"
                )
                _save_crop(image.crop((left, top, right, bottom)), crop_path, args.output_size)
                row = _base_row(
                    role="clean",
                    candidate_id=candidate_id,
                    crop_path=crop_path,
                    parent=_identity_path(image_row, clean_dir),
                    image=image_row,
                    metadata=metadata,
                )
                row.update(
                    {
                        "source_annotation_id": None,
                        "bbox_xywh": None,
                        "patch_xyxy": [left, top, right, bottom],
                    }
                )
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("candidate preparation produced zero rows")
    report = {
        "status": "valid",
        "source_coco": str(Path(args.source_coco).expanduser().resolve()),
        "clean_coco": str(Path(args.clean_coco).expanduser().resolve()),
        "defect_context_scale": float(args.defect_context_scale),
        "clean_grids": grids,
        "output_size": int(args.output_size),
        "rows": {role: int(count) for role, count in frame["role"].value_counts().items()},
        "parents": {
            role: int(group["parent_filepath"].nunique())
            for role, group in frame.groupby("role")
        },
        "benchmarks": {
            role: {str(key): int(value) for key, value in group["benchmark"].value_counts().items()}
            for role, group in frame.groupby("role")
        },
    }
    return frame, report


def main() -> int:
    try:
        args = parse_args()
        frame, report = prepare(args)
        output = Path(args.output_parquet).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output)
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
