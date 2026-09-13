#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize SigLIP query crops from strict-FN and loose-FP gap rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from prepare_deft_od_aoi_siglip_candidates import (
    _context_crop,
    _identity_path,
    _metadata,
    _save_crop,
    _source_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-gaps", required=True)
    parser.add_argument("--loose-gaps", required=True)
    parser.add_argument("--kpi-coco", required=True)
    parser.add_argument("--kpi-images-dir", required=True)
    parser.add_argument("--output-crops-dir", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--background-iou-upper", type=float, default=0.05)
    parser.add_argument("--near-miss-iou-upper", type=float, default=0.5)
    return parser.parse_args()


def _read_gap(path: str, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(Path(path).expanduser().resolve()).reset_index(drop=True)
    missing = sorted({"gap_type", "filepath", "bbox", "best_iou"} - set(frame.columns))
    if missing:
        raise ValueError(f"{name} gap parquet lacks columns {missing}")
    return frame


def _image_lookup(coco: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for image in coco.get("images", []):
        file_name = str(image.get("file_name") or "")
        source_path = str(image.get("source_path") or "")
        for key in (
            file_name,
            Path(file_name).name,
            Path(file_name).stem,
            source_path,
            Path(source_path).name,
            Path(source_path).stem,
        ):
            if key:
                output[key] = image
    return output


def _lookup(row: pd.Series, images: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    value = str(row["filepath"])
    return images.get(value) or images.get(Path(value).name) or images.get(Path(value).stem)


def _xyxy_to_xywh(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"invalid gap bbox {value}")
    x0, y0, x1, y1 = [float(item) for item in value]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"non-positive gap bbox {value}")
    return [x0, y0, x1 - x0, y1 - y0]


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.context_scale < 1:
        raise ValueError("--context-scale must be >= 1")
    if args.output_size <= 0:
        raise ValueError("--output-size must be positive")
    if not 0 <= args.background_iou_upper < args.near_miss_iou_upper <= 1:
        raise ValueError("IoU routing bounds are invalid")
    strict = _read_gap(args.strict_gaps, "strict")
    loose = _read_gap(args.loose_gaps, "loose")
    kpi_path = Path(args.kpi_coco).expanduser().resolve()
    kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
    images = _image_lookup(kpi)
    images_dir = Path(args.kpi_images_dir).expanduser().resolve()
    crops_dir = Path(args.output_crops_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    unresolved = 0

    routed: list[tuple[str, int, pd.Series, str, str]] = []
    for gap_row, row in strict.iterrows():
        if str(row["gap_type"]).upper() == "FN":
            routed.append(("strict", int(gap_row), row, "defect", "strict_fn"))
    for gap_row, row in loose.iterrows():
        if str(row["gap_type"]).upper() != "FP":
            continue
        best_iou = float(row["best_iou"])
        if best_iou < args.background_iou_upper:
            routed.append(("loose", int(gap_row), row, "clean", "background_fp"))
        elif best_iou < args.near_miss_iou_upper:
            routed.append(("loose", int(gap_row), row, "defect", "near_miss_fp"))

    for gap_pass, gap_row, row, role, branch in routed:
        image_row = _lookup(row, images)
        if image_row is None:
            unresolved += 1
            continue
        parent = _source_path(image_row, images_dir)
        metadata = _metadata(image_row, defect=role == "defect")
        bbox_xywh = _xyxy_to_xywh(row["bbox"])
        query_id = f"{gap_pass}:{gap_row}:{branch}"
        crop_path = crops_dir / branch / f"{gap_pass}_{gap_row:07d}.png"
        with Image.open(parent) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            crop = _context_crop(oriented, bbox_xywh, args.context_scale)
        _save_crop(crop, crop_path, args.output_size)
        rows.append(
            {
                "filepath": str(crop_path),
                "query_id": query_id,
                "role": role,
                "branch": branch,
                "gap_pass": gap_pass,
                "gap_row": gap_row,
                "parent_filepath": str(_identity_path(image_row, images_dir)),
                "source_image_id": image_row.get("id"),
                "benchmark": metadata["benchmark"],
                "texture": metadata["texture"],
                "defect_type": metadata["defect_type"],
                "generator_type": metadata["generator_type"],
                "bbox_xywh": bbox_xywh,
                "best_iou": float(row["best_iou"]),
                "confidence": float(row.get("confidence", row.get("score", 0.0))),
            }
        )
    if unresolved:
        raise ValueError(
            f"{unresolved} routed gap rows could not be resolved to frozen KPI images"
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("gap routing produced zero SigLIP queries")
    report = {
        "status": "valid",
        "strict_gap_rows": int(len(strict)),
        "loose_gap_rows": int(len(loose)),
        "queries": {str(key): int(value) for key, value in frame["branch"].value_counts().items()},
        "roles": {str(key): int(value) for key, value in frame["role"].value_counts().items()},
        "unresolved_gap_rows": int(unresolved),
        "context_scale": float(args.context_scale),
        "output_size": int(args.output_size),
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
