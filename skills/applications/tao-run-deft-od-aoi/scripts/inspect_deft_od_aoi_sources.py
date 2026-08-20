#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory candidate DEFT OD AOI dataset paths without modifying them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLEAN_DIR_NAMES = {"clean", "clean_image", "good", "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        action="append",
        required=True,
        help="COCO JSON or dataset directory; repeat for multiple sources",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--sample-paths", type=int, default=5)
    return parser.parse_args()


def _depth(path: Path, root: Path) -> int:
    return len(path.relative_to(root).parts)


def _looks_like_coco(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("images"), list)
        and isinstance(value.get("annotations"), list)
    )


def _discover_cocos(root: Path, max_depth: int) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" and _looks_like_coco(root) else []
    candidates = []
    for directory, child_dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        depth = _depth(current, root)
        if depth >= max_depth:
            child_dirs[:] = []
        for name in files:
            path = current / name
            if path.suffix.lower() == ".json" and _looks_like_coco(path):
                candidates.append(path.resolve())
    return sorted(candidates)


def _source_path(image: dict[str, Any], coco_path: Path) -> str:
    source = str(image.get("source_path") or "").strip()
    if source:
        return source
    file_name = str(image.get("file_name") or "").strip()
    return str((coco_path.parent / file_name).resolve()) if file_name else ""


def _existing_metadata(image: dict[str, Any], key: str) -> str:
    nested = image.get("deft_od_aoi")
    nested = nested if isinstance(nested, dict) else {}
    value = nested.get(key, image.get(key, ""))
    return str(value).strip() if value is not None else ""


def _summarize_coco(path: Path, sample_paths: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    images = value["images"]
    annotations = value["annotations"]
    annotation_counts: Counter[Any] = Counter(
        annotation.get("image_id") for annotation in annotations
    )
    annotated = sum(bool(annotation_counts.get(image.get("id"))) for image in images)
    boxless = len(images) - annotated
    metadata: dict[str, Counter[str]] = defaultdict(Counter)
    for image in images:
        for key in ("benchmark", "texture", "defect_type", "generator_type"):
            existing = _existing_metadata(image, key)
            if existing:
                metadata[key][existing] += 1
    annotation_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for annotation in annotations:
        for key in ("defect_label", "defect_type", "category_name", "label"):
            label = str(annotation.get(key, "")).strip()
            if label:
                annotation_labels[key][label] += 1
    info = value.get("info") if isinstance(value.get("info"), dict) else {}
    categories = [
        {"id": category.get("id"), "name": category.get("name")}
        for category in value.get("categories", [])
        if isinstance(category, dict)
    ]
    paths = [_source_path(image, path) for image in images]
    return {
        "path": str(path),
        "images": len(images),
        "annotated_images": annotated,
        "boxless_images": boxless,
        "annotations": len(annotations),
        "categories": categories,
        "info_hints": {
            key: info[key]
            for key in ("bench", "source_dataset", "description")
            if key in info
        },
        "existing_metadata": {
            key: dict(counter.most_common(20)) for key, counter in metadata.items()
        },
        "annotation_labels": {
            key: dict(counter.most_common(20))
            for key, counter in annotation_labels.items()
        },
        "sample_source_paths": paths[:sample_paths],
    }


def _discover_clean_dirs(root: Path, max_depth: int) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    output = []
    for directory, child_dirs, files in os.walk(root, followlinks=False):
        path = Path(directory)
        depth = _depth(path, root)
        if depth >= max_depth:
            child_dirs[:] = []
        if path.name.lower() in CLEAN_DIR_NAMES:
            count = sum(Path(name).suffix.lower() in IMAGE_SUFFIXES for name in files)
            if count:
                output.append({"path": str(path.resolve()), "direct_images": count})
    return sorted(output, key=lambda value: value["path"])


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    roots = [Path(raw).expanduser().resolve() for raw in args.dataset_path]
    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"dataset paths do not exist: {missing}")
    datasets = []
    for root in roots:
        cocos = _discover_cocos(root, args.max_depth)
        datasets.append(
            {
                "input_path": str(root),
                "coco_files": [
                    _summarize_coco(path, args.sample_paths) for path in cocos
                ],
                "candidate_clean_directories": _discover_clean_dirs(
                    root, args.max_depth
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "inspected",
        "datasets": datasets,
        "next_step": (
            "Assign KPI/test/mining roles and write dataset_sources.json with "
            "strict named-regex metadata rules; inspection does not infer roles."
        ),
    }


def main() -> int:
    try:
        args = parse_args()
        report = inspect(args)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
