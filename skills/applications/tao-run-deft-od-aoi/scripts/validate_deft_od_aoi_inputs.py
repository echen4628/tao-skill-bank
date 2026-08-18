#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate DEFT OD AOI binary COCO pools and cross-pool isolation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kpi-coco", required=True)
    parser.add_argument("--kpi-images-dir", required=True)
    parser.add_argument("--test-coco", required=True)
    parser.add_argument("--test-images-dir", required=True)
    parser.add_argument("--source-coco", required=True)
    parser.add_argument("--source-images-dir", required=True)
    parser.add_argument("--clean-coco", required=True)
    parser.add_argument("--clean-images-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_coco(path: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"COCO root must be an object: {resolved}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO {resolved} lacks array {key!r}")
    return resolved, value


def image_identity(image: dict[str, Any], images_dir: Path) -> str:
    raw = image.get("source_path")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = images_dir / path
    else:
        path = images_dir / str(image.get("file_name", ""))
    if not str(image.get("file_name", "")):
        raise ValueError("every COCO image needs source_path or file_name")
    resolved = path.absolute()
    if not resolved.is_file():
        raise FileNotFoundError(f"COCO image is missing: {resolved}")
    return str(resolved)


def metadata(image: dict[str, Any], *, defect: bool) -> None:
    nested = image.get("deft_od_aoi") if isinstance(image.get("deft_od_aoi"), dict) else {}
    required = ["benchmark", "texture"] + (["defect_type"] if defect else [])
    missing = [
        key
        for key in required
        if not str(nested.get(key, image.get(key, ""))).strip()
    ]
    if missing:
        raise ValueError(
            f"image {image.get('file_name')!r} lacks DEFT OD AOI metadata {missing}"
        )


def validate_pool(
    name: str,
    coco: dict[str, Any],
    images_dir: Path,
    *,
    kind: str,
) -> dict[str, Any]:
    categories = coco["categories"]
    if len(categories) != 1 or str(categories[0].get("name", "")).lower() != "defect":
        raise ValueError(f"{name} must declare exactly one category named defect")
    category_id = categories[0].get("id")
    images = coco["images"]
    ids = [image.get("id") for image in images]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} has duplicate image ids")
    identities = [image_identity(image, images_dir) for image in images]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} has duplicate image identities")
    basenames = [Path(str(image.get("file_name", ""))).name for image in images]
    if len(basenames) != len(set(basenames)):
        raise ValueError(f"{name} has duplicate image basenames")
    image_ids = set(ids)
    counts: Counter[Any] = Counter()
    for annotation in coco["annotations"]:
        if annotation.get("image_id") not in image_ids:
            raise ValueError(f"{name} annotation references an unknown image")
        if annotation.get("category_id") != category_id:
            raise ValueError(f"{name} annotation uses a non-defect category")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{name} annotation needs a four-value bbox")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"{name} annotation has a non-positive box")
        counts[annotation.get("image_id")] += 1
    if kind == "clean" and coco["annotations"]:
        raise ValueError("clean pool must have zero annotations")
    if kind == "source" and any(counts[image_id] == 0 for image_id in image_ids):
        raise ValueError("positive source pool contains an image with no defect box")
    for image in images:
        metadata(
            image,
            defect=kind == "source" or counts[image.get("id")] > 0,
        )
    return {
        "images": len(images),
        "annotations": len(coco["annotations"]),
        "identities": identities,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    pools: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for name in ("kpi", "test", "source", "clean"):
        path, coco = read_coco(getattr(args, f"{name}_coco"))
        images_dir = Path(getattr(args, f"{name}_images_dir")).expanduser().resolve()
        if not images_dir.is_dir():
            raise FileNotFoundError(f"{name} image directory is missing: {images_dir}")
        paths[name] = str(path)
        pools[name] = validate_pool(
            name,
            coco,
            images_dir,
            kind="clean" if name == "clean" else ("source" if name == "source" else "eval"),
        )
    sets = {name: set(report.pop("identities")) for name, report in pools.items()}
    overlaps = {}
    names = sorted(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = sorted(sets[left] & sets[right])
            if common:
                overlaps[f"{left}:{right}"] = common[:10]
    if overlaps:
        raise ValueError(f"DEFT OD AOI pools overlap: {overlaps}")
    report = {"status": "valid", "paths": paths, "pools": pools, "overlaps": {}}
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    try:
        report = run(parse_args())
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
