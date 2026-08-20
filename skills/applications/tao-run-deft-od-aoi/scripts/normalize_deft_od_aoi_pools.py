#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared COCO rewrite helpers used by prepare_deft_od_aoi_sources.

This module is not an intake path. For mixed or sharded datasets, author
dataset_sources.json and run prepare_deft_od_aoi_sources.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORY = [{"id": 1, "name": "defect"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kpi-coco", required=True)
    parser.add_argument("--kpi-images-dir", required=True)
    parser.add_argument("--test-coco", required=True)
    parser.add_argument("--test-images-dir", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--pool-root",
        help="Root containing benchmark subdirectories with COCO split files.",
    )
    inputs.add_argument(
        "--pool-coco",
        action="append",
        help="Mixed positive/clean COCO shard; repeat for multiple shards.",
    )
    parser.add_argument(
        "--pool-splits",
        default="train,mine",
        help="Comma-separated filenames discovered under --pool-root.",
    )
    parser.add_argument(
        "--pool-images-dir",
        default=None,
        help="Fallback image root for relative pool paths; absolute source_path wins.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and report normalization without writing output files.",
    )
    return parser.parse_args()


def _read_coco(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO {path} lacks array {key!r}")
    return value


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "dataset"


def _source_path(image: dict[str, Any], images_dir: Path, coco_path: Path) -> Path:
    raw = image.get("source_path") or image.get("file_name")
    if not str(raw or "").strip():
        raise ValueError(f"COCO image in {coco_path} needs source_path or file_name")
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        resolved = candidate.absolute()
    else:
        options = [images_dir / candidate, coco_path.parent / candidate]
        if candidate.parent == Path("."):
            options.append(coco_path.parent / "images" / candidate)
        resolved = next((path for path in options if path.is_file()), options[0]).absolute()
    if not resolved.is_file():
        raise FileNotFoundError(f"COCO image is missing: {resolved}")
    return resolved


def _labels(annotations: list[dict[str, Any]]) -> list[str]:
    values = set()
    for annotation in annotations:
        for key in ("defect_label", "defect_type", "category_name", "label"):
            value = str(annotation.get(key, "")).strip()
            if value:
                values.add(value)
                break
    return sorted(values)


def _metadata(
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    info: dict[str, Any],
    shard_name: str,
    sources: Counter[str],
) -> dict[str, str]:
    nested = image.get("deft_od_aoi")
    nested = nested if isinstance(nested, dict) else {}

    def existing(key: str) -> str:
        value = nested.get(key, image.get(key, ""))
        return str(value).strip() if value is not None else ""

    benchmark = existing("benchmark")
    if benchmark:
        sources["benchmark:input"] += 1
    else:
        benchmark = str(info.get("bench") or info.get("source_dataset") or "").strip()
        if benchmark:
            sources["benchmark:coco_info"] += 1
        else:
            benchmark = shard_name
            sources["benchmark:shard"] += 1

    texture = existing("texture")
    if texture:
        sources["texture:input"] += 1
    else:
        for key in ("product", "object", "class_name", "surface"):
            texture = str(image.get(key, "")).strip()
            if texture:
                sources[f"texture:image_{key}"] += 1
                break
        if not texture:
            texture = benchmark
            sources["texture:benchmark_fallback"] += 1

    result = {"benchmark": benchmark, "texture": texture}
    if annotations:
        defect_type = existing("defect_type")
        if defect_type:
            sources["defect_type:input"] += 1
        else:
            labels = _labels(annotations)
            if labels:
                defect_type = "+".join(labels)
                sources["defect_type:annotation_label"] += 1
            else:
                defect_type = "defect"
                sources["defect_type:default"] += 1
        result["defect_type"] = defect_type
        generator_type = existing("generator_type")
        if generator_type:
            result["generator_type"] = generator_type
    return result


def _annotation(
    value: dict[str, Any], image_id: int, annotation_id: int, source: Path
) -> dict[str, Any]:
    bbox = value.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"annotation in {source} needs a four-value bbox")
    box = [float(item) for item in bbox]
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"annotation in {source} has a non-positive bbox")
    output = dict(value)
    output.update(
        {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": box,
            "area": float(value.get("area", box[2] * box[3])),
            "iscrowd": int(value.get("iscrowd", 0)),
        }
    )
    return output


def _image(
    value: dict[str, Any],
    *,
    image_id: int,
    file_name: str,
    source_path: Path,
    metadata: dict[str, str],
    source: Path,
) -> dict[str, Any]:
    width = int(value.get("width", 0))
    height = int(value.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"image in {source} needs positive width and height")
    output = dict(value)
    output.update(
        {
            "id": image_id,
            "file_name": file_name,
            "source_path": str(source_path),
            "width": width,
            "height": height,
            "deft_od_aoi": metadata,
        }
    )
    return output


def _unique_name(preferred: str, used: set[str]) -> str:
    path = Path(preferred)
    stem = _slug(path.stem)
    suffix = path.suffix.lower()
    candidate = f"{stem}{suffix}"
    index = 1
    while candidate in used:
        candidate = f"{stem}_{index:04d}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _normalize_eval(
    name: str,
    coco_path: Path,
    images_dir: Path,
    sources: Counter[str],
) -> tuple[dict[str, Any], set[str]]:
    coco = _read_coco(coco_path)
    by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation.get("image_id")].append(annotation)
    info = coco.get("info") if isinstance(coco.get("info"), dict) else {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    identities: set[str] = set()
    used_names: set[str] = set()
    for source_image in coco["images"]:
        original_id = source_image.get("id")
        image_annotations = by_image.get(original_id, [])
        path = _source_path(source_image, images_dir, coco_path)
        identity = str(path)
        if identity in identities:
            raise ValueError(f"duplicate {name} image identity: {identity}")
        identities.add(identity)
        file_name = _unique_name(str(source_image.get("file_name") or path.name), used_names)
        image_id = len(images) + 1
        meta = _metadata(source_image, image_annotations, info, name, sources)
        images.append(
            _image(
                source_image,
                image_id=image_id,
                file_name=file_name,
                source_path=path,
                metadata=meta,
                source=coco_path,
            )
        )
        for value in image_annotations:
            annotations.append(_annotation(value, image_id, len(annotations) + 1, coco_path))
    return {"images": images, "annotations": annotations, "categories": CATEGORY}, identities


def _discover_shards(args: argparse.Namespace) -> list[Path]:
    if args.pool_coco:
        shards = [Path(value).expanduser().resolve() for value in args.pool_coco]
    else:
        root = Path(args.pool_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"pool root is missing: {root}")
        names = [value.strip() for value in args.pool_splits.split(",") if value.strip()]
        if not names:
            raise ValueError("pool_splits cannot be empty")
        shards = []
        for name in names:
            filename = name if name.endswith(".json") else f"{name}.json"
            direct = root / filename
            if direct.is_file():
                shards.append(direct)
            shards.extend(sorted(path for path in root.glob(f"*/{filename}") if path.is_file()))
    unique = sorted(set(shards))
    if not unique:
        raise ValueError("no pool COCO shards were found")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pool COCO shards are missing: {missing}")
    return unique


def _normalize_pool(
    shards: list[Path],
    pool_images_dir: Path | None,
    sources: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any], set[str], dict[str, Any]]:
    outputs = {
        "source": {"images": [], "annotations": [], "categories": CATEGORY},
        "clean": {"images": [], "annotations": [], "categories": CATEGORY},
    }
    names = {"source": set(), "clean": set()}
    identities: dict[str, str] = {}
    shard_reports = []
    for shard in shards:
        coco = _read_coco(shard)
        info = coco.get("info") if isinstance(coco.get("info"), dict) else {}
        shard_name = str(info.get("bench") or shard.parent.name or shard.stem)
        images_dir = pool_images_dir or shard.parent
        by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            by_image[annotation.get("image_id")].append(annotation)
        counts = Counter()
        for source_image in coco["images"]:
            image_annotations = by_image.get(source_image.get("id"), [])
            kind = "source" if image_annotations else "clean"
            path = _source_path(source_image, images_dir, shard)
            identity = str(path)
            previous = identities.get(identity)
            if previous:
                if previous != kind:
                    raise ValueError(
                        f"image is both defective and clean across shards: {identity}"
                    )
                counts["duplicates_skipped"] += 1
                continue
            identities[identity] = kind
            output = outputs[kind]
            image_id = len(output["images"]) + 1
            preferred = f"{_slug(shard_name)}__{Path(str(source_image.get('file_name') or path.name)).name}"
            file_name = _unique_name(preferred, names[kind])
            meta = _metadata(source_image, image_annotations, info, shard_name, sources)
            output["images"].append(
                _image(
                    source_image,
                    image_id=image_id,
                    file_name=file_name,
                    source_path=path,
                    metadata=meta,
                    source=shard,
                )
            )
            for value in image_annotations:
                output["annotations"].append(
                    _annotation(value, image_id, len(output["annotations"]) + 1, shard)
                )
            counts[kind] += 1
            counts["boxes"] += len(image_annotations)
        shard_reports.append({"path": str(shard), **dict(sorted(counts.items()))})
    return outputs["source"], outputs["clean"], set(identities), {"shards": shard_reports}


def normalize(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if args.check_only and args.output_dir:
        raise ValueError("use either --check-only or --output-dir, not both")
    if not args.check_only and not args.output_dir:
        raise ValueError("provide --check-only or --output-dir")
    sources: Counter[str] = Counter()
    kpi_path = Path(args.kpi_coco).expanduser().resolve()
    test_path = Path(args.test_coco).expanduser().resolve()
    kpi, kpi_ids = _normalize_eval(
        "kpi", kpi_path, Path(args.kpi_images_dir).expanduser().resolve(), sources
    )
    test, test_ids = _normalize_eval(
        "test", test_path, Path(args.test_images_dir).expanduser().resolve(), sources
    )
    pool_images = (
        Path(args.pool_images_dir).expanduser().resolve() if args.pool_images_dir else None
    )
    shards = _discover_shards(args)
    source, clean, pool_ids, pool_report = _normalize_pool(shards, pool_images, sources)
    overlaps = {
        "kpi:test": len(kpi_ids & test_ids),
        "kpi:pool": len(kpi_ids & pool_ids),
        "test:pool": len(test_ids & pool_ids),
    }
    overlaps = {key: value for key, value in overlaps.items() if value}
    if overlaps:
        raise ValueError(f"DEFT OD AOI pools overlap: {overlaps}")
    documents = {"kpi": kpi, "test": test, "source": source, "clean": clean}
    report = {
        "status": "valid",
        "inputs": {
            "kpi_coco": str(kpi_path),
            "test_coco": str(test_path),
            "pool_shards": [str(path) for path in shards],
        },
        "outputs": {
            name: {"images": len(value["images"]), "annotations": len(value["annotations"])}
            for name, value in documents.items()
        },
        "metadata_sources": dict(sorted(sources.items())),
        "overlaps": {},
        **pool_report,
    }
    return documents, report


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to replace different normalized artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    try:
        args = parse_args()
        documents, report = normalize(args)
        if args.output_dir:
            output = Path(args.output_dir).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            for name, value in documents.items():
                _write_json(output / f"{name}.json", value)
            _write_json(output / "normalization_report.json", report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
