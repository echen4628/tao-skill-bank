#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project a multiclass COCO detector taxonomy to one class without losing provenance.

The projected COCO remains ordinary TAO training input. Original category identity
is retained per annotation under ``annotation.deft`` and in a JSONL sidecar so
future per-defect KPI analysis does not depend on custom COCO fields surviving
every downstream converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


def _defect_type(annotation: dict, original_name: str) -> str:
    nested = annotation.get("deft") if isinstance(annotation.get("deft"), dict) else {}
    for value in (
        nested.get("defect_type"),
        annotation.get("defect_type"),
        annotation.get("defect_label"),
        annotation.get("category_name"),
        annotation.get("label"),
        original_name,
    ):
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return "unknown"


def project(coco: dict, source_tag: str, target_id: int, target_name: str) -> tuple[dict, list[dict]]:
    categories = coco.get("categories")
    annotations = coco.get("annotations")
    images = coco.get("images")
    if not isinstance(categories, list) or not categories:
        raise ValueError("input COCO has no categories")
    if not isinstance(annotations, list) or not isinstance(images, list):
        raise ValueError("input COCO requires images and annotations lists")
    if not all(isinstance(row, dict) and "id" in row for row in categories):
        raise ValueError("every source category must be an object with an id")
    category_names = {row.get("id"): str(row.get("name", "")).strip() for row in categories}
    if len(category_names) != len(categories):
        raise ValueError("source category ids must be unique")
    if any(not name or "\n" in name or "\r" in name for name in category_names.values()):
        raise ValueError("every source category must have a non-empty single-line name")

    projected_annotations: list[dict] = []
    sidecar: list[dict] = []
    for ordinal, annotation in enumerate(annotations):
        original_id = annotation.get("category_id")
        if original_id not in category_names:
            raise ValueError(
                f"annotation {annotation.get('id')} refers to unknown category_id {original_id}"
            )
        original_name = category_names[original_id]
        annotation_id = annotation.get("id", ordinal)
        uid_material = f"{source_tag}\0{annotation_id}\0{annotation.get('image_id')}"
        annotation_uid = hashlib.sha256(uid_material.encode("utf-8")).hexdigest()[:24]
        existing = annotation.get("deft") if isinstance(annotation.get("deft"), dict) else {}
        provenance = {
            **existing,
            "annotation_uid": annotation_uid,
            "source_tag": source_tag,
            "original_annotation_id": annotation_id,
            "original_category_id": original_id,
            "original_category_name": original_name,
            "defect_type": _defect_type(annotation, original_name),
        }
        projected_annotations.append({**annotation, "category_id": target_id, "deft": provenance})
        sidecar.append({
            "annotation_uid": annotation_uid,
            "source_tag": source_tag,
            "image_id": annotation.get("image_id"),
            "annotation_id": annotation_id,
            "projected_category_id": target_id,
            "projected_category_name": target_name,
            "original_category_id": original_id,
            "original_category_name": original_name,
            "defect_type": provenance["defect_type"],
        })

    payload = {k: v for k, v in coco.items() if k not in {"annotations", "categories"}}
    payload["annotations"] = projected_annotations
    payload["categories"] = [{"id": target_id, "name": target_name, "supercategory": target_name}]
    return payload, sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-coco", required=True)
    parser.add_argument("--output-coco", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--classmap-out", default=None)
    parser.add_argument("--kpi-mapping-out", default=None)
    parser.add_argument("--source-tag", default=None)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument("--target-name", default="defect")
    args = parser.parse_args()
    try:
        source = Path(args.input_coco).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"--input-coco does not exist: {source}")
        target_name = args.target_name.strip()
        if not target_name or "\n" in target_name or "\r" in target_name:
            raise ValueError("--target-name must be a non-empty single-line name")
        source_tag = args.source_tag or source.stem
        payload, sidecar = project(
            json.loads(source.read_text(encoding="utf-8")),
            source_tag,
            args.target_id,
            target_name,
        )
        output = Path(args.output_coco).expanduser().resolve()
        metadata = Path(args.metadata_jsonl).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        metadata.write_text("".join(json.dumps(row) + "\n" for row in sidecar), encoding="utf-8")
        if args.classmap_out:
            path = Path(args.classmap_out).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{target_name}\n", encoding="utf-8")
        if args.kpi_mapping_out:
            path = Path(args.kpi_mapping_out).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            aliases = list(dict.fromkeys(
                [target_name] + [row["original_category_name"] for row in sidecar]
            ))
            path.write_text(
                yaml.safe_dump([{target_name: aliases}], sort_keys=False),
                encoding="utf-8",
            )
        print(
            f"projected {len(sidecar)} annotations from {len(json.loads(source.read_text())['categories'])} "
            f"classes to {target_name!r}; provenance={metadata}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
