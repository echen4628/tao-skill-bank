#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate frozen AnomalyGenNext inputs and reconcile generated OD defects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_inputs(root: Path) -> dict[str, Any]:
    manifest_path = root / "phase1" / "phase1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or not manifest.get("phase2_ready"):
        raise ValueError("frozen-input manifest is not COMPLETE/phase2_ready")
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen input changed or is missing: {path}")
    return manifest


def validate_inputs(args: argparse.Namespace) -> None:
    root = Path(args.inputs_dir)
    manifest = _validate_frozen_inputs(root)
    manifest_path = root / "phase1" / "phase1_manifest.json"
    print(
        f"frozen-input gate PASS: rows={manifest['generator_row_count']} "
        f"sha256={_sha256(manifest_path)}"
    )


def _selected_groups(
    rows: list[dict[str, Any]], datasets: str | None
) -> list[dict[str, Any]]:
    if not datasets:
        return rows
    requested = [value.strip() for value in datasets.split(",") if value.strip()]
    if not requested:
        raise ValueError("--datasets must contain at least one dataset id")
    available = {str(row["dataset_id"]) for row in rows}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"unknown dataset ids {unknown}; available={sorted(available)}")
    requested_set = set(requested)
    return [row for row in rows if str(row["dataset_id"]) in requested_set]


def emit_plan(args: argparse.Namespace) -> None:
    root = Path(args.inputs_dir)
    _validate_frozen_inputs(root)
    rows = json.loads((root / "phase1" / "phase2_plan.json").read_text())
    rows = _selected_groups(rows, args.datasets)
    for row in rows:
        anomaly_types = row.get("anomaly_types", [row["anomaly_type"]])
        print(
            "\t".join(
                (
                    str(row["dataset_id"]),
                    ",".join(str(value) for value in anomaly_types),
                    str(row["testcase"]),
                    str(row["provenance"]),
                    str(row["checkpoint"]),
                    str(row["recipe"]),
                    str(row["real_root"]),
                    str(row["requested_rows"]),
                )
            )
        )


def _csv_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _blocked_count(path: Path) -> int:
    return _csv_count(path) if path.is_file() else 0


def finalize(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs_dir)
    run = Path(args.run_root)
    input_manifest = _validate_frozen_inputs(inputs)
    manifest_path = inputs / "phase1" / "phase1_manifest.json"
    plan = json.loads((inputs / "phase1" / "phase2_plan.json").read_text())
    plan = _selected_groups(plan, args.datasets)
    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    category_ids: dict[str, int] = {}
    generation_status = []
    next_image = 1
    next_annotation = 1

    for group in plan:
        dataset_id = str(group["dataset_id"])
        requested = int(group["requested_rows"])
        expected_types = set(map(str, group.get("anomaly_types", [group["anomaly_type"]])))
        raw = run / "generation" / dataset_id / "raw"
        searched = run / "generation" / dataset_id / "searched"
        raw_count = _csv_count(raw / "texture_ft_generation_result.csv")
        blocked = _blocked_count(raw / "guardrail_blocked.csv")
        if raw_count + blocked != requested:
            raise ValueError(
                f"generation accounting mismatch for {dataset_id}: "
                f"raw={raw_count} blocked={blocked} requested={requested}"
            )

        coco_path = searched / "pseudo_labels" / "coco_annotations.json"
        coco = json.loads(coco_path.read_text())
        local_images = {int(image["id"]): image for image in coco["images"]}
        if len(local_images) != raw_count:
            raise ValueError(
                f"pseudo-label image count mismatch for {dataset_id}: "
                f"coco={len(local_images)} generated={raw_count}"
            )
        local_categories = {int(row["id"]): str(row["name"]) for row in coco["categories"]}
        unexpected = sorted(set(local_categories.values()) - expected_types)
        if unexpected:
            raise ValueError(
                f"unexpected pseudo-label categories for {dataset_id}: {unexpected}; "
                f"expected={sorted(expected_types)}"
            )

        image_map: dict[int, int] = {}
        for old_id, image in sorted(local_images.items()):
            candidate = searched / "reconstructed_image" / str(image["file_name"])
            resolved = candidate if candidate.is_file() else Path(str(image["file_name"]))
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            merged = dict(image)
            merged["id"] = next_image
            merged["file_name"] = str(resolved)
            merged["dataset_id"] = dataset_id
            image_map[old_id] = next_image
            merged_images.append(merged)
            next_image += 1

        per_image: Counter[int] = Counter()
        for annotation in coco["annotations"]:
            old_image = int(annotation["image_id"])
            if old_image not in local_images:
                raise ValueError(f"COCO annotation references unknown image: {annotation}")
            x, y, width, height = map(float, annotation["bbox"])
            image = local_images[old_image]
            if width <= 0 or height <= 0 or x < 0 or y < 0:
                raise ValueError(f"invalid COCO bbox: {annotation}")
            if x + width > float(image["width"]) or y + height > float(image["height"]):
                raise ValueError(f"COCO bbox outside image: {annotation}")
            local_category = int(annotation["category_id"])
            if local_category not in local_categories:
                raise ValueError(f"COCO annotation references unknown category: {annotation}")
            category_name = local_categories[local_category]
            category_id = category_ids.setdefault(category_name, len(category_ids) + 1)
            merged = dict(annotation)
            merged["id"] = next_annotation
            merged["image_id"] = image_map[old_image]
            merged["category_id"] = category_id
            merged_annotations.append(merged)
            per_image[old_image] += 1
            next_annotation += 1
        if any(per_image[image_id] == 0 for image_id in local_images):
            raise ValueError(f"pseudo-label image without annotation in {dataset_id}")

        generation_status.append(
            {
                "dataset_id": dataset_id,
                "anomaly_type": group["anomaly_type"],
                "anomaly_types": sorted(expected_types),
                "requested": requested,
                "generated": raw_count,
                "guardrail_blocked": blocked,
                "pseudo_labeled_images": len(local_images),
                "annotations": len(coco["annotations"]),
            }
        )

    categories = [
        {"id": identifier, "name": name}
        for name, identifier in sorted(category_ids.items(), key=lambda item: item[1])
    ]
    merged = {"images": merged_images, "annotations": merged_annotations, "categories": categories}
    labels = run / "pseudo_labels"
    native_path = labels / "coco_annotations.json"
    _write_json(native_path, merged)
    collapsed = dict(merged)
    collapsed["categories"] = [{"id": 1, "name": "defect"}]
    collapsed["annotations"] = [dict(row, category_id=1) for row in merged_annotations]
    collapsed_path = labels / "coco_annotations_od_defect.json"
    _write_json(collapsed_path, collapsed)

    summary = {
        "schema_version": 1,
        "phase": "synthetic_data_generation",
        "status": "COMPLETE",
        "source_tag": input_manifest["source_tag"],
        "training_eligible": input_manifest["training_eligible"],
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "phase1_manifest": str(manifest_path),
        "phase1_manifest_sha256": _sha256(manifest_path),
        "groups": generation_status,
        "requested_rows": sum(row["requested"] for row in generation_status),
        "generated_images": sum(row["generated"] for row in generation_status),
        "guardrail_blocked": sum(row["guardrail_blocked"] for row in generation_status),
        "pseudo_labeled_images": len(merged_images),
        "annotations": len(merged_annotations),
        "native_categories": sorted(category_ids),
        "native_coco": str(native_path),
        "od_coco": str(collapsed_path),
        "training_pool_mutated": False,
    }
    _write_json(run / "validation_summary.json", summary)
    _write_report(run / "report" / "index.html", summary, generation_status)
    print(
        f"finalize generation PASS: generated={summary['generated_images']} "
        f"images={summary['pseudo_labeled_images']} annotations={summary['annotations']}"
    )


def _write_report(
    path: Path, summary: dict[str, Any], groups: list[dict[str, Any]]
) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{row['dataset_id']}</td><td>{', '.join(row['anomaly_types'])}</td>"
        f"<td>{row['generated']}/{row['requested']}</td>"
        f"<td>{row['annotations']}</td><td>{row['guardrail_blocked']}</td>"
        "</tr>"
        for row in groups
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AnomalyGenNext OD defects</title></head>
<body><h1>AnomalyGenNext OD defects</h1>
<p>Frozen input SHA-256: <code>{summary['input_manifest_sha256']}</code></p>
<p>Generated {summary['generated_images']} images with {summary['annotations']} annotations.</p>
<table><thead><tr><th>Dataset</th><th>Anomaly types</th><th>Generated/requested</th><th>Annotations</th><th>Blocked</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("validate-inputs")
    command.add_argument("--inputs-dir", required=True)
    command.set_defaults(func=validate_inputs)

    command = commands.add_parser("emit-plan")
    command.add_argument("--inputs-dir", required=True)
    command.add_argument("--datasets", default=None)
    command.set_defaults(func=emit_plan)

    command = commands.add_parser("finalize")
    command.add_argument("--inputs-dir", required=True)
    command.add_argument("--run-root", required=True)
    command.add_argument("--datasets", default=None)
    command.set_defaults(func=finalize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
