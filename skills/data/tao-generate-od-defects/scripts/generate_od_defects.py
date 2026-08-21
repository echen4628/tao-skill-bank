#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate frozen AnomalyGenNext inputs and reconcile generated OD defects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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
    manifest_path = (
        root / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or not manifest.get("generation_ready"):
        raise ValueError("frozen-input manifest is not COMPLETE/generation_ready")
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen input changed or is missing: {path}")
    return manifest


def validate_inputs(args: argparse.Namespace) -> None:
    root = Path(args.inputs_dir)
    manifest = _validate_frozen_inputs(root)
    manifest_path = (
        root / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    )
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
    rows = json.loads(
        (
            root
            / "prepared_anomalygennext_inputs"
            / "anomalygen_next_generation_plan.json"
        ).read_text()
    )
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


def _runtime_copy(source_value: str, directory: Path) -> Path:
    source = Path(source_value)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:24]
    target = directory / f"{digest}{source.suffix.lower()}"
    if not target.is_file():
        shutil.copy2(source, target)
    return target


def stage_runtime(args: argparse.Namespace) -> None:
    """Materialize a frozen generation-plan subset below node-local storage."""
    prepared = Path(args.inputs_dir)
    _validate_frozen_inputs(prepared)
    plan = json.loads(
        (
            prepared
            / "prepared_anomalygennext_inputs"
            / "anomalygen_next_generation_plan.json"
        ).read_text()
    )
    plan = _selected_groups(plan, args.datasets)
    if not plan:
        raise ValueError("no generation groups selected")
    checkpoint = Path(args.local_checkpoint)
    recipe = Path(args.local_recipe)
    real_root = Path(args.local_real_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not recipe.is_file():
        raise FileNotFoundError(recipe)
    if not real_root.is_dir():
        raise FileNotFoundError(real_root)

    runtime = Path(args.runtime_root)
    runtime.mkdir(parents=True, exist_ok=False)
    tsv_rows: list[str] = []
    summary: list[dict[str, Any]] = []
    for group in plan:
        dataset_id = str(group["dataset_id"])
        group_root = runtime / dataset_id
        images, masks = group_root / "images", group_root / "masks"
        images.mkdir(parents=True)
        masks.mkdir()
        testcase_rows = []
        for line in Path(group["testcase"]).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["image_filename"] = str(
                _runtime_copy(str(row["image_filename"]), images)
            )
            row["mask_filename"] = str(
                _runtime_copy(str(row["mask_filename"]), masks)
            )
            testcase_rows.append(row)
        requested = int(group["requested_rows"])
        if len(testcase_rows) != requested:
            raise ValueError(
                f"runtime testcase count mismatch for {dataset_id}: "
                f"rows={len(testcase_rows)} requested={requested}"
            )
        testcase = group_root / "testcase.jsonl"
        testcase.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in testcase_rows)
        )
        anomaly_types = list(map(str, group.get("anomaly_types", [group["anomaly_type"]])))
        tsv_rows.append(
            "\t".join(
                (
                    dataset_id,
                    ",".join(anomaly_types),
                    str(testcase),
                    str(checkpoint),
                    str(recipe),
                    str(real_root),
                    str(requested),
                )
            )
        )
        summary.append(
            {
                "dataset_id": dataset_id,
                "requested_rows": requested,
                "unique_images": len(list(images.iterdir())),
                "unique_masks": len(list(masks.iterdir())),
                "anomaly_types": anomaly_types,
            }
        )
    output = Path(args.output_tsv)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(tsv_rows) + "\n")
    _write_json(
        runtime / "runtime_summary.json",
        {
            "status": "COMPLETE",
            "groups": summary,
            "requested_rows": sum(row["requested_rows"] for row in summary),
        },
    )
    print(
        f"stage runtime PASS: groups={len(summary)} "
        f"requested={sum(row['requested_rows'] for row in summary)}"
    )


def _csv_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _blocked_count(path: Path) -> int:
    return _csv_count(path) if path.is_file() else 0


def validate_group(args: argparse.Namespace) -> None:
    root = Path(args.group_root)
    raw, searched = root / "raw", root / "searched"
    generated = _csv_count(raw / "texture_ft_generation_result.csv")
    blocked = _blocked_count(raw / "guardrail_blocked.csv")
    if generated + blocked != args.requested:
        raise ValueError(
            f"generation accounting mismatch: generated={generated} "
            f"blocked={blocked} requested={args.requested}"
        )
    coco = json.loads(
        (searched / "pseudo_labels" / "coco_annotations.json").read_text()
    )
    if len(coco.get("images", [])) != generated:
        raise ValueError(
            f"COCO/generated mismatch: coco={len(coco.get('images', []))} "
            f"generated={generated}"
        )
    if generated and not coco.get("annotations"):
        raise ValueError("generated group has no pseudo-label annotations")
    expected_types = {item for item in args.anomaly_types.split(",") if item}
    actual_types = {str(row["name"]) for row in coco.get("categories", [])}
    unexpected = sorted(actual_types - expected_types)
    if unexpected:
        raise ValueError(f"unexpected pseudo-label categories: {unexpected}")
    for image in coco.get("images", []):
        path = searched / "reconstructed_image" / str(image["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
    report = {
        "status": "COMPLETE",
        "dataset_id": args.dataset_id,
        "requested": int(args.requested),
        "generated": generated,
        "guardrail_blocked": blocked,
        "pseudo_labeled_images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": sorted(actual_types),
    }
    _write_json(Path(args.output_json), report)
    print(json.dumps(report, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs_dir)
    run = Path(args.run_root)
    input_manifest = _validate_frozen_inputs(inputs)
    manifest_path = (
        inputs / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    )
    plan = json.loads(
        (
            inputs
            / "prepared_anomalygennext_inputs"
            / "anomalygen_next_generation_plan.json"
        ).read_text()
    )
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
        raw = run / dataset_id / "raw"
        searched = run / dataset_id / "searched"
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
        "schema_version": 2,
        "phase": "anomalygen_next_generation",
        "status": "COMPLETE",
        "source_tag": input_manifest["source_tag"],
        "prepared_inputs_manifest": str(manifest_path),
        "prepared_inputs_manifest_sha256": _sha256(manifest_path),
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
<p>Prepared-input SHA-256: <code>{summary['prepared_inputs_manifest_sha256']}</code></p>
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

    command = commands.add_parser("stage-runtime")
    command.add_argument("--inputs-dir", required=True)
    command.add_argument("--runtime-root", required=True)
    command.add_argument("--local-real-root", required=True)
    command.add_argument("--local-checkpoint", required=True)
    command.add_argument("--local-recipe", required=True)
    command.add_argument("--datasets", default=None)
    command.add_argument("--output-tsv", required=True)
    command.set_defaults(func=stage_runtime)

    command = commands.add_parser("validate-group")
    command.add_argument("--group-root", required=True)
    command.add_argument("--dataset-id", required=True)
    command.add_argument("--requested", type=int, required=True)
    command.add_argument("--anomaly-types", required=True)
    command.add_argument("--output-json", required=True)
    command.set_defaults(func=validate_group)

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
