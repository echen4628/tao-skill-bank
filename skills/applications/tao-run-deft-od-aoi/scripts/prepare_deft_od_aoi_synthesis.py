#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize DEFT AOI false negatives into AnomalyGenNext preparation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FIELDS = ("dataset_id", "texture_id", "defect_class", "fn_mask_source")
AMP_CANDIDATE_TOPN = 3
AMP_MAX_NEIGHBORS_PER_FN = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_plan(path: Path, expected_sha256: str) -> tuple[dict[str, int], str]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"synthetic plan hash mismatch: expected {expected_sha256}, got {actual}"
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("synthetic plan must be a non-empty JSON object")
    plan: dict[str, int] = {}
    for key, value in raw.items():
        anomaly_type = str(key).strip()
        if (not anomaly_type or isinstance(value, bool) or not isinstance(value, int)
                or value < 0):
            raise ValueError(f"invalid synthetic plan entry: {key!r}={value!r}")
        plan[anomaly_type] = value
    if not any(plan.values()):
        raise ValueError("synthetic plan requests zero images")
    return dict(sorted(plan.items())), actual


def _source(images: Path, row: dict[str, Any]) -> Path:
    file_name = str(row.get("file_name") or "").strip()
    if not file_name:
        raise ValueError("KPI COCO image lacks file_name")
    root = images.resolve()
    path = (root / file_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"KPI COCO file_name escapes the configured images root: {file_name}") from exc
    return path


def _image_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index the identities emitted by COCO and the gap-analysis bridge."""
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        aliases = {str(row.get("id"))}
        file_name = str(row.get("file_name") or "").strip()
        if file_name:
            aliases.add(Path(file_name).stem)
        for alias in aliases:
            if not alias or alias == "None":
                raise ValueError("KPI COCO image lacks an id")
            existing = output.get(alias)
            if existing is not None and existing is not row:
                raise ValueError(f"duplicate KPI image identity: {alias}")
            output[alias] = row
    return output


def _xyxy(value: Any) -> tuple[float, float, float, float]:
    values = tuple(map(float, value))
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"invalid xyxy gap box: {value}")
    return values


def _iou(left: tuple[float, ...], annotation: dict[str, Any]) -> float:
    x, y, width, height = map(float, annotation["bbox"])
    right = (x, y, x + width, y + height)
    ix1, iy1, ix2, iy2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + width * height - intersection
    return intersection / union if union else 0.0


def prepare(
    policy_path: Path,
    strict_gaps: Path,
    synthetic_plan_path: Path,
    synthetic_plan_sha256: str,
    output: Path,
    published_output: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    published = (published_output or output).expanduser().resolve()
    plan, plan_sha256 = _synthetic_plan(synthetic_plan_path, synthetic_plan_sha256)
    synthesis = policy["synthesis"]
    if not synthesis.get("enabled"):
        raise ValueError("synthesis is disabled in the frozen policy")
    routes = synthesis.get("routes") or {}
    pool, defect_spec = Path(str(synthesis["pool_dataset_root"])), Path(str(synthesis["defect_spec"]))
    if not pool.is_dir() or not defect_spec.is_file() or not routes:
        raise ValueError("synthesis pool, defect spec, and routes are required")
    for name, route in routes.items():
        checkpoint = Path(str(route.get("checkpoint") or ""))
        recipe = Path(str(route.get("recipe") or ""))
        base = Path(str(route.get("base_checkpoint") or ""))
        vae = Path(str(route.get("vae_path") or ""))
        if not checkpoint.is_file() or not recipe.is_file():
            raise ValueError(f"route {name} needs resolved checkpoint and recipe; run synthesis bootstrap")
        if not base.is_dir() or not vae.is_file():
            raise ValueError(
                f"route {name} needs a generation DCP base checkpoint and Wan2.2 VAE"
            )
    kpi = policy["sources"]["kpi"]
    images_root = Path(kpi["images"])
    coco = json.loads(Path(kpi["coco"]).read_text())
    images = _image_index(coco["images"])
    annotations: dict[str, list[dict[str, Any]]] = {}
    for row in coco.get("annotations", []):
        annotations.setdefault(str(row["image_id"]), []).append(row)
    gaps = pd.read_parquet(strict_gaps)
    required = {"image_id", "filepath", "gap_type", "bbox", "class"}
    if not required.issubset(gaps):
        raise ValueError(f"strict gaps lack {sorted(required - set(gaps))}")
    rows = []
    for gap in gaps[gaps.gap_type.astype(str).str.upper().eq("FN")].to_dict("records"):
        image_id, box = str(gap["image_id"]), _xyxy(gap["bbox"])
        if image_id not in images:
            raise ValueError(f"FN image id is absent from KPI COCO: {image_id}")
        image = images[image_id]
        annotation_image_id = str(image["id"])
        ranked = sorted(((_iou(box, row), row) for row in annotations.get(annotation_image_id, [])),
                        key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.999:
            raise ValueError(f"FN box has no exact KPI annotation match: {image_id} {box}")
        annotation = ranked[0][1]
        metadata = {}
        for source in (image, image.get("deft_od_aoi", {}), annotation,
                       annotation.get("deft_od_aoi", {})):
            metadata.update({key: source[key] for key in FIELDS if key in source})
        dataset = str(metadata.get("dataset_id") or "")
        if not dataset:
            raise ValueError(f"FN metadata is missing dataset_id for {image_id}")
        if dataset not in routes:
            continue
        missing = [key for key in FIELDS if not str(metadata.get(key) or "").strip()]
        if missing:
            raise ValueError(f"FN metadata is incomplete for {image_id}: {missing}")
        mask = Path(str(metadata["fn_mask_source"])).expanduser().resolve()
        if not mask.is_file():
            raise ValueError(f"FN pixel mask is missing: {mask}")
        metadata["fn_mask_source"] = str(mask)
        source_path = _source(images_root, image)
        if source_path != Path(str(gap["filepath"])).resolve():
            raise ValueError(f"gap filepath does not match frozen KPI image: {image_id}")
        rows.append({**gap, "filepath": str(source_path), "split": "kpi", **metadata,
                     "anomaly_type": f"{metadata['texture_id']}+{metadata['defect_class']}"})
    if not rows:
        raise ValueError("strict gaps contain no synthesis-eligible false negatives")
    normalized_rows = pd.DataFrame(rows)
    normalized_rows["_image_sort"] = normalized_rows["image_id"].map(str)
    normalized_rows["_bbox_sort"] = normalized_rows["bbox"].map(
        lambda value: json.dumps([float(item) for item in value], separators=(",", ":"))
    )
    selected_parts: list[pd.DataFrame] = []
    per_type: dict[str, dict[str, int]] = {}
    for anomaly_type, requested in plan.items():
        available = normalized_rows[
            normalized_rows.anomaly_type.astype(str).eq(anomaly_type)
        ].sort_values(["_image_sort", "filepath", "_bbox_sort"], kind="stable")
        selected_pairs = min(len(available), requested // 2)
        if selected_pairs:
            selected_parts.append(available.head(selected_pairs))
        frozen_rows = selected_pairs * 2
        per_type[anomaly_type] = {
            "available_fn_boxes": int(len(available)),
            "requested_images": requested,
            "selected_fn_pairs": selected_pairs,
            "frozen_generator_rows": frozen_rows,
            "bounded_shortfall": requested - frozen_rows,
        }
    if not selected_parts:
        raise ValueError("synthetic plan yielded no synthesis-eligible false negatives")
    selected = pd.concat(selected_parts, ignore_index=True).drop(
        columns=["_image_sort", "_bbox_sort"]
    )
    output.mkdir(parents=True)
    normalized = output / "normalized_fn_gaps.parquet"
    selected.to_parquet(normalized, index=False)
    config = {"source_tag": "deft_od_aoi",
              "gap_parquet": str(published / normalized.name),
              "pool_dataset_root": str(pool.resolve()), "defect_spec": str(defect_spec.resolve()),
              "datasets": {name: {"checkpoint": str(Path(route["checkpoint"]).resolve()),
                                    "recipe": str(Path(route["recipe"]).resolve()),
                                    "base_checkpoint": str(Path(route["base_checkpoint"]).resolve()),
                                    "vae_checkpoint": str(Path(route["vae_path"]).resolve())}
                           for name, route in routes.items()},
              "selection": {"mode": "all_eligible", "datasets": sorted(routes),
                            "mask_sample_seed": 42},
              "embedding": {"model": policy["retrieval"]["model"],
                            "model_path": policy["retrieval"]["model_path"], "batch_size": 64},
              "retrieval": {"metric": "cosine", "candidate_topn": AMP_CANDIDATE_TOPN,
                            "max_neighbors_per_fn": AMP_MAX_NEIGHBORS_PER_FN,
                            "min_similarity": float(synthesis["min_similarity"]),
                            "prior_clean_exclusion_manifest": ""},
              "amp": {"model_id": synthesis["amp_model_id"], "seed": 43},
              "synthetic_plan": {"path": str(synthetic_plan_path.resolve()),
                                 "sha256": plan_sha256, "counts": plan}}
    config_path = output / "anomalygen_filtering.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "COMPLETE",
        "fn_count": len(selected),
        "eligible_fn_count": len(rows),
        "requested_images": sum(plan.values()),
        "frozen_generator_rows": int(len(selected) * 2),
        "bounded_shortfall": int(sum(plan.values()) - len(selected) * 2),
        "synthetic_plan": {"path": str(synthetic_plan_path.resolve()),
                           "sha256": plan_sha256},
        "per_type": per_type,
        "config": str(published / config_path.name),
    }
    (output / "synthesis_request.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--strict-gaps", type=Path, required=True)
    parser.add_argument("--synthetic-plan", type=Path, required=True)
    parser.add_argument("--synthetic-plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-output-dir", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.policy.resolve(), args.strict_gaps.resolve(), args.synthetic_plan.resolve(),
        args.synthetic_plan_sha256, args.output_dir.resolve(),
        args.published_output_dir.resolve() if args.published_output_dir else None,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
