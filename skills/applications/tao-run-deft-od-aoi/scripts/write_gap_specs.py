#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write loose and strict DEFT OD AOI object-detection gap specs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from deft_od_aoi_policy import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--ground-truth-ann-path", required=True)
    parser.add_argument(
        "--ground-truth-coco",
        default=None,
        help="Optional frozen binary COCO to project into the KITTI label directory before writing specs.",
    )
    parser.add_argument("--inference-ann-path", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kpi", required=True)
    parser.add_argument(
        "--analyze-binary",
        action="store_true",
        help="Run the binary KITTI matching implementation and write both artifact sets after spec generation.",
    )
    return parser.parse_args()


def _absolute(raw: str) -> str:
    return str(Path(raw).expanduser().resolve())


def build_spec(args: argparse.Namespace, policy: dict, kind: str) -> dict:
    gap = policy["gap"]
    confidence = gap[f"{kind}_confidence"]
    return {
        "ground_truth_ann_path": _absolute(args.ground_truth_ann_path),
        "inference_ann_path": _absolute(args.inference_ann_path),
        "images_dir": _absolute(args.images_dir),
        "results_dir": str(Path(args.output_dir).expanduser().resolve() / kind),
        "kpi": f"{args.kpi}_{kind}",
        "input_format": "kitti",
        "iou_threshold": gap["match_iou"],
        "conf_threshold": confidence,
        "min_area": 0,
        "class_mapping": {},
        "weak_thresholds": {
            policy["task"]["class_name"]: {
                "recall": gap["weak_recall_threshold"],
                "ap50": gap["weak_ap50_threshold"],
            }
        },
        "default_ap50_threshold": 0.0,
        "default_recall_threshold": 0.0,
        "default_precision_threshold": gap["weak_precision_threshold"],
    }


def project_coco_to_kitti(coco_path: str, labels_dir: str, class_name: str) -> dict:
    document = json.loads(Path(coco_path).expanduser().resolve().read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in document.get("categories", [])}
    if set(categories.values()) != {class_name}:
        raise ValueError(f"binary KPI COCO must contain only {class_name!r}; got {categories}")
    images = {int(row["id"]): row for row in document.get("images", [])}
    stems = [Path(str(row["file_name"])).stem for row in images.values()]
    if len(stems) != len(set(stems)):
        raise ValueError("KPI image stems must be unique for KITTI projection")
    rows: dict[int, list[str]] = {image_id: [] for image_id in images}
    for annotation in document.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"annotation references unknown image_id {image_id}")
        if categories.get(int(annotation["category_id"])) != class_name:
            raise ValueError("annotation category is not the binary defect class")
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive COCO bbox for image_id {image_id}")
        x2, y2 = x + width, y + height
        rows[image_id].append(
            f"{class_name} 0.0 0 0.0 {x:.6f} {y:.6f} {x2:.6f} {y2:.6f} 0 0 0 0 0 0 0"
        )
    output = Path(labels_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = set()
    for image_id, image in images.items():
        target = output / f"{Path(str(image['file_name'])).stem}.txt"
        expected.add(target.name)
        temporary = target.with_suffix(".txt.tmp")
        temporary.write_text("\n".join(rows[image_id]) + ("\n" if rows[image_id] else ""), encoding="utf-8")
        os.replace(temporary, target)
    unexpected = sorted(path.name for path in output.glob("*.txt") if path.name not in expected)
    if unexpected:
        raise ValueError(f"KITTI output contains labels outside frozen KPI COCO: {unexpected[:5]}")
    return {"images": len(images), "annotations": sum(len(value) for value in rows.values()), "labels_dir": str(output)}


def _iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0]); y0 = max(left[1], right[1])
    x1 = min(left[2], right[2]); y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _predictions(path: Path, class_name: str) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 8:
            raise ValueError(f"invalid KITTI prediction {path}:{number}")
        if values[0] != class_name:
            continue
        bbox = [float(value) for value in values[4:8]]
        score = float(values[-1]) if len(values) >= 16 else 1.0
        rows.append({"bbox": bbox, "score": score, "order": number})
    return rows


def _match_image(
    ground_truth: list[list[float]],
    predictions: list[dict[str, Any]],
    confidence: float,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    ranked = sorted(
        (row for row in predictions if float(row["score"]) >= confidence),
        key=lambda row: (-float(row["score"]), int(row["order"])),
    )
    unmatched = set(range(len(ground_truth)))
    gaps = []
    tp_flags = []
    for prediction in ranked:
        overlaps = [_iou(prediction["bbox"], box) for box in ground_truth]
        best_iou = max(overlaps, default=0.0)
        eligible = [(overlaps[index], index) for index in unmatched if overlaps[index] >= iou_threshold]
        if eligible:
            _, matched = max(eligible, key=lambda item: (item[0], -item[1]))
            unmatched.remove(matched)
            tp_flags.append(1)
        else:
            tp_flags.append(0)
            gaps.append({"gap_type": "FP", "bbox": prediction["bbox"], "confidence": float(prediction["score"]), "best_iou": best_iou})
    for index in sorted(unmatched):
        overlaps = [_iou(ground_truth[index], row["bbox"]) for row in ranked]
        gaps.append({"gap_type": "FN", "bbox": ground_truth[index], "confidence": 0.0, "best_iou": max(overlaps, default=0.0)})
    tp = sum(tp_flags); fp = len(tp_flags) - tp; fn = len(unmatched)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not ground_truth else 0.0)
    recall = tp / len(ground_truth) if ground_truth else 1.0
    running_tp = running_fp = 0
    precision_at_tp = []
    for flag in tp_flags:
        running_tp += flag; running_fp += 1 - flag
        if flag:
            precision_at_tp.append(running_tp / (running_tp + running_fp))
    ap50 = sum(precision_at_tp) / len(ground_truth) if ground_truth else (1.0 if not ranked else 0.0)
    return gaps, {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "ap50": ap50}


def analyze_binary(args: argparse.Namespace, policy: dict) -> dict[str, Any]:
    if not args.ground_truth_coco:
        raise ValueError("--analyze-binary requires --ground-truth-coco")
    coco = json.loads(Path(args.ground_truth_coco).read_text(encoding="utf-8"))
    class_name = policy["task"]["class_name"]
    categories = {int(row["id"]): str(row["name"]) for row in coco["categories"]}
    images = {int(row["id"]): row for row in coco["images"]}
    gt: dict[int, list[list[float]]] = {image_id: [] for image_id in images}
    for row in coco["annotations"]:
        if categories[int(row["category_id"])] != class_name:
            raise ValueError("binary gap analysis found a non-defect annotation")
        x, y, width, height = [float(value) for value in row["bbox"]]
        gt[int(row["image_id"])].append([x, y, x + width, y + height])
    output = Path(args.output_dir).expanduser().resolve()
    summary: dict[str, Any] = {}
    for kind in ("loose", "strict"):
        confidence = float(policy["gap"][f"{kind}_confidence"])
        gap_rows = []; metric_rows = []; weak_rows = []
        for image_id, image in images.items():
            stem = Path(str(image["file_name"])).stem
            predictions = _predictions(Path(args.inference_ann_path) / f"{stem}.txt", class_name)
            gaps, metrics = _match_image(gt[image_id], predictions, confidence, float(policy["gap"]["match_iou"]))
            filepath = str(image.get("source_path") or image["file_name"])
            for gap_row in gaps:
                gap_rows.append({"kpi": f"{args.kpi}_{kind}", "image_id": image_id, "filepath": filepath, "class": class_name, **gap_row})
            metric = {"image_id": image_id, "filepath": filepath, "class": class_name, **metrics}
            metric_rows.append(metric)
            if metrics["recall"] < float(policy["gap"]["weak_recall_threshold"]) or metrics["ap50"] < float(policy["gap"]["weak_ap50_threshold"]):
                weak_rows.append({"image_id": image_id, "filepath": filepath, "weak_classes": [class_name], "weak_recall": metrics["recall"], "weak_precision": metrics["precision"], "weak_ap50": metrics["ap50"]})
        directory = output / kind
        gap_columns = ["kpi", "image_id", "filepath", "class", "gap_type", "bbox", "confidence", "best_iou"]
        pd.DataFrame(gap_rows, columns=gap_columns).to_parquet(directory / "box_gaps.parquet", index=False)
        pd.DataFrame(metric_rows).to_parquet(directory / "image_metrics.parquet", index=False)
        pd.DataFrame(weak_rows, columns=["image_id", "filepath", "weak_classes", "weak_recall", "weak_precision", "weak_ap50"]).to_parquet(directory / "weak_images.parquet", index=False)
        counts = pd.Series([row["gap_type"] for row in gap_rows]).value_counts().to_dict() if gap_rows else {}
        report = {"status": "COMPLETE", "implementation": "deft_od_aoi_binary", "kpi": f"{args.kpi}_{kind}", "images": len(images), "confidence_threshold": confidence, "iou_threshold": float(policy["gap"]["match_iou"]), "gap_counts": counts, "weak_images": len(weak_rows)}
        (directory / "gap_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary[kind] = report
    return summary


def main() -> int:
    try:
        args = parse_args()
        policy = load_policy(args.policy)
        if args.ground_truth_coco:
            report = project_coco_to_kitti(
                args.ground_truth_coco,
                args.ground_truth_ann_path,
                policy["task"]["class_name"],
            )
            print(f"KPI COCO -> KITTI: {report}")
        output = Path(args.output_dir).expanduser().resolve()
        for kind in ("loose", "strict"):
            directory = output / kind
            directory.mkdir(parents=True, exist_ok=True)
            spec_path = directory / "od_gap_spec.yaml"
            spec_path.write_text(
                yaml.safe_dump(build_spec(args, policy, kind), sort_keys=False),
                encoding="utf-8",
            )
            print(f"{kind} gap spec -> {spec_path}")
        if args.analyze_binary:
            print(json.dumps(analyze_binary(args, policy), indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
