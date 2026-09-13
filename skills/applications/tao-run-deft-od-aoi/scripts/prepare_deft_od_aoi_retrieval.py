#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the proven role-aware SigLIP candidate or gap-query inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from prepare_deft_od_aoi_siglip_candidates import prepare as prepare_candidates
from prepare_deft_od_aoi_siglip_queries import prepare as prepare_queries


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _embedding_spec(policy: dict[str, Any], input_path: Path, output: Path) -> dict[str, Any]:
    retrieval = policy["retrieval"]
    return {
        "input_parquet": str(input_path),
        "output_parquet": str(output),
        "model": retrieval["model"],
        "model_path": retrieval["model_path"],
        "model_config_path": "",
        "batch_size": 64,
    }


def _publish_paths(frame: Any, output: Path, published: Path) -> None:
    scratch_prefix = str(output.resolve())
    durable_prefix = str(published.resolve())
    frame["filepath"] = frame["filepath"].map(
        lambda value: durable_prefix + str(value)[len(scratch_prefix):]
        if str(value).startswith(scratch_prefix + "/") else str(value)
    )


def candidates(
    policy_path: Path,
    output: Path,
    published_output: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    published = (published_output or output).expanduser().resolve()
    source, clean = policy["sources"]["real"], policy["sources"]["clean"]
    output.mkdir(parents=True)
    inputs = output / "candidate_inputs.parquet"
    report_path = output / "candidate_report.json"
    args = argparse.Namespace(
        source_coco=source["coco"], source_images_dir=source["images"],
        clean_coco=clean["coco"], clean_images_dir=clean["images"],
        output_crops_dir=str(output / "crops"), output_parquet=str(inputs),
        report_json=str(report_path),
        defect_context_scale=float(policy["retrieval"]["defect_context_scale"]),
        clean_grids=",".join(map(str, policy["retrieval"]["clean_grids"])),
        output_size=int(policy["retrieval"]["output_size"]),
    )
    frame, report = prepare_candidates(args)
    _publish_paths(frame, output, published)
    frame.to_parquet(inputs, index=False)
    _json(report_path, report)
    embedded = output / "candidate_embeddings.parquet"
    (output / "embed_candidates.yaml").write_text(
        yaml.safe_dump(_embedding_spec(
            policy, published / inputs.name, published / embedded.name
        ), sort_keys=False)
    )
    manifest = {
        "status": "COMPLETE",
        "counts": report["rows"],
        "parents": report["parents"],
        "input_parquet": str(published / inputs.name),
        "embedding_output": str(published / embedded.name),
        "encoder": policy["retrieval"],
    }
    _json(output / "candidate_manifest.json", manifest)
    return manifest


def queries(policy_path: Path, strict_path: Path, loose_path: Path, iteration: int,
            output: Path, candidate_root: Path,
            published_output: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    published = (published_output or output).expanduser().resolve()
    kpi = policy["sources"]["kpi"]
    output.mkdir(parents=True)
    inputs = output / "query_inputs.parquet"
    report_path = output / "query_report.json"
    args = argparse.Namespace(
        strict_gaps=str(strict_path), loose_gaps=str(loose_path),
        kpi_coco=kpi["coco"], kpi_images_dir=kpi["images"],
        output_crops_dir=str(output / "crops"), output_parquet=str(inputs),
        report_json=str(report_path),
        context_scale=float(policy["retrieval"]["defect_context_scale"]),
        output_size=int(policy["retrieval"]["output_size"]),
        background_iou_upper=float(policy["gap"]["background_iou_upper"]),
        near_miss_iou_upper=float(policy["gap"]["near_miss_iou_upper"]),
    )
    frame, report = prepare_queries(args)
    _publish_paths(frame, output, published)
    frame.to_parquet(inputs, index=False)
    _json(report_path, report)
    embedded = output / "query_embeddings.parquet"
    (output / "embed_queries.yaml").write_text(
        yaml.safe_dump(_embedding_spec(
            policy, published / inputs.name, published / embedded.name
        ), sort_keys=False)
    )
    manifest = {
        "status": "COMPLETE",
        "iteration": iteration,
        "query_counts": report["queries"],
        "role_counts": report["roles"],
        "candidate_embeddings": str((candidate_root / "candidate_embeddings.parquet").resolve()),
        "input_parquet": str(published / inputs.name),
        "embedding_output": str(published / embedded.name),
        "controller": "role_and_pocket_aware_siglip",
    }
    _json(output / "query_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidates")
    candidate.add_argument("--policy", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    candidate.add_argument("--published-output-dir", type=Path)
    query = sub.add_parser("queries")
    query.add_argument("--policy", type=Path, required=True)
    query.add_argument("--strict-gaps", type=Path, required=True)
    query.add_argument("--loose-gaps", type=Path, required=True)
    query.add_argument("--iteration", type=int, required=True)
    query.add_argument("--output-dir", type=Path, required=True)
    query.add_argument("--candidate-root", type=Path, required=True)
    query.add_argument("--published-output-dir", type=Path)
    args = parser.parse_args()
    result = (
        candidates(
            args.policy.resolve(), args.output_dir.resolve(),
            args.published_output_dir.resolve() if args.published_output_dir else None,
        )
        if args.command == "candidates"
        else queries(args.policy.resolve(), args.strict_gaps.resolve(), args.loose_gaps.resolve(),
                     args.iteration, args.output_dir.resolve(), args.candidate_root.resolve(),
                     args.published_output_dir.resolve()
                     if args.published_output_dir else None)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
