#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One public driver for DEFT OD AOI SigLIP query preparation and routing.

The GPU embedding remains a separate tracked leaf job. ``prepare`` writes its
nested YAML spec; ``commit`` consumes that result, performs routing, and gates
the complete route artifact set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from deft_od_aoi_policy import load_policy
from prepare_deft_od_aoi_siglip_queries import prepare as prepare_queries
from route_deft_od_aoi_siglip import route


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    retrieval = policy["retrieval"]
    root = Path(args.output_root).expanduser().resolve()
    queries = root / "queries"
    specs = root / "specs"
    queries.mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    input_parquet = queries / "query_inputs.parquet"
    report_json = queries / "query_report.json"
    frame, query_report = prepare_queries(
        argparse.Namespace(
            strict_gaps=args.strict_gaps,
            loose_gaps=args.loose_gaps,
            kpi_coco=args.kpi_coco,
            kpi_images_dir=args.kpi_images_dir,
            output_crops_dir=str(queries / "crops"),
            output_parquet=str(input_parquet),
            report_json=str(report_json),
            context_scale=float(retrieval["defect_context_scale"]),
            output_size=int(retrieval["output_size"]),
            background_iou_upper=float(policy["gap"]["background_iou_upper"]),
            near_miss_iou_upper=float(policy["gap"]["near_miss_iou_upper"]),
        )
    )
    temporary = input_parquet.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, input_parquet)
    _atomic_json(report_json, query_report)
    embedding_output = queries / "query_embeddings.parquet"
    embedding_spec = {
        "input_parquet": str(input_parquet),
        "output_parquet": str(embedding_output),
        "model": retrieval["model"],
        "model_path": args.runtime_model_path or retrieval["model_path"],
        "model_config_path": "",
        "batch_size": int(args.embedding_batch_size),
    }
    spec_path = specs / "query_embeddings.yaml"
    spec_path.write_text(yaml.safe_dump(embedding_spec, sort_keys=False), encoding="utf-8")
    state = {
        "status": "PREPARED",
        "iteration": int(args.iteration),
        "query_inputs": str(input_parquet),
        "query_embeddings": str(embedding_output),
        "embedding_spec": str(spec_path),
        "query_report": str(report_json),
        "encoder": {"model": retrieval["model"], "model_path": retrieval["model_path"]},
    }
    _atomic_json(root / "routing_state.json", state)
    return state


def _stage_coco(args: argparse.Namespace) -> dict[str, Any]:
    document = json.loads(Path(args.coco).expanduser().resolve().read_text(encoding="utf-8"))
    output = Path(args.output_images).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    for image_row in document.get("images", []):
        name = Path(str(image_row.get("file_name") or "")).name
        if not name or name in seen:
            raise ValueError(f"COCO file_name basenames must be nonempty and unique: {name!r}")
        seen.add(name)
        source = Path(str(image_row.get("source_path") or image_row["file_name"])).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, output / name)
    report = {"status": "COMPLETE", "images": len(seen), "output_images": str(output)}
    _atomic_json(Path(args.report), report)
    return report


def _finalize_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    """Remove ephemeral crop paths before copying query embeddings durably."""
    frame = pd.read_parquet(Path(args.embedding_parquet).expanduser().resolve()).copy()
    required = {"query_id", "role", "branch", "embedding"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"query embeddings are empty or lack {sorted(required - set(frame.columns))}")
    if frame["query_id"].astype(str).duplicated().any():
        raise ValueError("query embedding query_id values must be unique")
    ephemeral = []
    for column in ("filepath", "parent_filepath"):
        if column in frame and frame[column].astype(str).str.startswith("/raid/scratch/").any():
            ephemeral.append(column)
    frame = frame.drop(columns=[column for column in ("filepath", "parent_filepath") if column in frame])
    output = Path(args.durable_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    published = (
        str(Path(args.published_output).expanduser().resolve())
        if args.published_output
        else str(output)
    )
    report = {
        "status": "COMPLETE",
        "rows": len(frame),
        "removed_path_columns": ephemeral,
        "local_output": str(output),
        "durable_output": published,
    }
    _atomic_json(Path(args.report), report)
    if args.routing_state:
        state_path = Path(args.routing_state).expanduser().resolve()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["query_embeddings"] = published
        state["embedding_status"] = "COMPLETE"
        _atomic_json(state_path, state)
    return report


def _validate_commit(root: Path, report: dict[str, Any]) -> None:
    required = (
        "mined_manifest.json",
        "synthetic_plan.json",
        "routing_report.json",
        "defect_ledger.json",
        "clean_ledger.json",
        "admission_index.npy",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"route output gate missing artifacts: {missing}")
    selected = report.get("selected") or {}
    if any(int(value) < 0 for value in selected.values()):
        raise ValueError("route report contains a negative selected count")
    manifest = json.loads((root / "mined_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != sum(int(value) for value in selected.values()):
        raise ValueError("mined manifest row count does not match routing report")


def _commit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).expanduser().resolve()
    state_path = root / "routing_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state["iteration"]) != int(args.iteration):
        raise ValueError("routing state iteration does not match --iteration")
    query_embeddings = Path(args.query_embeddings or state["query_embeddings"]).resolve()
    if not query_embeddings.is_file():
        raise FileNotFoundError(query_embeddings)
    frame = pd.read_parquet(query_embeddings)
    if frame.empty or not {"query_id", "role", "branch", "embedding"}.issubset(frame.columns):
        raise ValueError("query embedding output is empty or lacks routing columns")
    output = root / "routing"
    report = route(
        argparse.Namespace(
            policy=args.policy,
            iteration=args.iteration,
            candidate_embeddings=args.candidate_embeddings,
            query_embeddings=str(query_embeddings),
            kpi_coco=args.kpi_coco,
            source_coco=args.source_coco,
            source_images_dir=args.source_images_dir,
            clean_coco=args.clean_coco,
            clean_images_dir=args.clean_images_dir,
            output_dir=str(output),
            previous_defect_ledger=args.previous_defect_ledger,
            previous_clean_ledger=args.previous_clean_ledger,
            previous_admission_index=args.previous_admission_index,
            conversion_old_strict=args.conversion_old_strict,
            conversion_new_strict=args.conversion_new_strict,
            prior_admitted_synthetic=args.prior_admitted_synthetic,
            valid_generator_types=args.valid_generator_types,
        )
    )
    _validate_commit(output, report)
    state.update({"status": "COMPLETE", "query_embeddings": str(query_embeddings), "routing_dir": str(output)})
    _atomic_json(state_path, state)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Crop gap queries and write the embedding spec.")
    for command in (prepare,):
        command.add_argument("--policy", required=True)
        command.add_argument("--iteration", type=int, required=True)
        command.add_argument("--output-root", required=True)
    prepare.add_argument("--strict-gaps", required=True)
    prepare.add_argument("--loose-gaps", required=True)
    prepare.add_argument("--kpi-coco", required=True)
    prepare.add_argument("--kpi-images-dir", required=True)
    prepare.add_argument("--embedding-batch-size", type=int, default=64)
    prepare.add_argument(
        "--runtime-model-path",
        help="Node-local copy of the frozen encoder; policy identity remains unchanged in routing state.",
    )
    prepare.set_defaults(func=_prepare)

    commit = commands.add_parser("commit", help="Consume embeddings, route, and validate outputs.")
    commit.add_argument("--policy", required=True)
    commit.add_argument("--iteration", type=int, required=True)
    commit.add_argument("--output-root", required=True)
    commit.add_argument("--candidate-embeddings", required=True)
    commit.add_argument("--query-embeddings")
    commit.add_argument("--kpi-coco", required=True)
    commit.add_argument("--source-coco", required=True)
    commit.add_argument("--source-images-dir", required=True)
    commit.add_argument("--clean-coco", required=True)
    commit.add_argument("--clean-images-dir", required=True)
    commit.add_argument("--previous-defect-ledger")
    commit.add_argument("--previous-clean-ledger")
    commit.add_argument("--previous-admission-index")
    commit.add_argument("--conversion-old-strict")
    commit.add_argument("--conversion-new-strict")
    commit.add_argument("--prior-admitted-synthetic", type=int, default=0)
    commit.add_argument("--valid-generator-types")
    commit.set_defaults(func=_commit)

    stage = commands.add_parser("stage-coco", help="Materialize a frozen COCO image view on node-local storage.")
    stage.add_argument("--coco", required=True)
    stage.add_argument("--output-images", required=True)
    stage.add_argument("--report", required=True)
    stage.set_defaults(func=_stage_coco)

    finalize = commands.add_parser("finalize-embeddings", help="Strip ephemeral crop paths and gate durable query embeddings.")
    finalize.add_argument("--embedding-parquet", required=True)
    finalize.add_argument("--durable-output", required=True)
    finalize.add_argument("--published-output")
    finalize.add_argument("--report", required=True)
    finalize.add_argument("--routing-state")
    finalize.set_defaults(func=_finalize_embeddings)
    return root


def main() -> int:
    try:
        args = parser().parse_args()
        print(json.dumps(args.func(args), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
