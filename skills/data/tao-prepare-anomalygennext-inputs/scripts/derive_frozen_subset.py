#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive a small immutable AnomalyGenNext inference input root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


KNOWN_BRANCHES = {"fn_mask", "same_type_sampled_mask"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _validate_parent(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = (
        root / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or not manifest.get("generation_ready"):
        raise ValueError("parent prepared-input manifest is not COMPLETE/generation_ready")
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"parent frozen artifact changed or is missing: {path}")
    return manifest, manifest_path


def derive(args: argparse.Namespace) -> None:
    source = Path(args.inputs_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived input root: {output}")
    mask_branches = {
        value.strip() for value in args.mask_branches.split(",") if value.strip()
    }
    if not mask_branches:
        raise ValueError("--mask-branches must contain at least one branch")
    unknown_branches = sorted(mask_branches - KNOWN_BRANCHES)
    if unknown_branches:
        raise ValueError(
            f"unknown mask branches {unknown_branches}; available={sorted(KNOWN_BRANCHES)}"
        )

    parent, parent_path = _validate_parent(source)
    plan_rows = json.loads(
        (
            source
            / "prepared_anomalygennext_inputs"
            / "anomalygen_next_generation_plan.json"
        ).read_text()
    )
    matches = [row for row in plan_rows if str(row["dataset_id"]) == args.dataset]
    if len(matches) != 1:
        raise ValueError(f"expected one plan row for dataset={args.dataset!r}, found {len(matches)}")
    parent_plan = matches[0]
    testcase_rows = _read_jsonl(Path(parent_plan["testcase"]))
    provenance_rows = _read_jsonl(Path(parent_plan["provenance"]))
    if len(testcase_rows) != len(provenance_rows):
        raise ValueError("parent testcase/provenance cardinality mismatch")

    if args.all_pairs:
        pair_ids = list(
            dict.fromkeys(str(row["pair_id"]) for row in provenance_rows)
        )
    else:
        pair_ids = [value.strip() for value in args.pair_ids.split(",") if value.strip()]
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("--pair-ids must contain unique, non-empty pair ids")

    wanted = set(pair_ids)
    selected = [
        (testcase, provenance)
        for testcase, provenance in zip(testcase_rows, provenance_rows, strict=True)
        if str(provenance.get("pair_id")) in wanted
        and str(provenance.get("mask_branch")) in mask_branches
    ]
    observed = {str(provenance["pair_id"]) for _, provenance in selected}
    if observed != wanted:
        raise ValueError(f"unknown pair ids: {sorted(wanted - observed)}")
    for pair_id in pair_ids:
        branches = {
            str(provenance["mask_branch"])
            for _, provenance in selected
            if str(provenance["pair_id"]) == pair_id
        }
        if branches != mask_branches:
            raise ValueError(
                f"pair {pair_id} must contain exactly {sorted(mask_branches)}, got {sorted(branches)}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        prepared_inputs = temporary / "prepared_anomalygennext_inputs"
        directory = prepared_inputs / "anomalygen_inputs" / args.dataset
        testcase = directory / "testcase.jsonl"
        provenance = directory / "provenance.jsonl"
        selected_testcase = [row[0] for row in selected]
        selected_provenance = [row[1] for row in selected]
        _write_jsonl(testcase, selected_testcase)
        _write_jsonl(provenance, selected_provenance)

        anomaly_types = sorted({str(row["anomaly_type"]) for row in selected_provenance})
        plan = {
            **parent_plan,
            "anomaly_type": anomaly_types[0] if len(anomaly_types) == 1 else ",".join(anomaly_types),
            "anomaly_types": anomaly_types,
            "testcase": str(
                output
                / "prepared_anomalygennext_inputs"
                / "anomalygen_inputs"
                / args.dataset
                / "testcase.jsonl"
            ),
            "provenance": str(
                output
                / "prepared_anomalygennext_inputs"
                / "anomalygen_inputs"
                / args.dataset
                / "provenance.jsonl"
            ),
            "requested_rows": len(selected),
        }
        plan_path = prepared_inputs / "anomalygen_next_generation_plan.json"
        _write_json(plan_path, [plan])

        unified = prepared_inputs / "anomalygen_inputs.jsonl"
        _write_jsonl(
            unified,
            [
                {
                    **provenance_row,
                    "checkpoint": str(parent_plan["checkpoint"]),
                    "recipe": str(parent_plan["recipe"]),
                    "generator_input": testcase_row,
                }
                for testcase_row, provenance_row in selected
            ],
        )
        request_path = prepared_inputs / "subset_request.json"
        _write_json(
            request_path,
            {
                "schema_version": 1,
                "dataset_id": args.dataset,
                "pair_ids": pair_ids,
                "required_mask_branches": sorted(mask_branches),
                "parent_manifest": str(parent_path),
                "parent_manifest_sha256": _sha256(parent_path),
            },
        )

        artifacts = []
        for temporary_path, final_path in (
            (testcase, Path(plan["testcase"])),
            (provenance, Path(plan["provenance"])),
            (
                unified,
                output / "prepared_anomalygennext_inputs" / "anomalygen_inputs.jsonl",
            ),
            (
                plan_path,
                output
                / "prepared_anomalygennext_inputs"
                / "anomalygen_next_generation_plan.json",
            ),
            (
                request_path,
                output / "prepared_anomalygennext_inputs" / "subset_request.json",
            ),
        ):
            artifacts.append(
                {
                    "path": str(final_path),
                    "sha256": _sha256(temporary_path),
                    "bytes": temporary_path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": 2,
            "phase": "prepared_anomalygennext_inputs_subset",
            "status": "COMPLETE",
            "source_tag": parent["source_tag"],
            "selected_fn_count": len({str(row["fn_id"]) for row in selected_provenance}),
            "selected_pair_count": len(pair_ids),
            "generator_row_count": len(selected),
            "generator_groups": [plan],
            "artifacts": artifacts,
            "parent_prepared_inputs_manifest": str(parent_path),
            "parent_prepared_inputs_manifest_sha256": _sha256(parent_path),
            "generation_ready": True,
            "training_pool_mutated": False,
        }
        _write_json(prepared_inputs / "prepared_inputs_manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"derived frozen subset PASS: dataset={args.dataset} pairs={len(pair_ids)} "
        f"rows={len(selected)} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--inputs-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--dataset", required=True)
    pair_selection = result.add_mutually_exclusive_group(required=True)
    pair_selection.add_argument("--pair-ids", help="Comma-separated frozen pair ids")
    pair_selection.add_argument(
        "--all-pairs",
        action="store_true",
        help="Retain every frozen pair for the selected dataset",
    )
    result.add_argument(
        "--mask-branches",
        default=",".join(sorted(KNOWN_BRANCHES)),
        help="Comma-separated branches to retain (default: both frozen branches)",
    )
    result.set_defaults(func=derive)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
