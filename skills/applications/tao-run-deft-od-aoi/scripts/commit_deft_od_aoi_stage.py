#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomically commit one verified stage to the DEFT OD AOI state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NEXT = {"synthesis_bootstrap": "candidate_cache", "candidate_cache": "baseline_measurement",
        "baseline_measurement": "baseline_gaps",
        "baseline_gaps": "iteration_retrieval", "iteration_retrieval": "iteration_admission",
        "iteration_admission": "iteration_training", "iteration_synthesis": "iteration_training",
        "iteration_training": "iteration_measurement",
        "iteration_measurement": "iteration_gaps"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_artifact(artifacts: dict[str, dict[str, Any]], name: str) -> Any:
    if name not in artifacts:
        raise ValueError(f"stage commit requires artifact {name}")
    return json.loads(Path(artifacts[name]["path"]).read_text())


def _validate_iteration_retrieval(
    state: dict[str, Any], iteration: int, artifacts: dict[str, dict[str, Any]]
) -> None:
    required_outputs = {
        "routing_report", "mined_manifest", "defect_ledger", "clean_ledger",
        "admission_index", "synthetic_plan",
    }
    preview = _read_json_artifact(artifacts, "admission_preview")
    query_manifest = _read_json_artifact(artifacts, "query_manifest")
    if preview.get("status") != "COMPLETE" or int(preview.get("iteration", -1)) != iteration:
        raise ValueError("admission preview is incomplete or for another iteration")
    controller = preview.get("controller") or {}
    required_controller = {
        "near_miss_real_cap_per_pocket", "near_miss_real_factor",
        "clean_factor", "clean_cumulative_cap_per_real",
    }
    missing_controller = required_controller - set(controller)
    if controller.get("strict") != "adaptive_per_pocket" or missing_controller:
        raise ValueError("admission preview lacks the role/pocket-aware controller contract")
    outputs = preview.get("outputs") or {}
    if set(outputs) != required_outputs:
        raise ValueError("admission preview output set is incomplete")
    for name in sorted(required_outputs):
        if name not in artifacts:
            raise ValueError(f"iteration_retrieval requires artifact {name}")
        if (outputs[name].get("path") != artifacts[name]["path"]
                or outputs[name].get("sha256") != artifacts[name]["sha256"]
                or int(outputs[name].get("bytes", -1)) != artifacts[name]["bytes"]):
            raise ValueError(f"admission preview hash/path mismatch for {name}")
    preview_inputs = preview.get("inputs") or {}
    frozen_routing_sha = state.get("routing_policy_sha256")
    if not frozen_routing_sha or (preview_inputs.get("routing_policy") or {}).get("sha256") != frozen_routing_sha:
        raise ValueError("admission preview does not use the frozen routing policy")
    query_input = preview_inputs.get("query_manifest") or {}
    if (query_input.get("path") != artifacts["query_manifest"]["path"]
            or query_input.get("sha256") != artifacts["query_manifest"]["sha256"]):
        raise ValueError("admission preview does not bind the committed query manifest")
    if (query_manifest.get("status") != "COMPLETE"
            or int(query_manifest.get("iteration", -1)) != iteration):
        raise ValueError("query manifest is incomplete or for another iteration")
    report = _read_json_artifact(artifacts, "routing_report")
    manifest = _read_json_artifact(artifacts, "mined_manifest")
    defect_ledger = _read_json_artifact(artifacts, "defect_ledger")
    clean_ledger = _read_json_artifact(artifacts, "clean_ledger")
    counts = preview.get("counts") or {}
    if report.get("queries") != query_manifest.get("query_counts") or counts.get("queries") != report.get("queries"):
        raise ValueError("routing/query counts disagree")
    if counts.get("selected") != report.get("selected"):
        raise ValueError("routing selected counts disagree")
    if int(counts.get("manifest_records", -1)) != len(manifest):
        raise ValueError("mined manifest count disagrees with admission preview")
    if int(counts.get("defect_ledger", -1)) != len(defect_ledger):
        raise ValueError("defect ledger count disagrees with admission preview")
    if int(counts.get("clean_ledger", -1)) != len(clean_ledger):
        raise ValueError("clean ledger count disagrees with admission preview")


def _committed_synthetic_plan(
    state: dict[str, Any], iteration: int
) -> tuple[dict[str, int], dict[str, Any]]:
    for event in reversed(state.get("events", [])):
        if event.get("stage") != "iteration_retrieval" or int(event.get("iteration", -1)) != iteration:
            continue
        artifact = (event.get("artifacts") or {}).get("synthetic_plan")
        if not artifact:
            break
        path = Path(artifact["path"])
        if _sha(path) != artifact["sha256"]:
            raise ValueError("committed synthetic plan hash is stale")
        plan = json.loads(path.read_text())
        if not isinstance(plan, dict):
            raise ValueError("committed synthetic plan is not a JSON object")
        return {str(key): int(value) for key, value in plan.items()}, artifact
    raise ValueError("iteration_synthesis has no committed retrieval synthetic plan")


def _validate_iteration_synthesis(
    state: dict[str, Any], iteration: int, artifacts: dict[str, dict[str, Any]]
) -> None:
    reconciliation = _read_json_artifact(artifacts, "synthesis_plan_reconciliation")
    if reconciliation.get("status") != "COMPLETE":
        raise ValueError("synthesis plan reconciliation is incomplete")
    plan, committed = _committed_synthetic_plan(state, iteration)
    reference = reconciliation.get("synthetic_plan") or {}
    if (reference.get("path") != committed["path"]
            or reference.get("sha256") != committed["sha256"]):
        raise ValueError("synthesis reconciliation does not bind the committed synthetic plan")
    per_type = reconciliation.get("per_type") or {}
    if set(per_type) != set(plan):
        raise ValueError("synthesis reconciliation type set disagrees with synthetic plan")
    actual_total = shortfall_total = 0
    for anomaly_type, requested in plan.items():
        row = per_type[anomaly_type]
        actual = int(row.get("generator_rows", -1))
        shortfall = int(row.get("explicit_shortfall", -1))
        if int(row.get("requested_images", -1)) != requested:
            raise ValueError(f"synthesis requested count mismatch for {anomaly_type}")
        if actual < 0 or shortfall < 0 or actual + shortfall != requested:
            raise ValueError(f"synthesis count/shortfall mismatch for {anomaly_type}")
        actual_total += actual
        shortfall_total += shortfall
    if (int(reconciliation.get("requested_total", -1)) != sum(plan.values())
            or int(reconciliation.get("generator_row_count", -1)) != actual_total
            or int(reconciliation.get("explicit_shortfall_total", -1)) != shortfall_total):
        raise ValueError("synthesis reconciliation totals disagree with per-type counts")


def commit(state_path: Path, stage: str, iteration: int, values: list[str]) -> dict[str, Any]:
    state = json.loads(state_path.read_text())
    if state.get("status") not in {"READY", "RUNNING"} or state.get("next_stage") != stage:
        raise ValueError(f"expected stage {state.get('next_stage')}, not {stage}")
    expected = int(state["current_iteration"])
    if stage == "iteration_retrieval" and state.get("last_stage") in {"baseline_gaps", "iteration_gaps"}:
        expected += 1
    if iteration != expected:
        raise ValueError(f"expected iteration {expected}, not {iteration}")
    artifacts = {}
    for value in values:
        name, separator, raw = value.partition("=")
        path = Path(raw).expanduser().resolve()
        if not separator or not name or not path.is_file():
            raise ValueError(f"artifact must be name=existing-file: {value}")
        artifacts[name] = {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}
    if not artifacts:
        raise ValueError("at least one completion artifact is required")
    if stage == "iteration_retrieval":
        _validate_iteration_retrieval(state, iteration, artifacts)
    if stage == "iteration_synthesis":
        _validate_iteration_synthesis(state, iteration, artifacts)
    next_stage, status = NEXT.get(stage), "RUNNING"
    if stage == "iteration_admission" and state.get("synthesis_enabled"):
        next_stage = "iteration_synthesis"
    if stage == "iteration_gaps":
        if iteration >= int(state["max_iterations"]):
            next_stage, status = None, "COMPLETE"
        else:
            next_stage = "iteration_retrieval"
    state.update(status=status, current_iteration=iteration, last_stage=stage,
                 next_stage=next_stage)
    event = {"stage": stage, "iteration": iteration, "artifacts": artifacts,
             "committed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    state.setdefault("events", []).append(event)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)
    with (state_path.parent / "loop_log.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(NEXT) + ("iteration_gaps",), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()
    result = commit(args.state.resolve(), args.stage, args.iteration, args.artifact)
    print(json.dumps({"status": result["status"], "next_stage": result["next_stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
