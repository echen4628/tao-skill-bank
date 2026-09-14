# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "commit_deft_od_aoi_stage.py"
SPEC = importlib.util.spec_from_file_location("commit_deft_od_aoi_stage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(root: Path) -> tuple[Path, Path]:
    state = root / "deft_state.json"
    state.write_text(json.dumps({
        "status": "READY", "next_stage": "candidate_cache",
        "current_iteration": 0, "max_iterations": 1,
        "routing_policy_sha256": "routing-sha",
    }))
    artifact = root / "done.json"
    artifact.write_text("{}")
    return state, artifact


def _retrieval_artifacts(
    root: Path, stale: bool = False, sparse_queries: bool = False
) -> list[str]:
    query = root / "query_manifest.json"
    report = root / "routing_report.json"
    mined = root / "mined_manifest.json"
    defect = root / "defect_ledger.json"
    clean = root / "clean_ledger.json"
    synthetic = root / "synthetic_plan.json"
    index = root / "admission_index.npy"
    query_counts = (
        {"strict_fn": 1}
        if sparse_queries
        else {"strict_fn": 1, "near_miss_fp": 1, "background_fp": 1}
    )
    query.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "query_counts": query_counts,
    }))
    report_queries = (
        {"strict_fn": 1, "near_miss_fp": 0, "background_fp": 0}
        if sparse_queries
        else {"strict_fn": 1, "near_miss_fp": 1, "background_fp": 1}
    )
    selected = (
        {"strict_fn_real": 1, "near_miss_real": 0,
         "uniform_real": 0, "background_clean": 0}
        if sparse_queries
        else {"strict_fn_real": 1, "near_miss_real": 1,
              "uniform_real": 0, "background_clean": 1}
    )
    report.write_text(json.dumps({
        "queries": report_queries,
        "selected": selected,
    }))
    mined_rows = [{"branch": "strict_fn_real"}]
    defect_rows = ["real-a"]
    clean_rows = []
    if not sparse_queries:
        mined_rows += [
            {"branch": "near_miss_real"}, {"branch": "background_clean"},
        ]
        defect_rows.append("real-b")
        clean_rows.append("clean-a")
    mined.write_text(json.dumps(mined_rows))
    defect.write_text(json.dumps(defect_rows))
    clean.write_text(json.dumps(clean_rows))
    synthetic.write_text("{}")
    index.write_bytes(b"index")
    outputs = {}
    for name, path in {
        "routing_report": report, "mined_manifest": mined,
        "defect_ledger": defect, "clean_ledger": clean,
        "synthetic_plan": synthetic, "admission_index": index,
    }.items():
        outputs[name] = {
            "path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size,
        }
    preview = root / "admission_preview.json"
    preview.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "controller": {
            "strict": "adaptive_per_pocket", "near_miss_real_factor": 2,
            "near_miss_real_cap_per_pocket": 20, "clean_factor": 5,
            "clean_cumulative_cap_per_real": 1.0,
        },
        "counts": {
            "queries": report_queries,
            "selected": selected, "manifest_records": len(mined_rows),
            "defect_ledger": len(defect_rows), "clean_ledger": len(clean_rows),
        },
        "inputs": {
            "routing_policy": {"sha256": "routing-sha"},
            "query_manifest": {"path": str(query.resolve()), "sha256": _sha(query)},
        },
        "outputs": outputs,
    }))
    if stale:
        report.write_text('{"queries": {}, "selected": {}}')
    return [
        f"query_manifest={query}", f"admission_preview={preview}",
        f"routing_report={report}", f"mined_manifest={mined}",
        f"defect_ledger={defect}", f"clean_ledger={clean}",
        f"synthetic_plan={synthetic}", f"admission_index={index}",
    ]


def _successful_status(path: Path, exit_code: int = 0) -> str:
    path.write_text(json.dumps({"status": "COMPLETE", "exit_code": exit_code}))
    return str(path)


def _iteration_artifacts(root: Path) -> dict[str, list[str]]:
    checkpoint = root / "model_epoch_001.pth"
    checkpoint.write_bytes(b"checkpoint")
    leaf_status = root / "train_leaf_status.jsonl"
    leaf_status.write_text('{"epoch": 1, "kpi": {"val_mAP50": 0.75}}\n')
    selection = root / "checkpoint_selection.json"
    selection.write_text(json.dumps({
        "status": "COMPLETE", "action": "select", "best_epoch": 1,
        "best_kpi_mAP50": 0.75, "selected_checkpoint": str(checkpoint.resolve()),
        "planned_epochs": 2, "extension_applied": False,
        "status_files": [str(leaf_status.resolve())],
    }))
    main_status = root / "main_status.json"
    _successful_status(main_status)
    manifest = root / "measurement_manifest.json"
    manifest.write_text(json.dumps({
        "status": "COMPLETE", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_provenance": {
            "role": "iteration_selected", "path": str(checkpoint.resolve()),
            "sha256": _sha(checkpoint), "source_iteration": 1,
        },
        "specs": {name: str((root / name).resolve()) for name in (
            "kpi_inference.yaml", "test_inference.yaml",
            "gap_loose.yaml", "gap_strict.yaml",
        )},
    }))
    kpi_status = root / "kpi_status.json"
    test_status = root / "test_status.json"
    _successful_status(kpi_status)
    _successful_status(test_status)
    gap_artifacts = []
    for kind, confidence in (("loose", 0.3), ("strict", 0.8)):
        status = root / f"{kind}_status.json"
        report = root / f"{kind}_gap_report.json"
        boxes = root / f"{kind}_box_gaps.parquet"
        _successful_status(status)
        report.write_text(json.dumps({
            "kpi": f"kpi_{kind}", "counts_by_type": {"FP": 2, "FN": 1},
            "counts_by_class": {"defect": {"FP": 2, "FN": 1}},
            "settings": {"iou_threshold": 0.5, "conf_threshold": confidence,
                         "min_area": 0},
        }))
        boxes.write_bytes(b"PAR1")
        gap_artifacts.extend([
            f"{kind}_status={status}", f"{kind}_gap_report={report}",
            f"{kind}_box_gaps={boxes}",
        ])
    return {
        "iteration_training": [
            f"checkpoint_selection={selection}", f"main_status={main_status}",
        ],
        "iteration_measurement": [
            f"measurement_manifest={manifest}",
            f"kpi_inference_status={kpi_status}",
            f"test_inference_status={test_status}",
        ],
        "iteration_gaps": gap_artifacts,
    }


def test_commit_enforces_order_and_completes_after_final_gaps(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    iteration_artifacts = _iteration_artifacts(tmp_path)
    stages = [("candidate_cache", 0), ("baseline_measurement", 0),
              ("baseline_gaps", 0), ("iteration_retrieval", 1),
              ("iteration_admission", 1), ("iteration_training", 1),
              ("iteration_measurement", 1), ("iteration_gaps", 1)]
    value = None
    for stage, iteration in stages:
        values = (
            _retrieval_artifacts(tmp_path)
            if stage == "iteration_retrieval" else
            iteration_artifacts.get(stage, [f"done={artifact}"])
        )
        value = MODULE.commit(state, stage, iteration, values)
    assert value["status"] == "COMPLETE" and value["next_stage"] is None
    assert len(value["events"]) == len(stages)
    assert len((tmp_path / "loop_log.jsonl").read_text().splitlines()) == len(stages)


def test_training_rejects_false_complete_wrapper_status(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_training",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    artifacts = _iteration_artifacts(tmp_path)["iteration_training"]
    bad_status = tmp_path / "main_status.json"
    _successful_status(bad_status, exit_code=1)

    with pytest.raises(ValueError, match="does not prove terminal success"):
        MODULE.commit(state, "iteration_training", 1, artifacts)


def test_gaps_reject_missing_terminal_status_evidence(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_gaps",
                 current_iteration=1, last_stage="iteration_measurement")
    state.write_text(json.dumps(value))
    artifacts = [
        value for value in _iteration_artifacts(tmp_path)["iteration_gaps"]
        if not value.startswith(("loose_status=", "strict_status="))
    ]

    with pytest.raises(ValueError, match="requires artifact loose_status"):
        MODULE.commit(state, "iteration_gaps", 1, artifacts)


def test_training_rejects_complete_but_malformed_checkpoint_report(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_training",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    artifacts = _iteration_artifacts(tmp_path)["iteration_training"]
    selection = tmp_path / "checkpoint_selection.json"
    report = json.loads(selection.read_text())
    report.pop("selected_checkpoint")
    selection.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="existing checkpoint"):
        MODULE.commit(state, "iteration_training", 1, artifacts)


def test_gaps_reject_complete_status_with_inconsistent_report(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_gaps",
                 current_iteration=1, last_stage="iteration_measurement")
    state.write_text(json.dumps(value))
    artifacts = _iteration_artifacts(tmp_path)["iteration_gaps"]
    report_path = tmp_path / "strict_gap_report.json"
    report = json.loads(report_path.read_text())
    report["counts_by_type"]["FN"] = 99
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="type and class counts disagree"):
        MODULE.commit(state, "iteration_gaps", 1, artifacts)


def test_commit_rejects_out_of_order_stage(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    with pytest.raises(ValueError, match="expected stage candidate_cache"):
        MODULE.commit(state, "baseline_measurement", 0, [f"done={artifact}"])


def test_synthesis_stage_is_required_when_enabled(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_admission",
                 current_iteration=1, last_stage="iteration_retrieval")
    state.write_text(json.dumps(value))
    value = MODULE.commit(state, "iteration_admission", 1, [f"done={artifact}"])
    assert value["next_stage"] == "iteration_synthesis"


def test_failed_merged_role_fixture_cannot_commit_without_admission_preview(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))
    merged = tmp_path / "merged_query_manifest.json"
    merged.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "query_counts": {"real": 1063, "clean": 1396},
    }))
    with pytest.raises(ValueError, match="requires artifact admission_preview"):
        MODULE.commit(state, "iteration_retrieval", 1, [f"query_manifest={merged}"])


def test_commit_rejects_stale_routing_report_hash(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash/path mismatch for routing_report"):
        MODULE.commit(state, "iteration_retrieval", 1, _retrieval_artifacts(tmp_path, stale=True))


def test_commit_accepts_omitted_zero_count_query_branches(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))

    result = MODULE.commit(
        state, "iteration_retrieval", 1,
        _retrieval_artifacts(tmp_path, sparse_queries=True),
    )

    assert result["next_stage"] == "iteration_admission"


def _synthesis_state(root: Path) -> tuple[Path, Path, dict]:
    plan = root / "synthetic_plan.json"
    plan.write_text(json.dumps({"texture+crack": 5}))
    plan_artifact = {"path": str(plan.resolve()), "sha256": _sha(plan),
                     "bytes": plan.stat().st_size}
    state = root / "deft_state.json"
    state.write_text(json.dumps({
        "status": "RUNNING", "next_stage": "iteration_synthesis",
        "current_iteration": 1, "max_iterations": 1,
        "events": [{"stage": "iteration_retrieval", "iteration": 1,
                    "artifacts": {"synthetic_plan": plan_artifact}}],
    }))
    return state, plan, plan_artifact


def test_synthesis_commit_requires_plan_reconciliation(tmp_path: Path) -> None:
    state, _, plan_artifact = _synthesis_state(tmp_path)
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text(json.dumps({
        "status": "COMPLETE",
        "synthetic_plan": {"path": plan_artifact["path"],
                           "sha256": plan_artifact["sha256"]},
        "requested_total": 5, "generator_row_count": 4,
        "explicit_shortfall_total": 1,
        "per_type": {"texture+crack": {"requested_images": 5,
                                          "generator_rows": 4,
                                          "explicit_shortfall": 1}},
    }))
    value = MODULE.commit(
        state, "iteration_synthesis", 1,
        [f"synthesis_plan_reconciliation={reconciliation}"],
    )
    assert value["next_stage"] == "iteration_training"


def test_synthesis_commit_refuses_count_mismatch(tmp_path: Path) -> None:
    state, _, plan_artifact = _synthesis_state(tmp_path)
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text(json.dumps({
        "status": "COMPLETE",
        "synthetic_plan": {"path": plan_artifact["path"],
                           "sha256": plan_artifact["sha256"]},
        "requested_total": 5, "generator_row_count": 4,
        "explicit_shortfall_total": 0,
        "per_type": {"texture+crack": {"requested_images": 5,
                                          "generator_rows": 4,
                                          "explicit_shortfall": 0}},
    }))
    with pytest.raises(ValueError, match="count/shortfall mismatch"):
        MODULE.commit(
            state, "iteration_synthesis", 1,
            [f"synthesis_plan_reconciliation={reconciliation}"],
        )


def test_synthesis_commit_refuses_stale_plan_hash(tmp_path: Path) -> None:
    state, plan, plan_artifact = _synthesis_state(tmp_path)
    plan.write_text(json.dumps({"texture+crack": 7}))
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text(json.dumps({
        "status": "COMPLETE",
        "synthetic_plan": {"path": plan_artifact["path"],
                           "sha256": plan_artifact["sha256"]},
        "requested_total": 5, "generator_row_count": 4,
        "explicit_shortfall_total": 1,
        "per_type": {"texture+crack": {"requested_images": 5,
                                          "generator_rows": 4,
                                          "explicit_shortfall": 1}},
    }))
    with pytest.raises(ValueError, match="committed synthetic plan hash is stale"):
        MODULE.commit(
            state, "iteration_synthesis", 1,
            [f"synthesis_plan_reconciliation={reconciliation}"],
        )
