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


def _retrieval_artifacts(root: Path, stale: bool = False) -> list[str]:
    query = root / "query_manifest.json"
    report = root / "routing_report.json"
    mined = root / "mined_manifest.json"
    defect = root / "defect_ledger.json"
    clean = root / "clean_ledger.json"
    synthetic = root / "synthetic_plan.json"
    index = root / "admission_index.npy"
    query.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "query_counts": {"strict_fn": 1, "near_miss_fp": 1, "background_fp": 1},
    }))
    selected = {
        "strict_fn_real": 1, "near_miss_real": 1,
        "uniform_real": 0, "background_clean": 1,
    }
    report.write_text(json.dumps({
        "queries": {"strict_fn": 1, "near_miss_fp": 1, "background_fp": 1},
        "selected": selected,
    }))
    mined.write_text(json.dumps([
        {"branch": "strict_fn_real"}, {"branch": "near_miss_real"},
        {"branch": "background_clean"},
    ]))
    defect.write_text(json.dumps(["real-a", "real-b"]))
    clean.write_text(json.dumps(["clean-a"]))
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
            "queries": {"strict_fn": 1, "near_miss_fp": 1, "background_fp": 1},
            "selected": selected, "manifest_records": 3,
            "defect_ledger": 2, "clean_ledger": 1,
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


def test_commit_enforces_order_and_completes_after_final_gaps(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    stages = [("candidate_cache", 0), ("baseline_measurement", 0),
              ("baseline_gaps", 0), ("iteration_retrieval", 1),
              ("iteration_admission", 1), ("iteration_training", 1),
              ("iteration_measurement", 1), ("iteration_gaps", 1)]
    value = None
    for stage, iteration in stages:
        values = (
            _retrieval_artifacts(tmp_path)
            if stage == "iteration_retrieval" else [f"done={artifact}"]
        )
        value = MODULE.commit(state, stage, iteration, values)
    assert value["status"] == "COMPLETE" and value["next_stage"] is None
    assert len(value["events"]) == len(stages)
    assert len((tmp_path / "loop_log.jsonl").read_text().splitlines()) == len(stages)


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
