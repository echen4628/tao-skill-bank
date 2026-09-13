# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "resolve_deft_od_aoi_synthesis.py"
SPEC = importlib.util.spec_from_file_location("resolve_deft_od_aoi_synthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _policy(root: Path) -> tuple[Path, Path]:
    dataset, base, nn = root / "dataset", root / "base", root / "nn"
    dataset.mkdir()
    base.mkdir()
    nn.mkdir()
    validation, vae, defect = root / "validation.jsonl", root / "vae.pth", root / "defect.jsonl"
    validation.write_text("{}\n")
    vae.write_bytes(b"vae")
    defect.write_text("{}\n")
    handoff = root / "training" / "training_handoff.json"
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump({"synthesis": {"enabled": True,
        "defect_spec": str(defect), "routes": {"route": {"finetune": {
            "dataset_root": str(dataset), "validation_testcase": str(validation),
            "base_checkpoint": str(base), "vae_path": str(vae), "nn_backbone": str(nn),
            "result_handoff": str(handoff)}}}}}))
    return policy, handoff


def test_resolution_requests_training_then_accepts_hash_bound_handoff(tmp_path: Path) -> None:
    policy, handoff = _policy(tmp_path)
    requested = MODULE.resolve(policy, tmp_path / "request")
    assert requested["status"] == "NEEDS_TRAINING"
    assert requested["requests"][0]["dataset_name"] == "route"
    checkpoint, recipe = tmp_path / "adapter.pt", tmp_path / "recipe.yaml"
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, defect]]\n")
    handoff.parent.mkdir()
    handoff.write_text(json.dumps({"status": "COMPLETE", "dataset_name": "route",
                                   "checkpoint": str(checkpoint), "recipe": str(recipe),
                                   "checkpoint_sha256": hashlib.sha256(b"adapter").hexdigest(),
                                   "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                                   "anomaly_types": ["texture+defect"]}))
    resolved = MODULE.resolve(policy, tmp_path / "resolved")
    assert resolved["status"] == "COMPLETE"
    value = yaml.safe_load(Path(resolved["resolved_policy"]).read_text())
    route = value["synthesis"]["routes"]["route"]
    assert route["checkpoint"] == str(checkpoint.resolve())
    assert route["base_checkpoint"] == str((tmp_path / "base").resolve())
    assert route["vae_path"] == str((tmp_path / "vae.pth").resolve())


def test_ready_route_requires_generation_runtime_assets(tmp_path: Path) -> None:
    checkpoint, recipe, defect = tmp_path / "adapter.pt", tmp_path / "recipe.yaml", tmp_path / "defect.jsonl"
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, defect]]\n")
    defect.write_text("{}\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"synthesis": {"enabled": True,
        "defect_spec": str(defect), "routes": {"route": {
            "checkpoint": str(checkpoint), "recipe": str(recipe)}}}}))

    with pytest.raises(ValueError, match="generation runtime fields"):
        MODULE.resolve(policy, tmp_path / "resolved")
