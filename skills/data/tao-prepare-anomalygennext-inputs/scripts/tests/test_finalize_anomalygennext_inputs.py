# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "finalize_anomalygennext_inputs.py"
SPEC = importlib.util.spec_from_file_location("finalize_anomalygennext_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _image(path: Path, full: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.startswith("clean"):
        value = np.full((16, 16, 3), 80, dtype=np.uint8)
    else:
        value = np.full((16, 16), 255 if full else 0, dtype=np.uint8)
        if not full:
            value[3:9, 4:10] = 255
    Image.fromarray(value).save(path)


def _fixture(root: Path, reject_first: bool = False, requested: int = 4) -> None:
    manifests = root / "manifests"
    prepared = root / "prepared_anomalygennext_inputs"
    manifests.mkdir(parents=True)
    prepared.mkdir()
    checkpoint, recipe = root / "model.pt", root / "recipe.yaml"
    base, vae = root / "generation-base", root / "Wan2.2_VAE.pth"
    base.mkdir()
    vae.write_bytes(b"vae")
    checkpoint.write_bytes(b"model")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    plan = root / "synthetic_plan.json"
    plan.write_text(json.dumps({"texture+crack": requested}) + "\n")
    config = {"source_tag": "test", "pool_dataset_root": str(root / "pool"),
              "datasets": {"dataset": {"checkpoint": str(checkpoint), "recipe": str(recipe),
                                          "base_checkpoint": str(base),
                                          "vae_checkpoint": str(vae)}},
              "retrieval": {"max_neighbors_per_fn": 1},
              "synthetic_plan": {"path": str(plan),
                                 "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                                 "counts": {"texture+crack": requested}}}
    (prepared / "filtering_config.yaml").write_text(yaml.safe_dump(config))
    (prepared / "input_contract.json").write_text('{"status":"COMPLETE"}\n')
    clean_a, clean_b = root / "clean-a.png", root / "clean-b.png"
    _image(clean_a)
    _image(clean_b)
    rows = []
    for order, fn_id in enumerate(("fn-1", "fn-2")):
        for rank, clean in enumerate((clean_a, clean_b), start=1):
            rows.append({"candidate_id": f"{fn_id}-pair-{rank}", "fn_id": fn_id,
                         "query_order": order, "dataset_id": "dataset",
                         "anomaly_type": "texture+crack", "od_category": "defect",
                         "fn_filepath": str(root / "defect.png"),
                         "clean_filepath": str(clean), "neighbor_rank": rank,
                         "cosine_similarity": 1.0 - rank / 10,
                         "eligible_for_amp": True, "gate_reason": ""})
    pd.DataFrame(rows).to_parquet(manifests / "knn_candidates.parquet")
    mask_rows, amp_rows = [], []
    for fn_id in ("fn-1", "fn-2"):
        for branch in sorted(MODULE.BRANCHES):
            source = root / "source" / f"{fn_id}__{branch}.png"
            _image(source)
            mask_rows.append({"fn_id": fn_id, "branch": branch, "mask_path": str(source)})
            for clean in (clean_a, clean_b):
                aligned = root / "amp" / clean.stem / f"{source.stem}__seed0.png"
                _image(aligned, full=reject_first and fn_id == "fn-1" and clean == clean_a)
                amp_rows.append({"image_filename": str(clean), "mask_filename": str(aligned),
                                 "anomaly_type": "texture+crack"})
    pd.DataFrame(mask_rows).to_parquet(manifests / "mask_selection.parquet")
    (root / "amp" / "testcase.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in amp_rows)
    )


def test_finalize_preserves_same_clean_across_false_negatives(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = MODULE.finalize(tmp_path)
    assert report["selected_fn_count"] == 2
    assert report["selected_pair_count"] == 2
    assert report["generator_row_count"] == 4
    reconciliation = report["synthesis_plan_reconciliation"]
    assert reconciliation["generator_row_count"] == 4
    assert reconciliation["explicit_shortfall_total"] == 0
    assert reconciliation["per_type"]["texture+crack"]["explicit_shortfall"] == 0
    assert any(
        Path(row["path"]).name == "synthesis_plan_reconciliation.json"
        for row in report["artifacts"]
    )
    selected = pd.read_parquet(tmp_path / "manifests" / "selected_pairs.parquet")
    assert selected.clean_filepath.nunique() == 1
    assert report["training_pool_mutated"] is False
    assert all(Path(row["path"]).is_file() for row in report["artifacts"])
    assert all(MODULE._sha256(Path(row["path"])) == row["sha256"] for row in report["artifacts"])
    generation_plan = json.loads(
        (tmp_path / "prepared_anomalygennext_inputs/anomalygen_next_generation_plan.json").read_text()
    )
    assert generation_plan[0]["base_checkpoint"] == str(tmp_path / "generation-base")
    assert generation_plan[0]["vae_checkpoint"] == str(tmp_path / "Wan2.2_VAE.pth")


def test_finalize_rejects_full_image_mask_and_uses_next_neighbor(tmp_path: Path) -> None:
    _fixture(tmp_path, reject_first=True)
    MODULE.finalize(tmp_path)
    selected = pd.read_parquet(tmp_path / "manifests" / "selected_pairs.parquet")
    first = selected[selected.fn_id == "fn-1"].iloc[0]
    assert Path(first.clean_filepath).name == "clean-b.png"
    status = pd.read_parquet(tmp_path / "manifests" / "knn_roi_status.parquet")
    rejected = status[(status.fn_id == "fn-1") & (status.neighbor_rank == 1)].iloc[0]
    assert rejected.selection_reason.endswith(":full_image")


def test_finalize_records_explicit_plan_shortfall(tmp_path: Path) -> None:
    _fixture(tmp_path, requested=5)
    report = MODULE.finalize(tmp_path)
    reconciliation = report["synthesis_plan_reconciliation"]
    assert reconciliation["requested_total"] == 5
    assert reconciliation["generator_row_count"] == 4
    assert reconciliation["explicit_shortfall_total"] == 1
    assert reconciliation["per_type"]["texture+crack"] == {
        "requested_images": 5,
        "generator_rows": 4,
        "explicit_shortfall": 1,
    }


def test_finalize_rejects_stale_hash_bound_plan(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "synthetic_plan.json").write_text(
        json.dumps({"texture+crack": 6}) + "\n"
    )
    with pytest.raises(ValueError, match="synthetic plan path/hash is stale"):
        MODULE.finalize(tmp_path)
