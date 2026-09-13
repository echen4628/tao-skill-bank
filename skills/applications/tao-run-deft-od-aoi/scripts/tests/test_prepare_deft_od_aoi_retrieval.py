# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_retrieval.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_retrieval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def _row(name: str, source: Path, defect: bool) -> dict:
    return {
        "id": 1, "file_name": source.name, "source_path": str(source),
        "width": 32, "height": 32,
        "deft_od_aoi": {
            "benchmark": "fixture", "texture": name,
            "defect_type": "bad" if defect else "",
            "generator_type": f"fixture_{name}+bad" if defect else "",
        },
    }


def _policy(root: Path) -> Path:
    sources = {}
    for role in ("real", "clean", "kpi"):
        images = root / role
        source = images / f"{role}.png"
        _image(source)
        boxed = role != "clean"
        annotations = ([{"id": 9, "image_id": 1, "category_id": 1,
                         "bbox": [8, 8, 12, 12]}] if boxed else [])
        coco = root / f"{role}.json"
        coco.write_text(json.dumps({
            "images": [_row(role, source, boxed)],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "defect"}],
        }))
        sources[role] = {"images": str(images), "coco": str(coco)}
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": sources,
        "gap": {"background_iou_upper": 0.05, "near_miss_iou_upper": 0.5},
        "retrieval": {
            "model": "SigLIP", "model_path": "siglip", "mode": "siglip_only",
            "defect_context_scale": 1.5, "clean_grids": [1, 2],
            "candidate_overfetch": 15, "output_size": 224,
            "audit_top_k_per_query": 20,
        },
    }))
    return policy


def test_candidate_cache_uses_proven_crops_and_one_role_aware_embedding_input(tmp_path: Path) -> None:
    report = MODULE.candidates(_policy(tmp_path), tmp_path / "candidates")
    assert report["counts"] == {"defect": 1, "clean": 5}
    frame = pd.read_parquet(tmp_path / "candidates/candidate_inputs.parquet")
    assert set(frame.role) == {"defect", "clean"}
    assert frame.groupby("role").parent_filepath.nunique().to_dict() == {
        "clean": 1, "defect": 1,
    }
    assert all(Path(path).is_file() for path in frame.filepath)
    assert (tmp_path / "candidates/embed_candidates.yaml").is_file()
    assert not list((tmp_path / "candidates").glob("mine_*.yaml"))


def test_queries_keep_strict_near_and_background_branches_distinct(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    kpi_image = tmp_path / "kpi/kpi.png"
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(kpi_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.1}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(kpi_image), "gap_type": "FP",
                   "bbox": [2, 2, 10, 10], "best_iou": value}
                  for value in (0.01, 0.2, 0.8)]).to_parquet(loose)
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    report = MODULE.queries(
        policy, strict, loose, 1, tmp_path / "queries", candidate_root
    )
    assert report["query_counts"] == {
        "strict_fn": 1, "near_miss_fp": 1, "background_fp": 1,
    }
    frame = pd.read_parquet(tmp_path / "queries/query_inputs.parquet")
    assert set(frame.branch) == {"strict_fn", "near_miss_fp", "background_fp"}
    assert set(frame.role) == {"defect", "clean"}
    assert not list((tmp_path / "queries").glob("mine_*.yaml"))


def test_retrieval_specs_and_crop_paths_bind_to_durable_copyback(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    scratch = tmp_path / "scratch_candidates"
    durable = tmp_path / "durable_candidates"
    report = MODULE.candidates(policy, scratch, durable)
    frame = pd.read_parquet(scratch / "candidate_inputs.parquet")
    spec = yaml.safe_load((scratch / "embed_candidates.yaml").read_text())

    assert all(str(path).startswith(str(durable.resolve()) + "/crops/")
               for path in frame.filepath)
    assert report["input_parquet"] == str(durable.resolve() / "candidate_inputs.parquet")
    assert spec["input_parquet"] == str(durable.resolve() / "candidate_inputs.parquet")
    assert spec["output_parquet"] == str(durable.resolve() / "candidate_embeddings.parquet")
