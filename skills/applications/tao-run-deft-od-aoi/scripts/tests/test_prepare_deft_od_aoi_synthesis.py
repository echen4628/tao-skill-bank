# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_synthesis.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_synthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _plan(root: Path, counts: dict[str, int]) -> tuple[Path, str]:
    path = root / "synthetic_plan.json"
    path.write_text(json.dumps(counts, sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _route(root: Path, checkpoint: Path, recipe: Path) -> dict[str, str]:
    base = root / "generation-base"
    base.mkdir(exist_ok=True)
    vae = root / "Wan2.2_VAE.pth"
    vae.write_bytes(b"vae")
    return {"checkpoint": str(checkpoint), "recipe": str(recipe),
            "base_checkpoint": str(base), "vae_path": str(vae)}


def test_synthesis_normalizes_exact_kpi_false_negative(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    image, mask = images / "image.png", tmp_path / "mask.png"
    no_route_image = images / "no-route.png"
    image.write_bytes(b"image")
    no_route_image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 7, "file_name": image.name, "dataset_id": "route",
                    "texture_id": "texture"},
                   {"id": 8, "file_name": no_route_image.name,
                    "dataset_id": "evaluation_only"}],
        "annotations": [{"id": 3, "image_id": 7, "category_id": 1,
                          "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)},
                        {"id": 4, "image_id": 8, "category_id": 1,
                         "bbox": [1, 2, 3, 4]}],
        "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec, checkpoint, recipe = tmp_path / "defect.jsonl", tmp_path / "adapter.pt", tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": {"kpi": {"images": str(images),
                                                             "coco": str(coco)}},
                                      "retrieval": {"model": "SigLIP", "model_path": "siglip",
                                                    "candidate_overfetch": 15},
                                      "synthesis": {"enabled": True,
                                                    "pool_dataset_root": str(pool),
                                                    "defect_spec": str(defect_spec),
                                                    "routes": {"route": _route(tmp_path, checkpoint, recipe)},
                                                    "max_neighbors_per_fn": 5,
                                                    "min_similarity": 0.9,
                                                    "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": 7, "filepath": str(image), "gap_type": "FN",
                   "bbox": [4, 5, 14, 17], "class": "defect"},
                  {"image_id": 8, "filepath": str(no_route_image), "gap_type": "FN",
                   "bbox": [1, 2, 4, 6], "class": "defect"}]).to_parquet(gaps)
    plan, plan_sha = _plan(tmp_path, {"texture+crack": 2})
    published = tmp_path / "durable"
    report = MODULE.prepare(
        policy, gaps, plan, plan_sha, tmp_path / "out", published
    )
    assert report["fn_count"] == 1
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert normalized.anomaly_type == "texture+crack"
    config = yaml.safe_load((tmp_path / "out/anomalygen_filtering.yaml").read_text())
    assert config["gap_parquet"] == str(
        published / "normalized_fn_gaps.parquet"
    )
    assert report["config"] == str(published / "anomalygen_filtering.yaml")
    assert config["datasets"]["route"]["checkpoint"] == str(checkpoint.resolve())
    assert config["datasets"]["route"]["base_checkpoint"] == str(
        (tmp_path / "generation-base").resolve()
    )
    assert config["datasets"]["route"]["vae_checkpoint"] == str(
        (tmp_path / "Wan2.2_VAE.pth").resolve()
    )
    assert config["retrieval"]["candidate_topn"] == 3
    assert config["retrieval"]["max_neighbors_per_fn"] == 1
    assert config["synthetic_plan"]["sha256"] == plan_sha


def test_synthesis_accepts_gap_filename_stem_and_uses_normalized_image(
    tmp_path: Path,
) -> None:
    images = tmp_path / "normalized" / "kpi_images"
    nested = images / "btad" / "01"
    nested.mkdir(parents=True)
    image = nested / "btad_kpi_0000001.bmp"
    image.write_bytes(b"same pixels")
    original = tmp_path / "commercial_base" / "BTAD" / "0001.bmp"
    original.parent.mkdir(parents=True)
    os.link(image, original)
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "btad/01/btad_kpi_0000001.bmp",
                    "source_path": str(original),
                    "deft_od_aoi": {"dataset_id": "route", "texture_id": "texture",
                                     "defect_class": "crack", "fn_mask_source": str(mask)}}],
        "annotations": [{"id": 3, "image_id": 1, "category_id": 1,
                          "bbox": [4, 5, 10, 12]}],
        "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "defect_spec": str(defect_spec),
                      "routes": {"route": _route(tmp_path, checkpoint, recipe)},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": image.stem, "filepath": str(image), "gap_type": "FN",
                   "bbox": [4, 5, 14, 17], "class": "defect"}]).to_parquet(gaps)

    plan, plan_sha = _plan(tmp_path, {"texture+crack": 2})
    report = MODULE.prepare(policy, gaps, plan, plan_sha, tmp_path / "out")

    assert report["fn_count"] == 1
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert normalized.filepath == str(image.resolve())
    assert normalized.filepath != str(original.resolve())


def test_synthesis_plan_quota_is_deterministic_and_records_odd_shortfall(
    tmp_path: Path,
) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    coco_images, coco_annotations, gap_rows = [], [], []
    for image_id in (3, 1, 2):
        image = images / f"image-{image_id}.png"
        image.write_bytes(b"image")
        coco_images.append({"id": image_id, "file_name": image.name,
                            "dataset_id": "route", "texture_id": "texture"})
        coco_annotations.append({"id": image_id, "image_id": image_id,
                                 "category_id": 1, "bbox": [image_id, 2, 4, 5],
                                 "defect_class": "crack", "fn_mask_source": str(mask)})
        gap_rows.append({"image_id": image_id, "filepath": str(image), "gap_type": "FN",
                         "bbox": [image_id, 2, image_id + 4, 7], "class": "defect"})
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({"images": coco_images, "annotations": coco_annotations,
                                "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec, checkpoint, recipe = (tmp_path / "defect.jsonl", tmp_path / "adapter.pt",
                                        tmp_path / "recipe.yaml")
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 99},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "defect_spec": str(defect_spec),
                      "routes": {"route": _route(tmp_path, checkpoint, recipe)},
                      "max_neighbors_per_fn": 9, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame(gap_rows).to_parquet(gaps)
    plan, plan_sha = _plan(tmp_path, {"texture+crack": 5})

    first = MODULE.prepare(policy, gaps, plan, plan_sha, tmp_path / "out-a")
    second = MODULE.prepare(policy, gaps, plan, plan_sha, tmp_path / "out-b")

    assert first["fn_count"] == 2
    assert first["frozen_generator_rows"] == 4
    assert first["bounded_shortfall"] == 1
    assert first["per_type"]["texture+crack"]["requested_images"] == 5
    left = pd.read_parquet(tmp_path / "out-a/normalized_fn_gaps.parquet")
    right = pd.read_parquet(tmp_path / "out-b/normalized_fn_gaps.parquet")
    assert left.image_id.astype(str).tolist() == ["1", "2"]
    pd.testing.assert_frame_equal(left, right)


def test_synthesis_rejects_stale_plan_hash(tmp_path: Path) -> None:
    plan, plan_sha = _plan(tmp_path, {"texture+crack": 2})
    plan.write_text(json.dumps({"texture+crack": 4}) + "\n")
    with pytest.raises(ValueError, match="synthetic plan hash mismatch"):
        MODULE._synthetic_plan(plan, plan_sha)


def test_kpi_image_index_rejects_duplicate_nested_filename_stems() -> None:
    rows = [
        {"id": 1, "file_name": "one/shared.png"},
        {"id": 2, "file_name": "two/shared.jpg"},
    ]
    with pytest.raises(ValueError, match="duplicate KPI image identity: shared"):
        MODULE._image_index(rows)


def test_kpi_image_index_preserves_numeric_id_and_nested_stem_aliases() -> None:
    row = {"id": 7, "file_name": "nested/path/kpi_name.png"}
    index = MODULE._image_index([row])
    assert index["7"] is row
    assert index["kpi_name"] is row
