# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "prepare_anomalygennext_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_anomalygennext_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def mask(path: Path, offset: int = 0) -> None:
    value = np.zeros((32, 32), dtype=np.uint8)
    value[4 + offset:20 + offset, 4:20] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    defective = tmp_path / "defect.png"
    source_mask = tmp_path / "defect_mask.png"
    image(defective)
    mask(source_mask)
    pool = tmp_path / "pool" / "texture_1"
    image(pool / "clean_image" / "clean1.png", 10)
    image(pool / "clean_image" / "clean2.png", 20)
    mask(pool / "mask" / "crack" / "mask1.png")
    mask(pool / "mask" / "crack" / "mask2.png", 1)
    gaps = tmp_path / "gaps.parquet"
    pd.DataFrame([
        {"image_id": 1, "filepath": str(defective), "gap_type": "FN",
         "bbox": [4, 4, 20, 20], "class": "defect", "split": "kpi",
         "dataset_id": "example", "texture_id": "texture_1",
         "defect_class": "crack", "anomaly_type": "texture_1+crack",
         "fn_mask_source": str(source_mask)},
        {"image_id": 1, "filepath": str(defective), "gap_type": "FN",
         "bbox": [6, 6, 18, 18], "class": "defect", "split": "kpi",
         "dataset_id": "example", "texture_id": "texture_1",
         "defect_class": "crack", "anomaly_type": "texture_1+crack",
         "fn_mask_source": str(source_mask)},
    ]).to_parquet(gaps)
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"checkpoint")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture_1", "crack"]]}))
    defect_spec = tmp_path / "defect_spec.jsonl"
    defect_spec.write_text(json.dumps({"defect_type": "texture_1+crack",
                                       "spatial_dependency": "free"}) + "\n")
    config = tmp_path / "filtering.yaml"
    config.write_text(yaml.safe_dump({
        "source_tag": "test", "gap_parquet": str(gaps),
        "pool_dataset_root": str(tmp_path / "pool"), "defect_spec": str(defect_spec),
        "datasets": {"example": {"checkpoint": str(checkpoint), "recipe": str(recipe)}},
        "selection": {"mode": "all_eligible", "datasets": ["example"],
                      "mask_sample_seed": 7},
        "embedding": {"model": "SigLIP", "model_path": "local/siglip", "batch_size": 8},
    }, sort_keys=False))
    return config, gaps


def test_prepare_preserves_box_level_queries_and_unique_embedding_input(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    output = tmp_path / "output"
    report = MODULE.prepare(config, output)
    assert report == {"status": "COMPLETE", "source_tag": "test",
                      "compatibility_determinism": "native",
                      "selected_fn_count": 2, "clean_image_count": 2,
                      "source_mask_count": 4, "training_pool_mutated": False}
    queries = pd.read_parquet(output / "manifests" / "selected_fn_queries.parquet")
    embedding = pd.read_parquet(output / "manifests" / "fn_embedding_inputs.parquet")
    masks = pd.read_parquet(output / "manifests" / "mask_selection.parquet")
    assert len(queries) == 2 and queries.fn_id.nunique() == 2
    assert len(embedding) == 1
    assert masks.groupby("fn_id").branch.nunique().eq(2).all()
    assert all(Path(path).is_file() for path in masks.mask_path)
    clean_spec = yaml.safe_load((output / "specs" / "clean_embeddings.yaml").read_text())
    fn_spec = yaml.safe_load((output / "specs" / "fn_embeddings.yaml").read_text())
    assert clean_spec["model_path"] == fn_spec["model_path"] == "local/siglip"


def test_prepare_requires_normalized_identity(tmp_path: Path) -> None:
    config, gaps = fixture(tmp_path)
    frame = pd.read_parquet(gaps).drop(columns=["fn_mask_source"])
    frame.to_parquet(gaps)
    with pytest.raises(ValueError, match="normalized columns"):
        MODULE.prepare(config, tmp_path / "output")


def test_prepare_refuses_output_reuse(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.prepare(config, output)


def test_legacy_compatibility_preserves_ids_and_inclusive_bbox(tmp_path: Path) -> None:
    config, gaps = fixture(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["compatibility"] = {"determinism": "legacy_v1"}
    config.write_text(yaml.safe_dump(value, sort_keys=False))
    output = tmp_path / "legacy-output"
    MODULE.prepare(config, output)

    contract = json.loads(
        (output / "prepared_anomalygennext_inputs" / "input_contract.json").read_text()
    )
    assert contract["compatibility_determinism"] == "legacy_v1"

    queries = pd.read_parquet(output / "manifests" / "selected_fn_queries.parquet")
    gap_rows = pd.read_parquet(gaps)
    row = gap_rows.iloc[0]
    bbox_json = json.dumps(np.asarray(row.bbox).reshape(-1).tolist(), separators=(",", ":"))
    expected = "fn-" + MODULE._legacy_stable_id(
        "example", str(row.image_id), str(Path(row.filepath).resolve()), bbox_json
    )
    assert expected in set(queries.fn_id)
    expected_order = []
    for ordered in gap_rows.assign(
        bbox_json=gap_rows.bbox.map(
            lambda box: json.dumps(np.asarray(box).reshape(-1).tolist(), separators=(",", ":"))
        )
    ).sort_values("bbox_json").itertuples():
        expected_order.append(
            "fn-" + MODULE._legacy_stable_id(
                "example", str(ordered.image_id), str(Path(ordered.filepath).resolve()),
                ordered.bbox_json,
            )
        )
    assert queries.sort_values("query_order").fn_id.tolist() == expected_order

    direct_native = tmp_path / "native-mask.png"
    direct_legacy = tmp_path / "legacy-mask.png"
    MODULE._isolate_mask(Path(row.fn_mask_source), Path(row.filepath), [4, 4, 19, 19], direct_native)
    MODULE._isolate_mask(
        Path(row.fn_mask_source), Path(row.filepath), [4, 4, 19, 19], direct_legacy,
        legacy_inclusive_max=True,
    )
    assert np.asarray(Image.open(direct_legacy)).sum() > np.asarray(Image.open(direct_native)).sum()


def test_legacy_compatibility_allows_sampled_mask_reuse(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    candidates = [first, second]

    assert MODULE._available_sampled_masks(
        candidates, {first}, "legacy_v1"
    ) == candidates
    assert MODULE._available_sampled_masks(candidates, {first}, "native") == [second]


def test_legacy_compatibility_sorts_image_ids_lexicographically() -> None:
    rows = pd.DataFrame([
        {"dataset_id": "example", "image_id": 2, "filepath": "/z.png",
         "bbox_json": "[0,0,1,1]", "fn_id": "fn-2", "split": "kpi"},
        {"dataset_id": "example", "image_id": 10, "filepath": "/a.png",
         "bbox_json": "[0,0,1,1]", "fn_id": "fn-10", "split": "kpi"},
    ])
    selection = {"mode": "all_eligible", "datasets": ["example"]}

    legacy = MODULE._select(rows, selection, "legacy_v1")
    native = MODULE._select(rows, selection, "native")

    assert legacy.image_id.tolist() == [10, 2]
    assert native.image_id.tolist() == [2, 10]


def test_prepare_rejects_unknown_determinism_mode(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["compatibility"] = {"determinism": "unknown"}
    config.write_text(yaml.safe_dump(value, sort_keys=False))

    with pytest.raises(ValueError, match="unsupported compatibility.determinism"):
        MODULE.prepare(config, tmp_path / "output")
