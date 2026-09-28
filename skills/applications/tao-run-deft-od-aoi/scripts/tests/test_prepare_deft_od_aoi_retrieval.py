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


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_retrieval.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_retrieval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def _oriented_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[274] = 8
    Image.fromarray(np.full((20, 40, 3), 80, dtype=np.uint8)).save(path, exif=exif)


def _policy(root: Path, profile: str = "tight_context",
            strategy: str = "max_similarity") -> Path:
    sources = {}
    for role in ("real", "clean"):
        images = root / role
        source = images / f"{role}.png"
        _image(source)
        annotations = ([{"id": 9, "image_id": 1, "category_id": 1,
                         "bbox": [8, 8, 12, 12]}] if role == "real" else [])
        coco = root / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": source.name}],
                                    "annotations": annotations,
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(images), "coco": str(coco)}
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": sources,
                                      "gap": {"background_iou_upper": 0.05,
                                              "near_miss_iou_upper": 0.5},
                                      "retrieval": {"model": "SigLIP", "model_path": "siglip",
                                                    "selection": {"strategy": strategy},
                                                    "preprocessing": {"profile": profile},
                                                    "defect_context_scale": 1.5,
                                                    "clean_grids": [1, 2],
                                                    "output_size": 224,
                                                    "audit_top_k_per_query": 20,
                                                    "candidate_overfetch": 15},
                                      "routing": {"real_mine_factor_min": 1,
                                                  "real_mine_factor_max": 6,
                                                  "round_robin_real_factor_default": 3,
                                                  "near_miss_real_factor": 2,
                                                  "near_miss_real_cap": 20,
                                                  "clean_factor": 2}}))
    return policy


def _add_kpi(policy: Path, image: Path) -> None:
    document = yaml.safe_load(policy.read_text())
    coco = image.parent / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image.name, "source_path": str(image),
                    "deft_od_aoi": {"benchmark": "visa", "texture": "pcb1",
                                    "defect_type": "bad"}}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(image.parent), "coco": str(coco)}
    policy.write_text(yaml.safe_dump(document))


def _empty_role(policy: Path, role: str) -> None:
    value = yaml.safe_load(policy.read_text())
    coco = Path(value["sources"][role]["coco"])
    document = json.loads(coco.read_text())
    document["images"] = []
    document["annotations"] = []
    coco.write_text(json.dumps(document))


def test_candidate_cache_uses_defect_crops_and_clean_grid(tmp_path: Path) -> None:
    report = MODULE.candidates(_policy(tmp_path), tmp_path / "candidates")
    assert report["counts"] == {"real": 1, "clean": 5}
    real = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    clean = pd.read_parquet(tmp_path / "candidates/clean_candidates.parquet")
    assert real.source_filepath.nunique() == clean.source_filepath.nunique() == 1
    assert all(Path(path).is_file() for path in list(real.filepath) + list(clean.filepath))
    assert all(value.startswith("real-") for value in real.candidate_id)
    assert Image.open(real.iloc[0].filepath).size == (18, 18)
    assert {Image.open(path).size for path in clean.filepath} == {(16, 16), (32, 32)}


def test_square_context_reproduces_historical_fixed_size_crops(tmp_path: Path) -> None:
    report = MODULE.candidates(
        _policy(tmp_path, "square_context"), tmp_path / "candidates"
    )

    assert report["counts"] == {"real": 1, "clean": 5}
    real = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    clean = pd.read_parquet(tmp_path / "candidates/clean_candidates.parquet")
    assert real.candidate_id.tolist() == ["defect:1:9"]
    assert clean.candidate_id.tolist() == [
        "clean:1:g1:r0:c0", "clean:1:g2:r0:c0", "clean:1:g2:r0:c1",
        "clean:1:g2:r1:c0", "clean:1:g2:r1:c1",
    ]
    assert all(Image.open(path).size == (224, 224)
               for path in [*real.filepath, *clean.filepath])
    manifest = json.loads((tmp_path / "candidates/candidate_manifest.json").read_text())
    assert manifest["preprocessing_profile"] == "square_context"


def test_square_context_preserves_historical_string_id_order(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "square_context")
    value = yaml.safe_load(policy.read_text())
    for role in ("real", "clean"):
        images = Path(value["sources"][role]["images"])
        for image_id in (2, 10):
            _image(images / f"{role}-{image_id}.png", image_id)
        coco = Path(value["sources"][role]["coco"])
        document = json.loads(coco.read_text())
        document["images"] = [
            {"id": 2, "file_name": f"{role}-2.png"},
            {"id": 10, "file_name": f"{role}-10.png"},
        ]
        document["annotations"] = (
            [
                {"id": 4, "image_id": 2, "category_id": 1, "bbox": [8, 8, 12, 12]},
                {"id": 3, "image_id": 10, "category_id": 1, "bbox": [8, 8, 12, 12]},
                {"id": 20, "image_id": 10, "category_id": 1, "bbox": [8, 8, 12, 12]},
            ] if role == "real" else []
        )
        coco.write_text(json.dumps(document))

    MODULE.candidates(policy, tmp_path / "candidates")

    real = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    clean = pd.read_parquet(tmp_path / "candidates/clean_candidates.parquet")
    assert real.candidate_id.tolist() == [
        "defect:10:20", "defect:10:3", "defect:2:4",
    ]
    assert clean.candidate_id.iloc[0] == "clean:10:g1:r0:c0"
    assert clean.candidate_id.iloc[5] == "clean:2:g1:r0:c0"


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([10, -0.284, 20, 10], (10, 0, 20, 10)),
        ([-0.284, 10, 20, 20], (0, 10, 20, 20)),
        ([10, 20, 20, 32.284], (10, 20, 20, 32)),
        ([20, 10, 32.284, 20], (20, 10, 32, 20)),
    ],
)
def test_gap_box_clips_partial_detector_boxes(
    bbox: list[float], expected: tuple[int, int, int, int]
) -> None:
    assert MODULE._gap_box(bbox, 32, 32, 1.0) == expected


def test_gap_box_rejects_fully_outside_detector_box() -> None:
    with pytest.raises(ValueError, match="gap box clips empty"):
        MODULE._gap_box([10, -20, 20, -1], 32, 32, 1.0)


def test_source_annotation_box_remains_strict() -> None:
    with pytest.raises(ValueError, match="invalid xywh box"):
        MODULE._box([-0.284, 10, 20, 10], 32, 32)


def test_candidate_cache_applies_exif_orientation_before_cropping(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    source = tmp_path / "real/real.png"
    _oriented_image(source)
    coco = tmp_path / "real.json"
    payload = json.loads(coco.read_text())
    payload["images"][0].update(width=20, height=40)
    payload["annotations"][0]["bbox"] = [2, 25, 5, 5]
    coco.write_text(json.dumps(payload))

    report = MODULE.candidates(policy, tmp_path / "candidates")

    assert report["counts"]["real"] == 1
    frame = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    with Image.open(frame.iloc[0].filepath) as crop:
        assert crop.size == (9, 9)


@pytest.mark.parametrize("empty_role", ("real", "clean"))
def test_candidate_cache_records_empty_role_without_embedding_spec(
        tmp_path: Path, empty_role: str) -> None:
    policy = _policy(tmp_path)
    _empty_role(policy, empty_role)

    report = MODULE.candidates(policy, tmp_path / "candidates")

    assert report["counts"][empty_role] == 0
    assert report["role_status"][empty_role] == {
        "status": "EXHAUSTED", "candidate_count": 0, "reason": "empty_source_role"}
    assert not (tmp_path / f"candidates/{empty_role}_candidates.parquet").exists()
    assert not (tmp_path / f"candidates/embed_{empty_role}_candidates.yaml").exists()


def test_queries_route_fn_near_miss_and_background_fp(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "query.png"
    _image(query_image)
    document = json.loads(policy.read_text()) if policy.suffix == ".json" else yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({"images": [{"id": 1, "file_name": query_image.name,
                                            "source_path": str(query_image),
                                            "deft_od_aoi": {"benchmark": "visa",
                                                            "texture": "pcb1",
                                                            "defect_type": "bad"}}],
                                "annotations": [], "categories": [{"id": 1, "name": "defect"}]}))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.1}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [2, -0.284, 10, 10], "best_iou": value}
                  for value in (0.01, 0.2, 0.8)]).to_parquet(loose)
    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None)
    assert report["query_counts"] == {"real": 2, "clean": 1}
    assert report["admission_targets"] == {
        "real": {"fn": 1, "near_miss_fp": 2},
        "clean": {"background_fp": 2},
    }
    real = pd.read_parquet(tmp_path / "queries/real_queries.parquet")
    assert set(real.reason) == {"fn", "near_miss_fp"}
    real_mining = yaml.safe_load((tmp_path / "queries/mine_real.yaml").read_text())
    clean_mining = yaml.safe_load((tmp_path / "queries/mine_clean.yaml").read_text())
    assert real_mining["desired_unique_count"] == 1
    assert clean_mining["desired_unique_count"] == 5
    assert real_mining["candidate_expansion_factor"] == 15


def test_queries_exclude_crops_from_previously_admitted_sources(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "query.png"
    _image(query_image)
    document = yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({
        "images": [{"id": 1, "file_name": query_image.name,
                    "source_path": str(query_image),
                    "deft_od_aoi": {"benchmark": "visa", "texture": "pcb1",
                                    "defect_type": "bad"}}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.1}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [2, 2, 10, 10], "best_iou": 0.01}]).to_parquet(loose)

    candidates = tmp_path / "candidates"
    candidates.mkdir()
    prior_real = tmp_path / "prior_real.png"
    novel_real = tmp_path / "novel_real.png"
    prior_clean = tmp_path / "prior_clean.png"
    novel_clean = tmp_path / "novel_clean.png"
    for path in (prior_real, novel_real, prior_clean, novel_clean):
        _image(path)
    pd.DataFrame([
        {"filepath": str(candidates / "real-prior-a.png"),
         "source_filepath": str(prior_real)},
        {"filepath": str(candidates / "real-prior-b.png"),
         "source_filepath": str(prior_real)},
        {"filepath": str(candidates / "real-novel.png"),
         "source_filepath": str(novel_real)},
    ]).to_parquet(candidates / "real_candidates.parquet", index=False)
    pd.DataFrame([
        {"filepath": str(candidates / "clean-prior.png"),
         "source_filepath": str(prior_clean)},
        {"filepath": str(candidates / "clean-novel.png"),
         "source_filepath": str(novel_clean)},
    ]).to_parquet(candidates / "clean_candidates.parquet", index=False)
    (candidates / "candidate_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "counts": {"real": 3, "clean": 2},
    }))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"images": [
        {"deft_kind": "real_defect", "original_source_path": str(prior_real)},
        {"deft_kind": "clean_negative", "source_path": str(prior_clean)},
        {"deft_kind": "synthetic_defect", "source_path": str(tmp_path / "synthetic.png")},
    ]}))

    report = MODULE.queries(policy, strict, loose, 4, tmp_path / "queries",
                            candidates, None, previous)

    assert report["excluded_candidate_crops"] == {"real": 2, "clean": 1}
    assert report["excluded_source_images"] == {"real": 1, "clean": 1}
    real_spec = yaml.safe_load((tmp_path / "queries/mine_real.yaml").read_text())
    clean_spec = yaml.safe_load((tmp_path / "queries/mine_clean.yaml").read_text())
    real_excluded = pd.read_parquet(real_spec["exclude_path"])
    clean_excluded = pd.read_parquet(clean_spec["exclude_path"])
    assert set(real_excluded.filepath) == {
        str(candidates / "real-prior-a.png"), str(candidates / "real-prior-b.png")
    }
    assert set(clean_excluded.filepath) == {str(candidates / "clean-prior.png")}


@pytest.mark.parametrize(
    ("size", "box", "expected_box", "expected_padding"),
    [
        ((32, 32), (0, 0, 2, 2), (0, 0, 8, 8), (0, 0, 0, 0)),
        ((32, 32), (30, 30, 32, 32), (24, 24, 32, 32), (0, 0, 0, 0)),
        ((5, 3), (0, 0, 1, 1), (0, 0, 5, 3), (3, 4, 0, 1)),
        ((32, 32), (4, 5, 20, 24), (4, 5, 20, 24), (0, 0, 0, 0)),
    ],
)
def test_minimum_crop_geometry_expands_then_pads_only_when_required(
    size: tuple[int, int], box: tuple[int, int, int, int],
    expected_box: tuple[int, int, int, int], expected_padding: tuple[int, int, int, int],
) -> None:
    assert MODULE._minimum_crop_geometry(box, *size) == (expected_box, expected_padding)


def test_tiny_gap_crops_are_embedding_safe_at_boundaries(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "narrow.png"
    Image.fromarray(np.full((3, 5, 3), 80, dtype=np.uint8)).save(query_image)
    _add_kpi(policy, query_image)
    MODULE.candidates(policy, tmp_path / "candidates")
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [0, 0, 1, 1], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 2, 5, 3], "best_iou": 0.01}]).to_parquet(loose)

    MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                   tmp_path / "candidates", None)

    crops = list((tmp_path / "queries/crops").rglob("*.png"))
    assert len(crops) == 2
    assert all(Image.open(path).size == (8, 8) for path in crops)


@pytest.mark.parametrize("empty_role", ("real", "clean"))
def test_queries_skip_empty_role_while_other_role_continues(
        tmp_path: Path, empty_role: str) -> None:
    policy = _policy(tmp_path)
    _empty_role(policy, empty_role)
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None)

    continuing_role = "clean" if empty_role == "real" else "real"
    assert report["enabled_roles"] == [continuing_role]
    assert report["role_status"][empty_role]["status"] == "EXHAUSTED"
    assert {key: report["role_status"][empty_role][key] for key in
            ("candidate_count", "excluded_count", "remaining_candidate_count")} == {
                "candidate_count": 0, "excluded_count": 0, "remaining_candidate_count": 0}
    assert report["role_status"][continuing_role]["status"] == "READY"
    assert not (tmp_path / f"queries/embed_{empty_role}_queries.yaml").exists()
    assert not (tmp_path / f"queries/mine_{empty_role}.yaml").exists()
    assert (tmp_path / f"queries/mine_{continuing_role}.yaml").is_file()


def test_initially_empty_roles_converge_when_synthesis_cannot_add_data(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    for role in ("real", "clean"):
        _empty_role(policy, role)
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None)

    assert report["enabled_roles"] == []
    assert report["converged"] is True and report["synthesis_pending"] is False
    assert all(evidence["status"] == "EXHAUSTED"
               for evidence in report["role_status"].values())
    assert not list((tmp_path / "queries").glob("embed_*_queries.yaml"))
    assert not list((tmp_path / "queries").glob("mine_*.yaml"))


def test_exhausted_role_is_skipped_while_other_role_continues(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    real_candidates = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "real.png",
                    "source_path": real_candidates.iloc[0].source_filepath}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None, previous)

    assert report["enabled_roles"] == ["clean"]
    assert report["role_status"]["real"]["status"] == "EXHAUSTED"
    assert report["role_status"]["clean"]["status"] == "READY"
    assert not (tmp_path / "queries/mine_real.yaml").exists()
    assert (tmp_path / "queries/mine_clean.yaml").is_file()


def test_all_roles_exhausted_emit_convergence_without_mining_specs(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    sources = []
    for role in ("real", "clean"):
        frame = pd.read_parquet(tmp_path / f"candidates/{role}_candidates.parquet")
        sources.extend(sorted(set(frame.source_filepath.astype(str))))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"images": [
        {"id": index, "file_name": Path(source).name, "source_path": source}
        for index, source in enumerate(sources, start=1)
    ], "annotations": [], "categories": [{"id": 1, "name": "defect"}]}))
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None, previous)

    assert report["enabled_roles"] == [] and report["converged"] is True
    assert {value["status"] for value in report["role_status"].values()} == {"EXHAUSTED"}
    assert not list((tmp_path / "queries").glob("mine_*.yaml"))


def test_all_roles_exhausted_do_not_preempt_pending_synthesis(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value["synthesis"] = {"enabled": True}
    policy.write_text(yaml.safe_dump(value))
    MODULE.candidates(policy, tmp_path / "candidates")
    sources = []
    for role in ("real", "clean"):
        frame = pd.read_parquet(tmp_path / f"candidates/{role}_candidates.parquet")
        sources.extend(sorted(set(frame.source_filepath.astype(str))))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"images": [
        {"id": index, "file_name": Path(source).name, "source_path": source}
        for index, source in enumerate(sources, start=1)
    ]}))
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None, previous)

    assert report["enabled_roles"] == []
    assert report["synthesis_pending"] is True and report["converged"] is False


def test_queries_reject_missing_candidate_manifest(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    with pytest.raises(FileNotFoundError, match="candidate manifest"):
        MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                       tmp_path / "missing-candidates", None)


def test_square_context_applies_to_queries_independently_of_routing(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "square_context")
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [0, 0, 8, 4], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    report = MODULE.queries(
        policy, strict, loose, 1, tmp_path / "queries", tmp_path / "candidates", None
    )

    frame = pd.read_parquet(tmp_path / "queries/real_queries.parquet")
    assert report["preprocessing_profile"] == "square_context"
    assert Image.open(frame.iloc[0].filepath).size == (224, 224)


@pytest.mark.parametrize(
    ("profile", "expected_size"),
    (("tight_context", (12, 8)), ("square_context", (224, 224))),
)
def test_round_robin_selection_is_independent_of_preprocessing(
        tmp_path: Path, profile: str, expected_size: tuple[int, int]) -> None:
    policy = _policy(tmp_path, profile, "round_robin_similarity")
    query_image = tmp_path / "query.png"
    _image(query_image)
    document = yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({
        "images": [{"id": 7, "file_name": query_image.name,
                    "source_path": str(query_image), "dataset_id": "line-a",
                    "texture_id": "board", "defect_class": "bridge"}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [8, 8, 16, 12], "best_iou": 0.0,
                   "real_factor": 3}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    candidates = tmp_path / "candidates"
    MODULE.candidates(policy, candidates)

    report = MODULE.queries(
        policy, strict, loose, 1, tmp_path / "queries", candidates, None
    )

    frame = pd.read_parquet(tmp_path / "queries/real_queries.parquet")
    assert report["selection_strategy"] == "round_robin_similarity"
    assert frame.loc[0, ["benchmark", "texture", "defect_type", "real_factor"]].tolist() == [
        "line-a", "board", "bridge", 3,
    ]
    assert Image.open(frame.iloc[0].filepath).size == expected_size
    assert not (tmp_path / "queries/mine_real.yaml").exists()


def test_round_robin_defaults_to_three_real_candidates_per_strict_fn(
        tmp_path: Path) -> None:
    policy = _policy(tmp_path, "square_context", "round_robin_similarity")
    query_image = tmp_path / "query.png"
    _image(query_image)
    document = yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({
        "images": [{"id": 7, "file_name": query_image.name,
                    "source_path": str(query_image), "dataset_id": "line-a",
                    "texture_id": "board", "defect_class": "bridge"}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [8, 8, 16, 12], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)
    MODULE.candidates(policy, tmp_path / "candidates")

    MODULE.queries(
        policy, strict, loose, 1, tmp_path / "queries", tmp_path / "candidates", None
    )

    frame = pd.read_parquet(tmp_path / "queries/real_queries.parquet")
    assert frame.real_factor.tolist() == [3]


def test_max_similarity_keeps_one_x_real_factor_default(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "square_context", "max_similarity")
    query_image = tmp_path / "query.png"
    _image(query_image)
    _add_kpi(policy, query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [8, 8, 16, 12], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)
    MODULE.candidates(policy, tmp_path / "candidates")

    MODULE.queries(
        policy, strict, loose, 1, tmp_path / "queries", tmp_path / "candidates", None
    )

    mining = yaml.safe_load((tmp_path / "queries/mine_real.yaml").read_text())
    assert mining["desired_unique_count"] == 1


def test_unknown_preprocessing_profile_is_rejected(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "unknown")
    try:
        MODULE.candidates(policy, tmp_path / "candidates")
    except ValueError as error:
        assert "tight_context or square_context" in str(error)
    else:
        raise AssertionError("unknown preprocessing profile was accepted")
