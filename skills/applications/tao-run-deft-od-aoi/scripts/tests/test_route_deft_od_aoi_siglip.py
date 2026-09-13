# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from deft_od_aoi_policy import build_policy  # noqa: E402
from route_deft_od_aoi import Admission  # noqa: E402
from route_deft_od_aoi_siglip import route  # noqa: E402


def _image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)


def _coco(root: Path, name: str, count: int, clean: bool) -> tuple[Path, Path, list[Path]]:
    images_dir = root / name
    images_dir.mkdir()
    images = []
    annotations = []
    paths = []
    for index in range(count):
        path = images_dir / f"{name}_{index}.png"
        _image(path, index + (100 if clean else 0))
        paths.append(path.resolve())
        images.append({
            "id": index + 1, "file_name": path.name, "source_path": str(path.resolve()),
            "width": 32, "height": 32,
            "deft_od_aoi": {
                "benchmark": "fixture", "texture": "board",
                "defect_type": "" if clean else "bad",
                "generator_type": "" if clean else "fixture_board+bad",
            },
        })
        if not clean:
            annotations.append({
                "id": index + 1, "image_id": index + 1, "category_id": 1,
                "bbox": [4, 4, 12, 12],
            })
    coco = root / f"{name}.json"
    coco.write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "defect"}],
    }))
    return coco, images_dir, paths


def test_role_and_pocket_controller_matches_golden_factor_behavior(tmp_path: Path) -> None:
    source_coco, source_dir, real_paths = _coco(tmp_path, "real", 6, False)
    clean_coco, clean_dir, clean_paths = _coco(tmp_path, "clean", 5, True)
    kpi_coco, _, kpi_paths = _coco(tmp_path, "kpi", 1, False)
    policy = build_policy(max_iterations=2, synthetic_enabled=False)
    policy["routing"]["clean_factor"] = 5
    policy["admission"].update({
        "duplicate_global_cosine": 1.1,
        "duplicate_defect_cosine": 1.1,
        "clean_duplicate_global_cosine": 1.1,
        "clean_cluster_cosine": 1.1,
        "minimum_box_area_px": 1,
    })
    policy_path = tmp_path / "routing_policy.json"
    policy_path.write_text(json.dumps(policy))

    candidate_rows = [
        {
            "filepath": f"/crop/real_{index}.png", "candidate_id": f"r{index}",
            "role": "defect", "parent_filepath": str(path), "embedding": [1.0, 0.0],
        }
        for index, path in enumerate(real_paths)
    ] + [
        {
            "filepath": f"/crop/clean_{index}.png", "candidate_id": f"c{index}",
            "role": "clean", "parent_filepath": str(path), "embedding": [1.0, 0.0],
        }
        for index, path in enumerate(clean_paths)
    ]
    query_rows = [
        {
            "filepath": "/query/strict.png", "query_id": "strict", "role": "defect",
            "branch": "strict_fn", "benchmark": "fixture", "texture": "board",
            "defect_type": "bad", "generator_type": "fixture_board+bad",
            "embedding": [1.0, 0.0],
        },
        {
            "filepath": "/query/near.png", "query_id": "near", "role": "defect",
            "branch": "near_miss_fp", "benchmark": "fixture", "texture": "board",
            "defect_type": "bad", "generator_type": "fixture_board+bad",
            "embedding": [1.0, 0.0],
        },
        {
            "filepath": "/query/clean.png", "query_id": "clean", "role": "clean",
            "branch": "background_fp", "benchmark": "fixture", "texture": "board",
            "defect_type": "", "generator_type": "", "embedding": [1.0, 0.0],
        },
    ]
    candidate_embeddings = tmp_path / "candidate_embeddings.parquet"
    query_embeddings = tmp_path / "query_embeddings.parquet"
    pd.DataFrame(candidate_rows).to_parquet(candidate_embeddings, index=False)
    pd.DataFrame(query_rows).to_parquet(query_embeddings, index=False)
    query_manifest = tmp_path / "query_manifest.json"
    query_manifest.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "query_counts": {"strict_fn": 1, "near_miss_fp": 1, "background_fp": 1},
    }))
    output = tmp_path / "route"
    report = route(argparse.Namespace(
        policy=str(policy_path), iteration=1,
        candidate_embeddings=str(candidate_embeddings),
        query_embeddings=str(query_embeddings), query_manifest=str(query_manifest),
        kpi_coco=str(kpi_coco), source_coco=str(source_coco),
        source_images_dir=str(source_dir), clean_coco=str(clean_coco),
        clean_images_dir=str(clean_dir), output_dir=str(output),
        previous_defect_ledger=None, previous_clean_ledger=None,
        previous_admission_index=None, conversion_old_strict=None,
        conversion_new_strict=None, prior_admitted_synthetic=0,
        valid_generator_types=None,
    ))

    assert report["adaptive_factors"] == {"fixture/board/bad": 3}
    assert report["selected"] == {
        "strict_fn_real": 3, "near_miss_real": 2,
        "uniform_real": 0, "background_clean": 5,
    }
    assert report["cumulative_real_defectives"] == 5
    assert report["cumulative_clean_negatives"] == 5
    preview = json.loads((output / "admission_preview.json").read_text())
    assert preview["status"] == "COMPLETE"
    assert preview["controller"]["near_miss_real_factor"] == 2
    assert preview["controller"]["near_miss_real_cap_per_pocket"] == 20
    assert preview["controller"]["clean_factor"] == 5
    assert preview["counts"]["manifest_records"] == 10


def test_admission_screens_boxes_duplicates_and_clean_clusters(tmp_path: Path) -> None:
    pattern = np.indices((32, 32)).sum(axis=0) % 2 * 255
    rgb = np.repeat(pattern[:, :, None], 3, axis=2).astype(np.uint8)
    paths = []
    for index in range(5):
        path = tmp_path / f"same_{index}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)

    policy = build_policy(max_iterations=1, synthetic_enabled=False)
    admission = Admission(policy, None)
    selected = admission.admit([
        {"source_path": str(paths[0]), "boxes": [[0, 0, 1, 1]]},
        {"source_path": str(paths[1]), "boxes": [[4, 4, 12, 12]]},
        {"source_path": str(paths[2]), "boxes": [[4, 4, 12, 12]]},
    ], 3, clean=False)
    assert len(selected) == 1
    assert admission.report["boxes_quarantined"] == 1
    assert admission.report["rejected_no_valid_boxes"] == 1
    assert admission.report["rejected_duplicate"] == 1

    clean_policy = build_policy(max_iterations=1, synthetic_enabled=False)
    clean_policy["admission"]["clean_duplicate_global_cosine"] = 1.1
    clean = Admission(clean_policy, None)
    selected_clean = clean.admit(
        [{"source_path": str(path), "boxes": []} for path in paths[:3]],
        8, clean=True,
    )
    assert len(selected_clean) == 2
    assert clean.report["rejected_cluster_cap"] == 1
