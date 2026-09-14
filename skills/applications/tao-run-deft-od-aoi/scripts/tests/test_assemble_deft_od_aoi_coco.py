# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "assemble_deft_od_aoi_coco.py"
SPEC = importlib.util.spec_from_file_location("assemble_deft_od_aoi_coco", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_assembly_enforces_deterministic_cumulative_synthetic_cap(tmp_path: Path) -> None:
    real_paths = []
    records = []
    for index in range(4):
        image = tmp_path / f"real-{index}.png"
        image.write_bytes(f"real-{index}".encode())
        real_paths.append(image)
        records.append({
            "source_path": str(image), "width": 16, "height": 16,
            "boxes": [[1, 1, 4, 4]], "kind": "real_defect",
        })
    route = tmp_path / "route.json"
    route.write_text(json.dumps(records))
    routing_report = tmp_path / "routing-report.json"
    routing_report.write_text(json.dumps({
        "cumulative_real_defectives": 4,
        "cumulative_clean_negatives": 0,
    }))
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "synthesis": {"cumulative_fraction_of_real_defects": 0.25},
        "admission": {"minimum_box_area_px": 64, "maximum_box_aspect": 25.0},
    }))

    generated = tmp_path / "generated"
    generated.mkdir()
    synthetic_images = []
    synthetic_annotations = []
    for index, name in enumerate(("z.png", "a.png", "m.png"), start=1):
        image = generated / name
        image.write_bytes(name.encode())
        synthetic_images.append({
            "id": index, "file_name": name, "width": 16, "height": 16,
        })
        synthetic_annotations.append({
            "id": index, "image_id": index, "category_id": 7,
            "bbox": [2, 2, 8, 8],
        })
    synthetic_coco = tmp_path / "synthetic.json"
    synthetic_coco.write_text(json.dumps({
        "images": synthetic_images,
        "annotations": synthetic_annotations,
        "categories": [{"id": 7, "name": "fine-grained-defect"}],
    }))

    output_coco = tmp_path / "out" / "train.json"
    result = MODULE.run(Namespace(
        policy=str(policy), previous_assembled_coco=None,
        route_manifest=[str(route)], current_routing_report=str(routing_report),
        synthetic_source=[f"{synthetic_coco}::{generated}"],
        output_coco=str(output_coco), output_images_dir=str(tmp_path / "out" / "images"),
        link_mode="symlink",
    ))

    assert result["by_kind"] == {"real_defect": 4, "synthetic_defect": 1}
    assert result["synthetic_admission"]["admitted_new"] == 1
    assert result["synthetic_admission"]["excluded_by_cap"] == 2
    assert result["synthetic_admission"]["quality_filter"]["eligible_images"] == 3
    output = json.loads(output_coco.read_text())
    admitted = [row for row in output["images"] if row["deft_od_aoi_kind"] == "synthetic_defect"]
    assert len(admitted) == 1
    expected = min(
        [(generated / name).resolve() for name in ("z.png", "a.png", "m.png")],
        key=lambda path: hashlib.sha256(str(path).encode()).hexdigest(),
    )
    assert admitted[0]["source_path"] == str(expected)


def test_synthetic_quality_and_stratified_cap_prevent_noisy_or_biased_admission(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(8):
        image = tmp_path / f"real-{index}.png"
        image.write_bytes(b"real")
        records.append({"source_path": str(image), "width": 20, "height": 20,
                        "boxes": [[1, 1, 8, 8]], "kind": "real_defect"})
    route = tmp_path / "route.json"
    route.write_text(json.dumps(records))
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({"cumulative_real_defectives": 8,
                                   "cumulative_clean_negatives": 0}))
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "synthesis": {"cumulative_fraction_of_real_defects": 0.5},
        "admission": {"minimum_box_area_px": 64, "maximum_box_aspect": 25.0},
    }))
    generated = tmp_path / "generated"
    generated.mkdir()
    images, annotations = [], []
    for index in range(8):
        name = f"{'z' if index < 4 else 'a'}-{index}.png"
        (generated / name).write_bytes(b"synthetic")
        images.append({"id": index + 1, "file_name": name, "width": 20, "height": 20,
                       "dataset_id": "group-a" if index < 4 else "group-b"})
        bbox = ([0, 0, 20, 20] if index == 0 else
                [1, 1, 1, 1] if index == 4 else [2, 2, 8, 8])
        annotations.append({"id": index + 1, "image_id": index + 1,
                            "category_id": 1, "bbox": bbox})
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps({"images": images, "annotations": annotations,
                                     "categories": [{"id": 1, "name": "defect"}]}))

    output = tmp_path / "out/train.json"
    result = MODULE.run(Namespace(
        policy=str(policy), previous_assembled_coco=None, route_manifest=[str(route)],
        current_routing_report=str(routing),
        synthetic_source=[f"{synthetic}::{generated}"], output_coco=str(output),
        output_images_dir=str(tmp_path / "out/images"), link_mode="symlink",
    ))

    admission = result["synthetic_admission"]
    assert admission["quality_filter"]["rejected_annotations_full_frame"] == 1
    assert admission["quality_filter"]["rejected_annotations_small"] == 1
    assert admission["requested_new"] == 6
    assert admission["admitted_by_stratum"] == {"group-a": 2, "group-b": 2}
