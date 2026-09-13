# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        "synthesis": {"cumulative_fraction_of_real_defects": 0.25}
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
            "bbox": [2, 2, 3, 3],
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
    assert result["synthetic_admission"] == {
        "fraction_of_real_defects": 0.25,
        "cumulative_limit": 1,
        "retained_previous": 0,
        "requested_new": 3,
        "admitted_new": 1,
        "excluded_by_cap": 2,
    }
    output = json.loads(output_coco.read_text())
    admitted = [row for row in output["images"] if row["deft_od_aoi_kind"] == "synthetic_defect"]
    assert len(admitted) == 1
    assert admitted[0]["source_path"] == str((generated / "a.png").resolve())
