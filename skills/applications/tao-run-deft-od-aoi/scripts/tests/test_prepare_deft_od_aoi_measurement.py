# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_measurement.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_measurement", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_measurement_freezes_binary_inference_and_dual_gap_specs(tmp_path: Path) -> None:
    sources = {}
    for role in ("kpi", "test"):
        images = tmp_path / role
        images.mkdir()
        (images / f"{role}.png").write_bytes(b"image")
        coco = tmp_path / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": f"{role}.png"}],
                                    "annotations": [{"id": 1, "image_id": 1,
                                                     "category_id": 1, "bbox": [1, 2, 3, 4]}],
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(images), "coco": str(coco)}
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"base_checkpoint": str(checkpoint),
                                      "routing_seed_checkpoint": str(checkpoint),
                                      "sources": sources,
                                      "gap": {"inference_confidence": 0.001,
                                              "loose_confidence": 0.3,
                                              "strict_confidence": 0.8,
                                              "match_iou": 0.5}}))
    report = MODULE.prepare(policy, checkpoint, tmp_path / "measure/kpi/inference/labels",
                            tmp_path / "measure", tmp_path / "specs")
    assert report["status"] == "COMPLETE"
    assert report["checkpoint_provenance"]["role"] == "routing_seed"
    inference = yaml.safe_load((tmp_path / "specs/kpi_inference.yaml").read_text())
    assert inference["dataset"]["num_classes"] == 2
    assert Path(inference["dataset"]["infer_data_sources"]["classmap"]).read_text() == (
        "background\ndefect\n"
    )
    loose = yaml.safe_load((tmp_path / "specs/gap_loose.yaml").read_text())
    strict = yaml.safe_load((tmp_path / "specs/gap_strict.yaml").read_text())
    assert loose["conf_threshold"] == 0.3 and strict["conf_threshold"] == 0.8
    label = tmp_path / "specs/kpi_ground_truth_kitti/kpi.txt"
    assert label.read_text().startswith("defect 0.0 0 0.0 1.000000 2.000000 4.000000 6.000000")


def test_yolo_measurement_emits_common_kitti_and_gap_specs_only(tmp_path: Path) -> None:
    sources = {}
    for role in ("kpi", "test"):
        images = tmp_path / role
        images.mkdir()
        (images / f"{role}.png").write_bytes(b"image")
        coco = tmp_path / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": f"{role}.png"}],
                                    "annotations": [{"id": 1, "image_id": 1,
                                                     "category_id": 1, "bbox": [1, 2, 3, 4]}],
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(images), "coco": str(coco)}
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"model": {"backend": "yolo"},
                                      "base_checkpoint": str(checkpoint),
                                      "routing_seed_checkpoint": str(checkpoint),
                                      "sources": sources,
                                      "gap": {"inference_confidence": 0.001,
                                              "loose_confidence": 0.3,
                                              "strict_confidence": 0.8,
                                              "match_iou": 0.5}}))
    report = MODULE.prepare(policy, checkpoint, tmp_path / "measure/kpi/labels",
                            tmp_path / "measure", tmp_path / "specs")

    assert report["status"] == "COMPLETE"
    assert set(report["specs"]) == {"gap_loose.yaml", "gap_strict.yaml"}
    assert not (tmp_path / "specs/kpi_inference.yaml").exists()
    assert not (tmp_path / "specs/test_inference.yaml").exists()
    assert not (tmp_path / "specs/inference_classmap.txt").exists()
    assert (tmp_path / "specs/kpi_ground_truth_kitti/kpi.txt").read_text().startswith(
        "defect 0.0 0 0.0 1.000000 2.000000 4.000000 6.000000"
    )
    loose = yaml.safe_load((tmp_path / "specs/gap_loose.yaml").read_text())
    strict = yaml.safe_load((tmp_path / "specs/gap_strict.yaml").read_text())
    assert loose["inference_ann_path"] == str(tmp_path / "measure/kpi/labels")
    assert loose["conf_threshold"] == 0.3
    assert strict["conf_threshold"] == 0.8

    selected = tmp_path / "iteration1-selected.pt"
    selected.write_bytes(b"selected")
    followup = MODULE.prepare(
        policy, selected, tmp_path / "measure2/kpi/labels", tmp_path / "measure2",
        tmp_path / "specs2", "iteration_selected", 1,
    )
    assert followup["checkpoint_provenance"]["role"] == "iteration_selected"
    assert followup["checkpoint_provenance"]["source_iteration"] == 1


def test_measurement_binds_specs_to_durable_copyback_path(tmp_path: Path) -> None:
    sources = {}
    for role in ("kpi", "test"):
        images = tmp_path / role
        images.mkdir()
        (images / f"{role}.png").write_bytes(b"image")
        coco = tmp_path / f"{role}.json"
        coco.write_text(json.dumps({
            "images": [{"id": 1, "file_name": f"{role}.png"}],
            "annotations": [],
            "categories": [{"id": 1, "name": "defect"}],
        }))
        sources[role] = {"images": str(images), "coco": str(coco)}
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "model": {"backend": "yolo"},
        "base_checkpoint": str(checkpoint),
        "routing_seed_checkpoint": str(checkpoint),
        "sources": sources,
        "gap": {"inference_confidence": 0.001, "loose_confidence": 0.3,
                "strict_confidence": 0.8, "match_iou": 0.5},
    }))
    scratch = tmp_path / "scratch_specs"
    durable = tmp_path / "durable_specs"
    report = MODULE.prepare(
        policy, checkpoint, tmp_path / "measure/kpi/labels", tmp_path / "measure",
        scratch, "routing_seed", 0, durable,
    )
    loose = yaml.safe_load((scratch / "gap_loose.yaml").read_text())
    assert loose["ground_truth_ann_path"] == str(
        durable.resolve() / "kpi_ground_truth_kitti"
    )
    assert report["specs"]["gap_loose.yaml"] == str(
        durable.resolve() / "gap_loose.yaml"
    )
