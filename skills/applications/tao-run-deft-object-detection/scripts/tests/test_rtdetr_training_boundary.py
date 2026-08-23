#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from audit_deft_run import audit  # noqa: E402
from project_coco_classes import project  # noqa: E402
from stage_mined_coco import stage  # noqa: E402


class RtdetrTrainingBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def coco() -> dict:
        return {
            "info": {"dataset": "fixture"},
            "images": [{"id": 10, "file_name": "a.jpg", "lot": "L1"}],
            "categories": [{"id": 1, "name": "scratch"}, {"id": 2, "name": "dent"}],
            "annotations": [
                {"id": 20, "image_id": 10, "category_id": 1, "bbox": [1, 2, 3, 4], "severity": 4},
                {"id": 21, "image_id": 10, "category_id": 2, "bbox": [5, 6, 7, 8], "defect_type": "deep_dent"},
            ],
        }

    def test_aoi_projection_preserves_per_box_provenance(self) -> None:
        projected, sidecar = project(self.coco(), "line7-v3", 1, "defect")
        self.assertEqual(projected["categories"], [{"id": 1, "name": "defect", "supercategory": "defect"}])
        self.assertEqual([row["category_id"] for row in projected["annotations"]], [1, 1])
        self.assertEqual(projected["annotations"][0]["severity"], 4)
        self.assertEqual(projected["annotations"][0]["deft"]["defect_type"], "scratch")
        self.assertEqual(projected["annotations"][1]["deft"]["defect_type"], "deep_dent")
        self.assertEqual(sidecar[0]["original_category_name"], "scratch")
        self.assertNotEqual(sidecar[0]["annotation_uid"], sidecar[1]["annotation_uid"])

    def test_aoi_cli_emits_kpi_aliases_for_original_ground_truth(self) -> None:
        source = self.root / "source.json"
        source.write_text(json.dumps(self.coco()), encoding="utf-8")
        mapping = self.root / "mapping.yaml"
        command = [
            sys.executable, str(SCRIPTS / "project_coco_classes.py"),
            "--input-coco", str(source),
            "--output-coco", str(self.root / "projected.json"),
            "--metadata-jsonl", str(self.root / "metadata.jsonl"),
            "--classmap-out", str(self.root / "classmap.txt"),
            "--kpi-mapping-out", str(mapping),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            yaml.safe_load(mapping.read_text()),
            [{"defect": ["defect", "scratch", "dent"]}],
        )

    def test_stage_coco_preserves_categories_and_custom_metadata(self) -> None:
        image = self.root / "a.jpg"
        image.write_bytes(b"jpeg")
        source = self.root / "source.json"
        source.write_text(json.dumps(self.coco()), encoding="utf-8")
        mined = self.root / "mined.parquet"
        pd.DataFrame({"filepath": [str(image)]}).to_parquet(mined)
        args = argparse.Namespace(
            mined_parquet=str(mined),
            source_coco=str(source),
            output_images_dir=str(self.root / "images"),
            output_coco=str(self.root / "staged.json"),
            output_classmap=str(self.root / "classmap.txt"),
            report_json=None,
            min_success_rate=0.0,
        )
        report = stage(args)
        staged = json.loads((self.root / "staged.json").read_text())
        self.assertEqual(report["class_names"], ["scratch", "dent"])
        self.assertEqual(staged["annotations"][0]["severity"], 4)
        self.assertEqual((self.root / "classmap.txt").read_text(), "scratch\ndent\n")

    def test_prepare_spec_sets_rtdetr_class_contract(self) -> None:
        image_dir = self.root / "images"
        image_dir.mkdir()
        (image_dir / "a.jpg").write_bytes(b"jpeg")
        coco = self.root / "train.json"
        coco.write_text(json.dumps(self.coco()), encoding="utf-8")
        checkpoint = self.root / "base.pth"
        checkpoint.write_bytes(b"weights")
        previous = self.root / "previous.yaml"
        previous.write_text("dataset:\n  train_data_sources: []\ntrain:\n  checkpoint_interval: 5\n  validation_interval: 5\n", encoding="utf-8")
        output = self.root / "train.yaml"
        command = [
            sys.executable, str(SCRIPTS / "prepare_rtdetr_spec_for_train.py"),
            "--previous-spec", str(previous), "--output-spec", str(output),
            "--tmm-image-dir", str(image_dir), "--tmm-coco-file", str(coco),
            "--val-image-dir", str(image_dir), "--val-json-file", str(coco),
            "--pretrained-model-path", str(checkpoint), "--num-epochs", "2",
            "--learning-rate", "0.0002", "--num-gpus", "2",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        spec = yaml.safe_load(output.read_text())
        self.assertEqual(spec["dataset"]["num_classes"], 3)
        self.assertEqual(spec["dataset"]["eval_class_ids"], [1, 2])
        self.assertEqual(spec["train"]["gpu_ids"], [0, 1])
        self.assertEqual(spec["train"]["checkpoint_interval"], 2)
        self.assertEqual(spec["train"]["pretrained_model_path"], str(checkpoint.resolve()))

    def test_init_freezes_detector_and_classmap(self) -> None:
        workspace = self.root / "workspace"
        workspace.mkdir()
        results = workspace / "results"
        checkpoint = workspace / "base.pth"
        checkpoint.write_bytes(b"weights")
        pool_annotations = workspace / "odvg"
        pool_annotations.mkdir()
        (pool_annotations / "pool.jsonl").write_text("{}\n", encoding="utf-8")
        pool_embeddings = workspace / "pool.parquet"
        pd.DataFrame({"filepath": ["a.jpg"], "embedding": [[0.0, 1.0]]}).to_parquet(
            pool_embeddings
        )
        source_coco = workspace / "source.json"
        source_coco.write_text(json.dumps(self.coco()), encoding="utf-8")
        kpi_images = workspace / "kpi_images"
        kpi_images.mkdir()
        (kpi_images / "a.jpg").write_bytes(b"jpeg")
        labels = workspace / "labels"
        labels.mkdir()
        (labels / "a.txt").write_text("scratch 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n")
        mapping = workspace / "mapping.yaml"
        mapping.write_text("- scratch:\n  - scratch\n- dent:\n  - dent\n", encoding="utf-8")
        encoder = workspace / "encoder"
        encoder.mkdir()
        (encoder / "config.json").write_text("{}", encoding="utf-8")
        command = [
            sys.executable, str(SCRIPTS / "init_deft_state.py"),
            "--results-dir", str(results), "--workspace", str(workspace),
            "--detector", "rtdetr", "--num-epochs", "2", "--learning-rate", "0.0001",
            "--zero-shot-checkpoint", str(checkpoint),
            "--source-pool-embeddings", str(pool_embeddings),
            "--source-pool-annotations", str(pool_annotations),
            "--source-detection-file", str(source_coco),
            "--embedding-model-path", str(encoder),
            "--kpi-images-dir", str(kpi_images),
            "--ground-truth-labels-dir", str(labels),
            "--class-mapping", str(mapping),
            "--target-classes", "scratch,dent",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        config = json.loads((results / "deft_state.json").read_text())["config"]
        self.assertEqual(config["detector"], "rtdetr")
        self.assertEqual(config["training_annotation_format"], "coco")
        self.assertEqual(config["rtdetr_category_ids"], [1, 2])
        self.assertEqual(config["rtdetr_num_classes"], 3)
        self.assertEqual(Path(config["inference_classmap"]).read_text(), "scratch\ndent\n")

    def test_audit_routes_training_to_rtdetr_reference(self) -> None:
        results = self.root / "results"
        results.mkdir()
        pool_annotations = self.root / "odvg"
        pool_annotations.mkdir()
        (pool_annotations / "pool.jsonl").write_text("{}\n", encoding="utf-8")
        pool_embeddings = self.root / "pool.parquet"
        pd.DataFrame({"filepath": ["a.jpg"], "embedding": [[0.0, 1.0]]}).to_parquet(
            pool_embeddings
        )
        classmap = results / "rtdetr_classmap.txt"
        classmap.write_text("scratch\ndent\n", encoding="utf-8")
        state = {
            "schema_version": 1,
            "results_dir": str(results),
            "current_iteration": 0,
            "status": "running",
            "iterations": {},
            "config": {
                "max_iterations": 1,
                "detector": "rtdetr",
                "source_pool_annotations": str(pool_annotations),
                "source_pool_embeddings": str(pool_embeddings),
                "inference_classmap": str(classmap),
                "rtdetr_category_ids": [1, 2],
                "rtdetr_num_classes": 3,
                "rtdetr_class_names": ["scratch", "dent"],
            },
        }
        (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
        (results / "loop_log.jsonl").write_text("", encoding="utf-8")
        report = audit(results)
        self.assertEqual(report["status"], "VALID", report["errors"])
        self.assertEqual(report["next_action"], "inference")
        self.assertEqual(report["read_before_action"], "references/rtdetr.md")

    def test_audit_keeps_legacy_grounding_dino_default(self) -> None:
        results = self.root / "legacy_results"
        results.mkdir()
        pool_annotations = self.root / "legacy_odvg"
        pool_annotations.mkdir()
        (pool_annotations / "pool.jsonl").write_text("{}\n", encoding="utf-8")
        pool_embeddings = self.root / "legacy_pool.parquet"
        pd.DataFrame({"filepath": ["a.jpg"], "embedding": [[0.0, 1.0]]}).to_parquet(
            pool_embeddings
        )
        state = {
            "schema_version": 1,
            "results_dir": str(results),
            "current_iteration": 0,
            "status": "running",
            "iterations": {},
            "config": {
                "max_iterations": 1,
                "source_pool_annotations": str(pool_annotations),
                "source_pool_embeddings": str(pool_embeddings),
            },
        }
        (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
        (results / "loop_log.jsonl").write_text("", encoding="utf-8")
        report = audit(results)
        self.assertEqual(report["status"], "VALID", report["errors"])
        self.assertEqual(report["read_before_action"], "references/grounding-dino.md")


if __name__ == "__main__":
    unittest.main()
