#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the optional synthetic-data training handoff."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_script = load_module("stage_anomalygen_coco", SCRIPTS / "stage_anomalygen_coco.py")
train_script = load_module("update_train_spec", SCRIPTS / "update_train_spec.py")
audit_script = load_module("audit_deft_run_handoff", SCRIPTS / "audit_deft_run.py")


class SyntheticTrainingHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.generated = self.root / "generated.png"
        Image.new("RGB", (32, 24), color=(20, 40, 60)).save(self.generated)
        self.source_coco = self.root / "coco_annotations_od_defect.json"
        self.source_coco.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 7, "file_name": str(self.generated), "width": 32, "height": 24}
                    ],
                    "annotations": [
                        {"id": 1, "image_id": 7, "category_id": 1, "bbox": [2, 3, 8, 9]}
                    ],
                    "categories": [{"id": 1, "name": "defect"}],
                }
            )
        )
        self.summary = self.root / "validation_summary.json"
        self._write_summary("test_fixture", training_eligible=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_summary(self, source_tag: str, *, training_eligible: bool) -> None:
        self.summary.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "source_tag": source_tag,
                    "training_eligible": training_eligible,
                    "od_coco": str(self.source_coco),
                    "training_pool_mutated": False,
                }
            )
        )

    def _stage(self) -> dict:
        return stage_script.stage(
            argparse.Namespace(
                validation_summary=str(self.summary),
                source_coco=str(self.source_coco),
                output_images_dir=str(self.root / "training" / "images"),
                output_coco=str(self.root / "training" / "synthetic_train.json"),
                target_class="defect",
                report_json=str(self.root / "training" / "staging_report.json"),
            )
        )

    def test_training_eligible_generation_is_staged_with_explicit_mutation_record(self) -> None:
        report = self._stage()
        self.assertTrue(report["training_pool_mutated"])
        staged = json.loads(Path(report["output_coco"]).read_text())
        self.assertEqual(staged["categories"], [{"id": 1, "name": "defect"}])
        self.assertEqual(staged["images"][0]["file_name"], "synthetic_00000007.png")
        self.assertTrue((Path(report["output_images_dir"]) / "synthetic_00000007.png").is_file())

    def test_ineligible_generation_cannot_cross_the_training_boundary(self) -> None:
        self._write_summary("evaluation_data", training_eligible=False)
        with self.assertRaisesRegex(ValueError, "training_eligible"):
            self._stage()

    def _source(self, phase: str, producer: str) -> tuple[Path, Path, Path]:
        root = self.root / phase / producer
        images = root / "images"
        images.mkdir(parents=True)
        odvg = root / "annotations.jsonl"
        odvg.write_text('{"file_name":"x.png","detection":{"instances":[]}}\n')
        labelmap = root / "labelmap.json"
        labelmap.write_text('{"0":"defect"}\n')
        return images, odvg, labelmap

    def _update(self, previous: Path, output: Path, phase: str) -> None:
        mined = self._source(phase, "mined")
        synthetic = self._source(phase, "synthetic")
        checkpoint = self.root / "base.pth"
        checkpoint.write_bytes(b"checkpoint")
        val_images = self.root / "val-images"
        val_images.mkdir(exist_ok=True)
        val_json = self.root / "val.json"
        val_json.write_text('{"images":[],"annotations":[],"categories":[]}')
        argv = [
            "update_train_spec.py",
            "--previous-spec", str(previous),
            "--output-spec", str(output),
            "--tmm-image-dir", str(mined[0]),
            "--tmm-odvg-file", str(mined[1]),
            "--tmm-label-map-file", str(mined[2]),
            "--synthetic-image-dir", str(synthetic[0]),
            "--synthetic-odvg-file", str(synthetic[1]),
            "--synthetic-label-map-file", str(synthetic[2]),
            "--pretrained-model-path", str(checkpoint),
            "--val-image-dir", str(val_images),
            "--val-json-file", str(val_json),
            "--num-epochs", "1",
            "--learning-rate", "0.0001",
        ]
        with patch.object(sys, "argv", argv):
            self.assertEqual(train_script.main(), 0)

    def test_two_iterations_accumulate_both_producers(self) -> None:
        template = self.root / "template.yaml"
        template.write_text(yaml.safe_dump({"dataset": {"train_data_sources": []}, "train": {}}))
        iter1 = self.root / "iter1.yaml"
        iter2 = self.root / "iter2.yaml"
        self._update(template, iter1, "iter1")
        self._update(iter1, iter2, "iter2")
        sources = yaml.safe_load(iter2.read_text())["dataset"]["train_data_sources"]
        self.assertEqual(len(sources), 4)
        self.assertEqual(
            [Path(row["image_dir"]).parts[-2:] for row in sources],
            [("mined", "images"), ("synthetic", "images"),
             ("mined", "images"), ("synthetic", "images")],
        )

    def test_init_preflight_freezes_complete_generic_anomalygen_config(self) -> None:
        workspace = self.root / "workspace"
        workspace.mkdir()
        checkpoint = workspace / "base.pth"
        checkpoint.write_bytes(b"checkpoint")
        train_template = workspace / "train.yaml"
        train_template.write_text("dataset:\n  train_data_sources: []\ntrain: {}\n")
        pool_embeddings = workspace / "pool.parquet"
        pool_embeddings.write_bytes(b"parquet-placeholder")
        pool_annotations = workspace / "pool-odvg"
        pool_annotations.mkdir()
        kpi_images = workspace / "kpi-images"
        kpi_images.mkdir()
        gt = workspace / "gt"
        gt.mkdir()
        (gt / "sample.txt").write_text("")
        class_mapping = workspace / "classes.yaml"
        class_mapping.write_text("mapping: {}\n")
        encoder = workspace / "siglip"
        encoder.mkdir()
        (encoder / "config.json").write_text("{}")

        split_root = workspace / "splits"
        split_root.mkdir()
        ag_pool = workspace / "ag-pool"
        ag_pool.mkdir()
        defect_spec = workspace / "defect_spec.jsonl"
        defect_spec.write_text("{}\n")
        ag_repo = workspace / "anomalygen-next"
        ag_repo.mkdir()
        base_checkpoint = workspace / "cosmos-base"
        base_checkpoint.mkdir()
        adapter = workspace / "adapter.pt"
        adapter.write_bytes(b"adapter")
        recipe = workspace / "recipe.yaml"
        recipe.write_text("anomaly_types: []\n")
        ag_config = workspace / "anomalygen.yaml"
        ag_config.write_text(
            yaml.safe_dump(
                {
                    "source_tag": "test_fixture",
                    "training_eligible": True,
                    "gap_parquet": "/replaced/per/iteration.parquet",
                    "split_root": str(split_root),
                    "pool_dataset_root": str(ag_pool),
                    "defect_spec": str(defect_spec),
                    "datasets": {
                        "example": {
                            "checkpoint": str(adapter),
                            "recipe": str(recipe),
                        }
                    },
                }
            )
        )

        results = workspace / "results" / "dry-run"
        command = [
            str(SCRIPTS.parents[3] / ".venv" / "deft" / "bin" / "python"),
            str(SCRIPTS / "init_deft_state.py"),
            "--workspace", str(workspace),
            "--results-dir", str(results),
            "--max-iterations", "2",
            "--num-gpus", "2",
            "--num-epochs", "1",
            "--learning-rate", "0.0001",
            "--zero-shot-checkpoint", str(checkpoint),
            "--train-spec-template", str(train_template),
            "--source-pool-embeddings", str(pool_embeddings),
            "--source-pool-annotations", str(pool_annotations),
            "--embedding-model-path", str(encoder),
            "--kpi-images-dir", str(kpi_images),
            "--ground-truth-labels-dir", str(gt),
            "--class-mapping", str(class_mapping),
            "--target-classes", "defect",
            "--ap50-thresholds-json", '{"defect": 0.7}',
            "--anomalygen-config-template", str(ag_config),
            "--anomalygen-repo", str(ag_repo),
            "--anomalygen-base-checkpoint", str(base_checkpoint),
            "--anomalygen-target-class", "defect",
            "--anomalygen-num-gpus", "2",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        config = json.loads((results / "deft_state.json").read_text())["config"]
        self.assertTrue(config["anomalygen_enabled"])
        self.assertEqual(config["anomalygen_target_class"], "defect")
        self.assertEqual(config["anomalygen_num_gpus"], 2)

    def test_audit_rejects_enabled_stage_without_synthetic_handoff(self) -> None:
        results = self.root / "audit-run"
        results.mkdir()

        def file(relative: str, body: str = "fixture") -> str:
            path = results / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            return str(path)

        def directory(relative: str) -> str:
            path = results / relative
            path.mkdir(parents=True, exist_ok=True)
            return str(path)

        pool_annotations = directory("pool/odvg")
        pool_embeddings = file("pool/embeddings.parquet")
        baseline_labels = directory("baseline/inference/labels")
        baseline_kpi = file("baseline/kpi/kpi.csv")
        baseline_log = file("baseline/kpi/kpi.log", "mAP: 0.2\n")
        iter1 = {
            "stage_completed": "stage",
            "weak_images_parquet": file("iter1/gaps/weak.parquet"),
            "gap_report_json": file("iter1/gaps/report.json", "{}"),
            "weak_image_count": 1,
            "embeddings_parquet": file("iter1/embed/embeddings.parquet"),
            "mining_output_parquet": file("iter1/mining/out.parquet"),
            "mining_summary_json": file("iter1/mining/summary.json", "{}"),
            "odvg_jsonl": file("iter1/tmm/annotations/data.jsonl", "{}\n"),
            "label_map_json": file("iter1/tmm/annotations/labelmap.json", "{}"),
            "staged_images_dir": directory("iter1/tmm/images"),
            "exclude_parquet": file("iter1/mined_cumulative.parquet"),
        }
        state = {
            "schema_version": 1,
            "workspace": str(self.root),
            "results_dir": str(results),
            "config": {
                "max_iterations": 2,
                "source_pool_annotations": pool_annotations,
                "source_pool_embeddings": pool_embeddings,
                "anomalygen_enabled": True,
            },
            "current_iteration": 1,
            "iterations": {
                "baseline": {
                    "stage_completed": "kpi_analyze",
                    "inference_labels_dir": baseline_labels,
                    "kpi_csv": baseline_kpi,
                    "kpi_log": baseline_log,
                    "map_value": 0.2,
                },
                "iter1": iter1,
            },
            "status": "running",
        }
        (results / "deft_state.json").write_text(json.dumps(state))
        stages = [
            ("baseline", "inference"),
            ("baseline", "kpi_analyze"),
            ("iter1", "gap_analysis"),
            ("iter1", "embed"),
            ("iter1", "mine"),
            ("iter1", "stage"),
        ]
        events = [
            {
                "seq": index,
                "ts": "2026-08-12T00:00:00Z",
                "iter": phase,
                "stage": stage,
                "status": "ok",
                "summary": f"{phase}/{stage}",
                "duration_sec": 1,
                "context_tokens": 0,
            }
            for index, (phase, stage) in enumerate(stages, 1)
        ]
        (results / "loop_log.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )

        report = audit_script.audit(results)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("did not record synthetic artifacts" in e for e in report["errors"]))

        generation = file(
            "iter1/synthetic/generation/validation_summary.json",
            json.dumps(
                {"status": "COMPLETE", "source_tag": "test_fixture",
                 "training_eligible": True,
                 "training_pool_mutated": False}
            ),
        )
        admission = file(
            "iter1/synthetic/training/staging_report.json",
            json.dumps(
                {"status": "COMPLETE", "source_tag": "test_fixture",
                 "training_eligible": True,
                 "training_pool_mutated": True}
            ),
        )
        iter1.update(
            {
                "synthetic_validation_summary": generation,
                "synthetic_coco": file("iter1/synthetic/training/data.json", "{}"),
                "synthetic_odvg": file("iter1/synthetic/training/data.jsonl", "{}\n"),
                "synthetic_label_map": file("iter1/synthetic/training/labelmap.json", "{}"),
                "synthetic_images_dir": directory("iter1/synthetic/training/images"),
                "synthetic_staging_report": admission,
            }
        )
        (results / "deft_state.json").write_text(json.dumps(state))
        report = audit_script.audit(results)
        self.assertEqual(report["status"], "VALID", report["errors"])

        iter1["stage_completed"] = "train"
        iter1["checkpoint_path"] = file("iter1/train/model.pth")
        iter1["training_spec"] = file("iter1/train.yaml", "dataset: {}\n")
        events.append(
            {
                "seq": 7,
                "ts": "2026-08-12T00:00:00Z",
                "iter": "iter1",
                "stage": "train",
                "status": "ok",
                "summary": "iter1/train",
                "duration_sec": 1,
                "context_tokens": 0,
            }
        )
        (results / "deft_state.json").write_text(json.dumps(state))
        (results / "loop_log.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        report = audit_script.audit(results)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("train spec omits" in e for e in report["errors"]))

        Path(iter1["training_spec"]).write_text(
            "\n".join(
                (
                    iter1["synthetic_images_dir"],
                    iter1["synthetic_odvg"],
                    iter1["synthetic_label_map"],
                )
            )
        )
        report = audit_script.audit(results)
        self.assertEqual(report["status"], "VALID", report["errors"])


if __name__ == "__main__":
    unittest.main()
