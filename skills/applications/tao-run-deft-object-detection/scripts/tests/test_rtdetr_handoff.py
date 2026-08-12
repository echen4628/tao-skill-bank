# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage_mined_coco = load_script("stage_mined_coco")
stage_anomalygen_coco = load_script("stage_anomalygen_coco")
resolve_checkpoint = load_script("resolve_rtdetr_checkpoint")


class RTDETRHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool_images = self.root / "pool"
        self.pool_images.mkdir()
        (self.pool_images / "car.jpg").write_bytes(b"car")
        (self.pool_images / "person.jpg").write_bytes(b"person")
        self.source_coco = self.root / "source.json"
        self.source_coco.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 10, "file_name": "car.jpg", "width": 8, "height": 8},
                        {"id": 20, "file_name": "person.jpg", "width": 8, "height": 8},
                    ],
                    "annotations": [
                        {"id": 100, "image_id": 10, "category_id": 1, "bbox": [0, 0, 4, 4]},
                        {"id": 200, "image_id": 20, "category_id": 2, "bbox": [1, 1, 3, 3]},
                    ],
                    "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
                }
            ),
            encoding="utf-8",
        )
        self.mined = self.root / "mined.parquet"
        pd.DataFrame(
            {"filepath": [str(self.pool_images / "person.jpg"), str(self.pool_images / "car.jpg")]}
        ).to_parquet(self.mined)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _stage(self) -> tuple[Path, Path, Path]:
        images = self.root / "staged" / "images"
        coco = self.root / "staged" / "tmm_coco.json"
        classmap = self.root / "staged" / "classmap.txt"
        report = stage_mined_coco.stage(
            argparse.Namespace(
                mined_parquet=str(self.mined),
                source_coco=str(self.source_coco),
                output_images_dir=str(images),
                output_coco=str(coco),
                output_classmap=str(classmap),
                report_json=str(self.root / "staged" / "report.json"),
            )
        )
        self.assertEqual(report["images_staged"], 2)
        self.assertEqual(report["annotations_written"], 2)
        return images, coco, classmap

    def test_stage_preserves_category_contract_and_rewrites_ids(self) -> None:
        images, coco, classmap = self._stage()
        payload = json.loads(coco.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in payload["images"]], [0, 1])
        self.assertEqual([row["id"] for row in payload["annotations"]], [0, 1])
        self.assertEqual([row["category_id"] for row in payload["annotations"]], [2, 1])
        self.assertEqual(classmap.read_text(encoding="utf-8"), "car\nperson\n")
        self.assertEqual(sorted(path.name for path in images.iterdir()), ["car.jpg", "person.jpg"])

    def test_update_train_spec_emits_rtdetr_coco_sources_and_class_shape(self) -> None:
        images, coco, _classmap = self._stage()
        template = self.root / "train.yaml"
        template.write_text(
            yaml.safe_dump(
                {
                    "dataset": {"train_data_sources": [], "num_classes": 80},
                    "train": {"checkpoint_interval": 5, "validation_interval": 5},
                }
            ),
            encoding="utf-8",
        )
        checkpoint = self.root / "base.pth"
        checkpoint.write_bytes(b"checkpoint")
        output = self.root / "iter1.yaml"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "update_train_spec.py"),
                "--detector",
                "rtdetr",
                "--previous-spec",
                str(template),
                "--output-spec",
                str(output),
                "--tmm-image-dir",
                str(images),
                "--tmm-coco-file",
                str(coco),
                "--val-image-dir",
                str(images),
                "--val-json-file",
                str(coco),
                "--pretrained-model-path",
                str(checkpoint),
                "--num-epochs",
                "2",
                "--num-gpus",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        emitted = yaml.safe_load(output.read_text(encoding="utf-8"))
        self.assertEqual(
            emitted["dataset"]["train_data_sources"],
            [{"image_dir": str(images.resolve()), "json_file": str(coco.resolve())}],
        )
        self.assertEqual(emitted["dataset"]["num_classes"], 3)
        self.assertEqual(emitted["dataset"]["eval_class_ids"], [1, 2])
        self.assertEqual(emitted["train"]["gpu_ids"], [0, 1])
        self.assertEqual(emitted["train"]["checkpoint_interval"], 2)

    def test_anomalygen_staging_preserves_full_rtdetr_category_contract(self) -> None:
        generated = self.root / "generated.png"
        generated.write_bytes(b"generated")
        synthetic_source = self.root / "synthetic_source.json"
        synthetic_source.write_text(
            json.dumps(
                {
                    "images": [{"id": 7, "file_name": str(generated)}],
                    "annotations": [
                        {"id": 8, "image_id": 7, "category_id": 99, "bbox": [0, 0, 2, 2]}
                    ],
                    "categories": [{"id": 99, "name": "generated_defect"}],
                }
            ),
            encoding="utf-8",
        )
        summary = self.root / "generation_summary.json"
        summary.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "training_pool_mutated": False,
                    "source_tag": "test",
                    "od_coco": str(synthetic_source),
                }
            ),
            encoding="utf-8",
        )
        output_coco = self.root / "synthetic" / "synthetic_train.json"
        report = stage_anomalygen_coco.stage(
            argparse.Namespace(
                validation_summary=str(summary),
                source_coco=str(synthetic_source),
                output_images_dir=str(self.root / "synthetic" / "images"),
                output_coco=str(output_coco),
                target_class="person",
                category_contract_coco=str(self.source_coco),
                report_json=None,
            )
        )
        staged = json.loads(output_coco.read_text(encoding="utf-8"))
        self.assertEqual(staged["categories"], json.loads(self.source_coco.read_text())["categories"])
        self.assertEqual(staged["annotations"][0]["category_id"], 2)
        self.assertEqual(report["target_category_id"], 2)

    def test_resolve_checkpoint_selects_highest_epoch_and_ema_policy(self) -> None:
        train_dir = self.root / "train"
        train_dir.mkdir()
        for name in ("model_epoch_000.pth", "model_epoch_002.pth", "model_epoch_003-EMA.pth"):
            (train_dir / name).write_bytes(b"checkpoint")
        self.assertEqual(
            resolve_checkpoint.resolve(train_dir, False).name, "model_epoch_002.pth"
        )
        self.assertEqual(
            resolve_checkpoint.resolve(train_dir, True).name, "model_epoch_003-EMA.pth"
        )

    def test_init_freezes_rtdetr_contract_and_audit_routes_overlay(self) -> None:
        annotations = self.root / "odvg"
        annotations.mkdir()
        (annotations / "pool.jsonl").write_text("{}\n", encoding="utf-8")
        embeddings = self.root / "source_embeddings.parquet"
        pd.DataFrame(
            {"filepath": [str(self.pool_images / "car.jpg")], "embedding": [[0.1, 0.2]]}
        ).to_parquet(embeddings)
        kpi_images = self.root / "kpi_images"
        kpi_labels = self.root / "kpi_labels"
        encoder = self.root / "encoder"
        for directory in (kpi_images, kpi_labels, encoder):
            directory.mkdir()
        (kpi_labels / "car.txt").write_text("", encoding="utf-8")
        (encoder / "config.json").write_text("{}", encoding="utf-8")
        mapping = self.root / "mapping.yaml"
        mapping.write_text("- car: car\n- person: person\n", encoding="utf-8")
        template = self.root / "template.yaml"
        template.write_text("dataset:\n  train_data_sources: []\n", encoding="utf-8")
        checkpoint = self.root / "base.pth"
        checkpoint.write_bytes(b"checkpoint")
        results = self.root / "results" / "run_rtdetr"

        init = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "init_deft_state.py"),
                "--workspace",
                str(self.root),
                "--results-dir",
                str(results),
                "--max-iterations",
                "1",
                "--detector",
                "rtdetr",
                "--num-epochs",
                "2",
                "--learning-rate",
                "0.0001",
                "--zero-shot-checkpoint",
                str(checkpoint),
                "--train-spec-template",
                str(template),
                "--source-pool-embeddings",
                str(embeddings),
                "--source-pool-annotations",
                str(annotations),
                "--source-detection-file",
                str(self.source_coco),
                "--embedding-model-path",
                str(encoder),
                "--kpi-images-dir",
                str(kpi_images),
                "--ground-truth-labels-dir",
                str(kpi_labels),
                "--class-mapping",
                str(mapping),
                "--target-classes",
                "car,person",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(init.returncode, 0, init.stderr)
        state = json.loads((results / "deft_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["config"]["detector"], "rtdetr")
        self.assertEqual(state["config"]["rtdetr_category_ids"], [1, 2])
        self.assertEqual(state["config"]["rtdetr_num_classes"], 3)
        self.assertEqual(state["config"]["rtdetr_class_names"], ["car", "person"])
        self.assertEqual(
            (results / "rtdetr_classmap.txt").read_text(encoding="utf-8"),
            "car\nperson\n",
        )

        audit = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "audit_deft_run.py"),
                "--results-dir",
                str(results),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(audit.returncode, 0, audit.stderr)
        report = json.loads(audit.stdout)
        self.assertEqual(report["next_action"], "inference")
        self.assertEqual(report["read_before_action"], "references/rtdetr.md")

        frozen_classmap = results / "rtdetr_classmap.txt"
        frozen_classmap.write_text("person\ncar\n", encoding="utf-8")
        corrupted = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "audit_deft_run.py"),
                "--results-dir",
                str(results),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(corrupted.returncode, 1)
        self.assertIn("contents differ", corrupted.stdout)
        frozen_classmap.write_text("car\nperson\n", encoding="utf-8")

        def commit(phase: str, stage: str, *extra: str) -> None:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "commit_stage.py"),
                    "--results-dir",
                    str(results),
                    "--iter-label",
                    phase,
                    "--stage",
                    stage,
                    "--summary",
                    f"test {phase}/{stage}",
                    *extra,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        baseline_labels = results / "baseline" / "inference" / "labels"
        baseline_labels.mkdir(parents=True)
        baseline_kpi = results / "baseline" / "kpi.csv"
        baseline_kpi.write_text("mAP\n0.1\n", encoding="utf-8")
        commit("baseline", "inference", "--inference-labels-dir", str(baseline_labels))
        commit("baseline", "kpi_analyze", "--kpi-csv", str(baseline_kpi))

        iter_dir = results / "iter1"
        weak = iter_dir / "gaps" / "weak.parquet"
        weak.parent.mkdir(parents=True)
        pd.DataFrame({"filepath": [str(self.pool_images / "car.jpg")]}).to_parquet(weak)
        gap_report = iter_dir / "gaps" / "report.json"
        gap_report.write_text("{}", encoding="utf-8")
        commit(
            "iter1",
            "gap_analysis",
            "--weak-images",
            str(weak),
            "--gap-report",
            str(gap_report),
            "--weak-image-count",
            "1",
        )
        embedded = iter_dir / "embeddings.parquet"
        pd.DataFrame({"filepath": ["car.jpg"], "embedding": [[0.1]]}).to_parquet(embedded)
        commit("iter1", "embed", "--embeddings-parquet", str(embedded))
        mined = iter_dir / "mined.parquet"
        pd.DataFrame({"filepath": [str(self.pool_images / "car.jpg")]}).to_parquet(mined)
        mining_summary = iter_dir / "mining.json"
        mining_summary.write_text("{}", encoding="utf-8")
        commit(
            "iter1",
            "mine",
            "--mining-output",
            str(mined),
            "--mining-summary",
            str(mining_summary),
        )

        staged_images = iter_dir / "tmm" / "images"
        staged_annotations = iter_dir / "tmm" / "annotations"
        staged_images.mkdir(parents=True)
        staged_annotations.mkdir(parents=True)
        (staged_images / "car.jpg").write_bytes(b"car")
        odvg = staged_annotations / "tmm_odvg.jsonl"
        odvg.write_text("{}\n", encoding="utf-8")
        labelmap = staged_annotations / "labelmap.json"
        labelmap.write_text('{"0":"car","1":"person"}', encoding="utf-8")
        staged_coco = staged_annotations / "tmm_coco.json"
        staged_coco.write_text(self.source_coco.read_text(encoding="utf-8"), encoding="utf-8")
        staged_classmap = staged_annotations / "rtdetr_classmap.txt"
        staged_classmap.write_text("car\nperson\n", encoding="utf-8")
        exclude = iter_dir / "exclude.parquet"
        pd.DataFrame({"filepath": ["car.jpg"]}).to_parquet(exclude)
        commit(
            "iter1",
            "stage",
            "--odvg",
            str(odvg),
            "--label-map",
            str(labelmap),
            "--staged-images-dir",
            str(staged_images),
            "--exclude-parquet",
            str(exclude),
            "--staged-coco",
            str(staged_coco),
            "--inference-classmap",
            str(staged_classmap),
        )
        final_audit = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "audit_deft_run.py"),
                "--results-dir",
                str(results),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(final_audit.returncode, 0, final_audit.stderr)
        self.assertEqual(json.loads(final_audit.stdout)["next_action"], "train")


if __name__ == "__main__":
    unittest.main()
