#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from yolo_action import load_config, run_evaluate, run_train  # noqa: E402
from yolo_common import (  # noqa: E402
    common_coco_score,
    merge_results,
    predictions_to_kitti,
    select_row,
    stage_coco,
)


class YoloSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "part.png"
        self.image.write_bytes(b"not-decoded-by-stager")
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"weights")
        self.coco = self.root / "data.json"
        self.coco.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 7, "file_name": "part.png", "width": 100, "height": 80}
                    ],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "defect"}],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, action: str) -> Path:
        value = {
            "model": {"architecture": "yolo26x", "checkpoint": str(self.checkpoint)},
            "dataset": {
                "eval_coco": str(self.coco),
                "eval_images": str(self.root),
            },
            "runtime": {"scratch_root": str(self.root / "scratch")},
            "results_dir": str(self.root / "results"),
        }
        if action == "train":
            value["dataset"].update(
                {"train_coco": str(self.coco), "train_images": str(self.root)}
            )
            value["train"] = {
                "num_gpus": 1,
                "epochs": 2,
                "patience": 1,
                "imgsz": 640,
                "batch": 4,
                "nbs": 4,
                "devices": "0",
                "workers": 0,
                "stage_workers": 1,
                "optimizer": "auto",
                "lr0": 0.01,
                "lrf": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "warmup_epochs": 1.0,
                "amp": True,
                "deterministic": True,
                "seed": 0,
                "close_mosaic": 1,
                "resume": False,
            }
        else:
            value["evaluation"] = {
                "split_name": "kpi",
                "imgsz": 640,
                "batch": 4,
                "device": "0",
                "workers": 0,
                "stage_workers": 1,
                "confidence": 0.001,
                "iou": 0.7,
                "max_det": 300,
                "report_only": False,
            }
        path = self.root / f"{action}.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    def test_nested_config_and_sealed_test_gate(self) -> None:
        self.assertEqual(load_config(self._config("train"), "train")["train"]["epochs"], 2)
        path = self._config("evaluate")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["evaluation"].update({"split_name": "test", "report_only": False})
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "report-only"):
            load_config(path, "evaluate")

    def test_action_helpers_are_checksum_closed(self) -> None:
        skill_root = SCRIPT_DIR.parents[0]
        info = yaml.safe_load((skill_root / "references/skill_info.yaml").read_text())
        for action in ("train", "evaluate"):
            execution = info["actions"][action]["execution_contract"]
            self.assertEqual(execution["distributed"]["launcher"], "direct")
            for helper in execution["supporting_files"]:
                payload = (skill_root / helper["source"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), helper["sha256"])

    def test_successful_cli_exit_is_not_reported_as_fatal(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "yolo_action.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertNotIn("FATAL", completed.stderr)

    def test_nested_flat_dotted_key_is_rejected(self) -> None:
        path = self._config("train")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["train"]["optimizer.lr0"] = 0.01
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dotted"):
            load_config(path, "train")

    def test_gpu_shape_matches_ultralytics_devices(self) -> None:
        path = self._config("train")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["train"]["num_gpus"] = 2
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "num_gpus must match"):
            load_config(path, "train")

    def test_env_runtime_path_is_accepted(self) -> None:
        path = self._config("evaluate")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["runtime"]["scratch_root"] = "${YOLO_TEST_RUNTIME}/run"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        old_runtime = os.environ.get("YOLO_TEST_RUNTIME")
        os.environ["YOLO_TEST_RUNTIME"] = "/tmp/yolo-test"
        try:
            load_config(path, "evaluate")
        finally:
            if old_runtime is None:
                os.environ.pop("YOLO_TEST_RUNTIME", None)
            else:
                os.environ["YOLO_TEST_RUNTIME"] = old_runtime

    def test_stage_clean_negative_writes_empty_label(self) -> None:
        report = stage_coco(
            self.coco, self.root, self.root / "staged", preserve_ids=True, workers=1
        )
        self.assertEqual(report["clean_images"], 1)
        self.assertEqual((self.root / "staged/labels/000000000007.txt").read_text(), "")

    def test_checkpoint_selection_uses_earliest_ap50_tie(self) -> None:
        rows = [
            {"epoch": "1", "metrics/mAP50(B)": "0.5"},
            {"epoch": "2", "metrics/mAP50(B)": "0.7"},
            {"epoch": "3", "metrics/mAP50(B)": "0.7"},
        ]
        index, row = select_row(rows)
        self.assertEqual(index, 1)
        self.assertEqual(row["epoch"], "2")

    def test_resume_curve_merges_by_reported_epoch(self) -> None:
        prior = self.root / "prior.csv"
        current = self.root / "current.csv"
        for path, rows in (
            (prior, [(1, 0.2), (2, 0.4)]),
            (current, [(2, 0.45), (3, 0.5)]),
        ):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "metrics/mAP50(B)"])
                writer.writerows(rows)
        rows, prior_last = merge_results(prior, current)
        self.assertEqual(prior_last, 2)
        self.assertEqual([row["epoch"] for row in rows], ["1", "2", "3"])
        self.assertEqual(rows[1]["metrics/mAP50(B)"], "0.4")

    def test_kitti_export_preserves_original_stem(self) -> None:
        predictions = self.root / "predictions.json"
        predictions.write_text(
            json.dumps([{"image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9}]),
            encoding="utf-8",
        )
        report = predictions_to_kitti(self.coco, predictions, self.root / "labels", 0.001)
        self.assertEqual(report["kept_predictions"], 1)
        self.assertTrue((self.root / "labels/part.txt").read_text().startswith("defect "))

    def test_common_coco_score_handles_no_detections(self) -> None:
        predictions = self.root / "empty-predictions.json"
        predictions.write_text("[]\n", encoding="utf-8")
        output = self.root / "metrics/common.json"
        output.parent.mkdir()
        metrics = common_coco_score(self.coco, predictions, output)
        self.assertEqual(metrics["AP50"], 0.0)
        self.assertEqual(metrics["detections"], 0)
        self.assertEqual(json.loads(output.read_text()), metrics)

    def test_fresh_train_action_publishes_allowlisted_checkpoints(self) -> None:
        class FakeYOLO:
            def __init__(self, checkpoint: str) -> None:
                self.checkpoint = checkpoint
                self.trainer = None

            def add_callback(self, name: str, callback: object) -> None:
                self.callback = (name, callback)

            def train(self, **kwargs: object) -> None:
                train_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                weights = train_dir / "weights"
                weights.mkdir(parents=True)
                with (train_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["epoch", "metrics/mAP50(B)"])
                    writer.writerows([(1, 0.7), (2, 0.6)])
                (weights / "epoch0.pt").write_bytes(b"best")
                (weights / "epoch1.pt").write_bytes(b"terminal")
                (weights / "last.pt").write_bytes(b"last")
                self.trainer = SimpleNamespace(save_dir=train_dir)

        config = load_config(self._config("train"), "train")
        with mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
            run_train(config)
        output = self.root / "results"
        self.assertEqual((output / "selected.pt").read_bytes(), b"best")
        self.assertEqual((output / "terminal_resume.pt").read_bytes(), b"terminal")
        self.assertEqual(json.loads((output / "status.json").read_text())["state"], "COMPLETE")
        self.assertFalse((output / "trainer").exists())

    def test_job_record_staged_directory_can_preexist(self) -> None:
        class FakeYOLO:
            def __init__(self, checkpoint: str) -> None:
                self.trainer = None

            def add_callback(self, name: str, callback: object) -> None:
                pass

            def train(self, **kwargs: object) -> None:
                train_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                weights = train_dir / "weights"
                weights.mkdir(parents=True)
                with (train_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["epoch", "metrics/mAP50(B)"])
                    writer.writerow([1, 0.7])
                (weights / "epoch0.pt").write_bytes(b"selected")
                (weights / "last.pt").write_bytes(b"last")
                self.trainer = SimpleNamespace(save_dir=train_dir)

        staged = self.root / "results/staged"
        staged.mkdir(parents=True)
        (staged / "spec.yaml").write_text("immutable input\n", encoding="utf-8")
        config = load_config(self._config("train"), "train")
        with mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
            run_train(config)
        self.assertEqual((staged / "spec.yaml").read_text(), "immutable input\n")
        self.assertEqual(json.loads((self.root / "results/status.json").read_text())["state"], "COMPLETE")

    def test_existing_workload_artifacts_are_rejected(self) -> None:
        output = self.root / "results"
        output.mkdir()
        (output / "selected.pt").write_bytes(b"old")
        config = load_config(self._config("train"), "train")
        with mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=object)}):
            with self.assertRaisesRegex(FileExistsError, "workload artifacts"):
                run_train(config)

    def test_evaluate_action_publishes_predictions_and_report(self) -> None:
        class FakeYOLO:
            def __init__(self, checkpoint: str) -> None:
                self.checkpoint = checkpoint

            def val(self, **kwargs: object) -> object:
                save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                save_dir.mkdir(parents=True)
                (save_dir / "predictions.json").write_text(
                    json.dumps(
                        [{"image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9}]
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(save_dir=save_dir, results_dict={"metrics/mAP50(B)": 0.8})

        config = load_config(self._config("evaluate"), "evaluate")
        with (
            mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}),
            mock.patch("yolo_action.common_coco_score", return_value={"AP50": 0.8}) as score,
        ):
            run_evaluate(config)
        output = self.root / "results"
        self.assertTrue((output / "predictions.json").is_file())
        self.assertTrue((output / "kitti_labels/part.txt").is_file())
        self.assertEqual(json.loads((output / "status.json").read_text())["state"], "COMPLETE")
        score.assert_called_once()

    def test_resume_action_can_retain_prior_kpi_best(self) -> None:
        prior_csv = self.root / "prior.csv"
        with prior_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "metrics/mAP50(B)"])
            writer.writerows([(1, 0.9), (2, 0.8)])
        prior_best = self.root / "prior-best.pt"
        prior_best.write_bytes(b"prior-best")
        resumed_dir = self.root / "resumed-trainer"

        class FakeYOLO:
            def __init__(self, checkpoint: str) -> None:
                self.trainer = None

            def add_callback(self, name: str, callback: object) -> None:
                self.callback = (name, callback)

            def train(self, **kwargs: object) -> None:
                self.testcase.assertTrue(kwargs["resume"])
                weights = resumed_dir / "weights"
                weights.mkdir(parents=True)
                with (resumed_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["epoch", "metrics/mAP50(B)"])
                    writer.writerows([(2, 0.8), (3, 0.85)])
                (weights / "epoch2.pt").write_bytes(b"terminal")
                (weights / "last.pt").write_bytes(b"last")
                self.trainer = SimpleNamespace(save_dir=resumed_dir)

        FakeYOLO.testcase = self
        path = self._config("train")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["train"].update(
            {
                "resume": True,
                "prior_results_csv": str(prior_csv),
                "prior_best_checkpoint": str(prior_best),
            }
        )
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        config = load_config(path, "train")
        with mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
            run_train(config)
        output = self.root / "results"
        self.assertEqual((output / "selected.pt").read_bytes(), b"prior-best")
        self.assertEqual((output / "terminal_resume.pt").read_bytes(), b"terminal")
        selection = json.loads((output / "selection.json").read_text())
        self.assertEqual(selection["selected_epoch_reported"], 1)


if __name__ == "__main__":
    unittest.main()
