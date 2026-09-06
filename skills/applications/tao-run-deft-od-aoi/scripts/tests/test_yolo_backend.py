#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(SCRIPT_DIR))

from write_yolo_specs import build_specs, run  # noqa: E402


class YoloSpecWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        sources = {}
        for name in ("train", "kpi", "test"):
            images = self.root / f"{name}_images"
            images.mkdir()
            (images / "one.png").write_bytes(b"image")
            coco = self.root / f"{name}.json"
            coco.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "one.png", "width": 8, "height": 8}],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "defect"}],
                    }
                ),
                encoding="utf-8",
            )
            sources[name] = {"images": str(images), "coco": str(coco)}
        (self.root / "base.pt").write_bytes(b"weights")
        policy = yaml.safe_load((SCRIPT_DIR.parents[0] / "assets/default_policy.yaml").read_text())
        policy.update(max_iterations=2, base_checkpoint=str(self.root / "base.pt"))
        policy["model"] = {"backend": "yolo", "architecture": "yolo26x"}
        policy["sources"].update({"kpi": sources["kpi"], "test": sources["test"]})
        self.policy_path = self.root / "policy.yaml"
        self.policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            policy=str(self.policy_path),
            iteration=1,
            train_coco=str(self.root / "train.json"),
            train_images=str(self.root / "train_images"),
            output_dir=str(self.root / "specs"),
            published_output_dir="/durable/specs",
            results_root="/durable/results",
            runtime_root="${TAO_RUNTIME_ROOT}",
            onelogger_enabled="true",
            onelogger_callback_module="company_onelogger",
        )

    def test_writes_three_ordered_specs(self) -> None:
        manifest = run(self._args())
        self.assertEqual(manifest["ordering"], ["train", "kpi_evaluate", "test_evaluate"])
        self.assertEqual(manifest["actions"]["train"], "/durable/specs/train.yaml")
        specs = build_specs(self._args())
        self.assertFalse(specs["train.yaml"]["train"]["resume"])
        self.assertEqual(specs["train.yaml"]["train"]["epochs"], 100)
        self.assertEqual(
            specs["train.yaml"]["dataset"]["train_images"],
            str((self.root / "train_images").resolve()),
        )
        self.assertFalse(specs["kpi_evaluate.yaml"]["evaluation"]["report_only"])
        self.assertTrue(specs["test_evaluate.yaml"]["evaluation"]["report_only"])
        self.assertEqual(
            specs["kpi_evaluate.yaml"]["model"]["checkpoint"],
            "/durable/results/iteration_1/train/selected.pt",
        )

    def test_requires_callback_when_onelogger_enabled(self) -> None:
        args = self._args()
        args.onelogger_callback_module = None
        with self.assertRaisesRegex(ValueError, "callback"):
            build_specs(args)

    def test_iteration_zero_writes_measurement_only_against_initializer(self) -> None:
        args = self._args()
        args.iteration = 0
        args.train_coco = None
        args.train_images = None
        specs = build_specs(args)
        self.assertEqual(set(specs), {"kpi_evaluate.yaml", "test_evaluate.yaml"})
        self.assertEqual(
            specs["kpi_evaluate.yaml"]["model"]["checkpoint"],
            str((self.root / "base.pt").resolve()),
        )
        self.assertTrue(specs["test_evaluate.yaml"]["evaluation"]["report_only"])

    def test_training_iteration_requires_train_coco(self) -> None:
        args = self._args()
        args.train_coco = None
        args.train_images = None
        with self.assertRaisesRegex(ValueError, "train-coco and --train-images"):
            build_specs(args)


if __name__ == "__main__":
    unittest.main()
