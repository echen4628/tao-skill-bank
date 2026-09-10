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
from select_yolo_probes import select as select_yolo_probes  # noqa: E402


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
        # Use an explicit optimizer recipe to verify that probe selection carries
        # fixed training values without changing the packaged defaults.
        policy["yolo"]["training"]["optimizer"] = "MuSGD"
        policy["yolo"]["training"]["momentum"] = 0.9
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
            phase="train",
            train_coco=str(self.root / "train.json"),
            train_images=str(self.root / "train_images"),
            selected_checkpoint=None,
            probe_winner=None,
            output_dir=str(self.root / "specs"),
            published_output_dir="/durable/specs",
            runtime_root="${TAO_RUNTIME_ROOT}",
        )

    def test_writes_three_ordered_specs(self) -> None:
        manifest = run(self._args())
        self.assertEqual(manifest["ordering"], ["train"])
        self.assertEqual(manifest["actions"]["train"], "/durable/specs/train.yaml")
        specs = build_specs(self._args())
        self.assertFalse(specs["train.yaml"]["train"]["resume"])
        self.assertEqual(specs["train.yaml"]["train"]["epochs"], 100)
        self.assertEqual(
            specs["train.yaml"]["dataset"]["train_images"],
            str((self.root / "train_images").resolve()),
        )
        self.assertEqual(specs["train.yaml"]["results_dir"], "{results_dir}")

    def test_iteration_zero_writes_measurement_only_against_initializer(self) -> None:
        args = self._args()
        args.iteration = 0
        args.phase = "baseline"
        args.train_coco = None
        args.train_images = None
        specs = build_specs(args)
        self.assertEqual(set(specs), {"kpi_evaluate.yaml", "test_evaluate.yaml"})
        self.assertEqual(
            specs["kpi_evaluate.yaml"]["model"]["checkpoint"],
            str((self.root / "base.pt").resolve()),
        )
        self.assertTrue(specs["test_evaluate.yaml"]["evaluation"]["report_only"])
        self.assertEqual(specs["kpi_evaluate.yaml"]["results_dir"], "{results_dir}")

    def test_measurement_requires_and_uses_frozen_selected_checkpoint(self) -> None:
        args = self._args()
        args.phase = "measure"
        args.selected_checkpoint = str(self.root / "base.pt")
        specs = build_specs(args)
        self.assertEqual(
            specs["kpi_evaluate.yaml"]["model"]["checkpoint"],
            str((self.root / "base.pt").resolve()),
        )
        self.assertFalse(specs["kpi_evaluate.yaml"]["evaluation"]["report_only"])
        self.assertTrue(specs["test_evaluate.yaml"]["evaluation"]["report_only"])

    def test_training_iteration_requires_train_coco(self) -> None:
        args = self._args()
        args.phase = "train"
        args.train_coco = None
        args.train_images = None
        with self.assertRaisesRegex(ValueError, "train-coco and --train-images"):
            build_specs(args)

    def test_optional_probe_phase_writes_three_fresh_base_specs(self) -> None:
        policy = yaml.safe_load(self.policy_path.read_text())
        policy["yolo"]["probes"]["enabled"] = True
        self.policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
        args = self._args()
        args.phase = "probe"
        specs = build_specs(args)
        self.assertEqual(
            list(specs),
            ["probe_0_conservative.yaml", "probe_1_baseline.yaml", "probe_2_aggressive.yaml"],
        )
        for spec in specs.values():
            self.assertEqual(spec["model"]["checkpoint"], str((self.root / "base.pt").resolve()))
            self.assertEqual(spec["train"]["epochs"], 10)
            self.assertEqual(spec["train"]["seed"], 4001)
            self.assertEqual(spec["results_dir"], "{results_dir}")

    def test_kpi_only_probe_selection_drives_fresh_base_main(self) -> None:
        policy = yaml.safe_load(self.policy_path.read_text())
        policy["yolo"]["probes"]["enabled"] = True
        self.policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
        args = self._args()
        args.phase = "probe"
        manifest = run(args)
        selections = []
        for index, score in enumerate((0.61, 0.72, 0.70)):
            path = self.root / f"selection{index}.json"
            path.write_text(json.dumps({
                "selection_metric": "metrics/mAP50(B)",
                "metric_value": score,
                "selected_epoch_reported": 8,
            }), encoding="utf-8")
            selections.append(path)
        winner_path = self.root / "winning_recipe.json"
        winner = select_yolo_probes(
            self.root / "specs/yolo_spec_manifest.json", selections, winner_path
        )
        self.assertEqual(winner["winner"]["name"], "baseline")
        self.assertFalse(winner["test_used_for_selection"])
        self.assertEqual(winner["overrides"]["optimizer"], "MuSGD")
        self.assertEqual(winner["overrides"]["momentum"], 0.9)
        args = self._args()
        args.probe_winner = str(winner_path)
        spec = build_specs(args)["train.yaml"]
        self.assertEqual(spec["model"]["checkpoint"], str((self.root / "base.pt").resolve()))
        self.assertEqual(spec["train"]["lr0"], 0.01)
        self.assertEqual(spec["train"]["seed"], 4001)
        self.assertEqual(spec["train"]["optimizer"], "MuSGD")
        self.assertEqual(spec["train"]["momentum"], 0.9)


if __name__ == "__main__":
    unittest.main()
