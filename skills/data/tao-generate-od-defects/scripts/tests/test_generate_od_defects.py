#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for AnomalyGenNext OD generation reconciliation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "generate_od_defects.py"
SPEC = importlib.util.spec_from_file_location("generate_od_defects", SCRIPT)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8)).save(path)


class GenerateOdDefectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.run = self.root / "run"
        prepared = self.inputs / "prepared_anomalygennext_inputs"
        testcase = prepared / "anomalygen_inputs" / "toy" / "testcase.jsonl"
        provenance = prepared / "anomalygen_inputs" / "toy" / "provenance.jsonl"
        testcase.parent.mkdir(parents=True)
        testcase.write_text('{}\n{}\n')
        provenance.write_text('{}\n{}\n')
        self.plan = [
            {
                "dataset_id": "toy",
                "anomaly_type": "toy_widget+scratch",
                "anomaly_types": ["toy_widget+scratch"],
                "testcase": str(testcase),
                "provenance": str(provenance),
                "checkpoint": "/path/to/checkpoint.pt",
                "recipe": "/path/to/recipe.yaml",
                "real_root": "/path/to/real",
                "requested_rows": 2,
            }
        ]
        plan_path = prepared / "anomalygen_next_generation_plan.json"
        generator._write_json(plan_path, self.plan)
        artifacts = [
            {"path": str(path), "sha256": generator._sha256(path), "bytes": path.stat().st_size}
            for path in (testcase, provenance, plan_path)
        ]
        generator._write_json(
            prepared / "prepared_inputs_manifest.json",
            {
                "status": "COMPLETE",
                "generation_ready": True,
                "generator_row_count": 2,
                "source_tag": "test_fixture",
                "artifacts": artifacts,
            },
        )
        self.raw = self.run / "toy" / "raw"
        self.searched = self.run / "toy" / "searched"
        self.raw.mkdir(parents=True)
        (self.raw / "texture_ft_generation_result.csv").write_text(
            "output_filename\nout0.png\nout1.png\n"
        )
        self._write_coco(count=2, category="toy_widget+scratch")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_coco(self, *, count: int, category: str) -> None:
        images = []
        annotations = []
        for index in range(count):
            image = self.searched / "reconstructed_image" / f"out{index}.png"
            write_image(image)
            images.append(
                {"id": index + 1, "file_name": image.name, "width": 32, "height": 32}
            )
            annotations.append(
                {
                    "id": index + 1,
                    "image_id": index + 1,
                    "category_id": 1,
                    "bbox": [8, 8, 8, 8],
                    "area": 64,
                    "iscrowd": 0,
                }
            )
        generator._write_json(
            self.searched / "pseudo_labels" / "coco_annotations.json",
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": category}],
            },
        )

    def _args(self) -> object:
        return type(
            "Args",
            (),
            {"inputs_dir": str(self.inputs), "run_root": str(self.run), "datasets": None},
        )()

    def test_dataset_subset_rejects_unknown_ids(self) -> None:
        rows = [{"dataset_id": "line_a"}, {"dataset_id": "line_b"}]
        self.assertEqual(generator._selected_groups(rows, "line_b"), [{"dataset_id": "line_b"}])
        with self.assertRaisesRegex(ValueError, "unknown dataset ids"):
            generator._selected_groups(rows, "line_c")

    def test_finalize_reconciles_coco_and_preserves_input_hash(self) -> None:
        generator.finalize(self._args())
        summary = json.loads((self.run / "validation_summary.json").read_text())
        self.assertEqual(summary["generated_images"], 2)
        self.assertEqual(summary["pseudo_labeled_images"], 2)
        self.assertEqual(summary["native_categories"], ["toy_widget+scratch"])
        self.assertFalse(summary["training_pool_mutated"])

    def test_finalize_rejects_missing_pseudo_labeled_image(self) -> None:
        self._write_coco(count=1, category="toy_widget+scratch")
        with self.assertRaisesRegex(ValueError, "pseudo-label image count mismatch"):
            generator.finalize(self._args())

    def test_finalize_rejects_unexpected_native_category(self) -> None:
        self._write_coco(count=2, category="wrong+type")
        with self.assertRaisesRegex(ValueError, "unexpected pseudo-label categories"):
            generator.finalize(self._args())


if __name__ == "__main__":
    unittest.main()
