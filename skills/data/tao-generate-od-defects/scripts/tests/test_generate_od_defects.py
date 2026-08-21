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

    def test_validate_group_gates_copyback(self) -> None:
        report = self.root / "group_validation.json"
        generator.validate_group(
            type(
                "Args",
                (),
                {
                    "group_root": str(self.run / "toy"),
                    "dataset_id": "toy",
                    "requested": 2,
                    "anomaly_types": "toy_widget+scratch",
                    "output_json": str(report),
                },
            )()
        )
        result = json.loads(report.read_text())
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["generated"], 2)

    def test_finalize_rejects_missing_pseudo_labeled_image(self) -> None:
        self._write_coco(count=1, category="toy_widget+scratch")
        with self.assertRaisesRegex(ValueError, "pseudo-label image count mismatch"):
            generator.finalize(self._args())

    def test_finalize_rejects_unexpected_native_category(self) -> None:
        self._write_coco(count=2, category="wrong+type")
        with self.assertRaisesRegex(ValueError, "unexpected pseudo-label categories"):
            generator.finalize(self._args())

    def test_stage_runtime_materializes_frozen_testcase(self) -> None:
        source_image = self.root / "durable" / "image.png"
        source_mask = self.root / "durable" / "mask.png"
        write_image(source_image)
        write_image(source_mask)
        testcase = Path(self.plan[0]["testcase"])
        testcase.write_text(
            "".join(
                json.dumps(
                    {
                        "image_filename": str(source_image),
                        "mask_filename": str(source_mask),
                    }
                )
                + "\n"
                for _ in range(2)
            )
        )
        prepared = self.inputs / "prepared_anomalygennext_inputs"
        plan_path = prepared / "anomalygen_next_generation_plan.json"
        provenance = prepared / "anomalygen_inputs" / "toy" / "provenance.jsonl"
        generator._write_json(
            prepared / "prepared_inputs_manifest.json",
            {
                "status": "COMPLETE",
                "generation_ready": True,
                "generator_row_count": 2,
                "source_tag": "test_fixture",
                "artifacts": [
                    {
                        "path": str(path),
                        "sha256": generator._sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in (testcase, provenance, plan_path)
                ],
            },
        )
        checkpoint = self.root / "checkpoint.pt"
        recipe = self.root / "recipe.yaml"
        checkpoint.write_bytes(b"checkpoint")
        recipe.write_text("recipe: test\n")
        real_root = self.root / "real"
        real_root.mkdir()
        runtime = self.root / "runtime"
        output_tsv = self.root / "runtime.tsv"
        args = type(
            "Args",
            (),
            {
                "inputs_dir": str(self.inputs),
                "runtime_root": str(runtime),
                "local_real_root": str(real_root),
                "local_checkpoint": str(checkpoint),
                "local_recipe": str(recipe),
                "datasets": "toy",
                "output_tsv": str(output_tsv),
            },
        )()
        generator.stage_runtime(args)
        fields = output_tsv.read_text().strip().split("\t")
        self.assertEqual(fields[0], "toy")
        self.assertEqual(fields[-1], "2")
        runtime_rows = [
            json.loads(line)
            for line in Path(fields[2]).read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(runtime_rows), 2)
        self.assertTrue(all("/runtime/toy/images/" in row["image_filename"] for row in runtime_rows))
        self.assertTrue(all("/runtime/toy/masks/" in row["mask_filename"] for row in runtime_rows))


if __name__ == "__main__":
    unittest.main()
