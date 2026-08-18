#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import assemble_deft_od_aoi_coco  # noqa: E402
import route_deft_od_aoi  # noqa: E402
import select_deft_od_aoi_checkpoint  # noqa: E402
import select_deft_od_aoi_probe  # noqa: E402
import validate_deft_od_aoi_inputs  # noqa: E402
import write_gap_specs  # noqa: E402
import write_rtdetr_specs  # noqa: E402
from deft_od_aoi_policy import build_policy, probes_active, uniform_mine_for_iteration  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def coco(images: list[dict], annotations: list[dict]) -> dict:
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "defect"}],
    }


def image_record(image_id: int, file_name: str, *, defect_type: str = "scratch") -> dict:
    return {
        "id": image_id,
        "file_name": file_name,
        "width": 64,
        "height": 64,
        "deft_od_aoi": {
            "benchmark": "bench",
            "texture": "metal",
            "defect_type": defect_type,
            "generator_type": "metal+scratch",
        },
    }


class PolicyTest(unittest.TestCase):
    def test_reference_uniform_schedule(self) -> None:
        policy = build_policy(
            profile="deft_od_aoi_reference",
            max_iterations=10,
            uniform_mine_per_pocket=None,
            synthetic_enabled=None,
        )
        self.assertEqual(uniform_mine_for_iteration(policy, 1), 12)
        self.assertEqual(uniform_mine_for_iteration(policy, 2), 12)
        self.assertEqual(uniform_mine_for_iteration(policy, 3), 0)
        self.assertEqual(uniform_mine_for_iteration(policy, 10), 0)

    def test_configurable_requires_uniform(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires uniform"):
            build_policy(
                profile="configurable",
                max_iterations=3,
                uniform_mine_per_pocket=None,
                synthetic_enabled=True,
            )
        policy = build_policy(
            profile="configurable",
            max_iterations=3,
            uniform_mine_per_pocket=0,
            synthetic_enabled=False,
        )
        self.assertEqual(uniform_mine_for_iteration(policy, 2), 0)
        self.assertTrue(policy["training"]["probes_enabled"])
        self.assertTrue(probes_active(policy, 3))
        disabled = build_policy(
            profile="configurable",
            max_iterations=3,
            uniform_mine_per_pocket=0,
            synthetic_enabled=False,
            probes_enabled=False,
        )
        self.assertFalse(disabled["training"]["probes_enabled"])
        self.assertFalse(probes_active(disabled, 3))

    def test_dual_gap_specs(self) -> None:
        policy = build_policy(
            profile="configurable",
            max_iterations=3,
            uniform_mine_per_pocket=0,
            synthetic_enabled=False,
        )
        args = argparse.Namespace(
            ground_truth_ann_path="ground_truth",
            inference_ann_path="predictions",
            images_dir="images",
            output_dir="gaps",
            kpi="iter0",
        )
        loose = write_gap_specs.build_spec(args, policy, "loose")
        strict = write_gap_specs.build_spec(args, policy, "strict")
        self.assertEqual(loose["conf_threshold"], 0.3)
        self.assertEqual(strict["conf_threshold"], 0.8)
        self.assertEqual(loose["iou_threshold"], 0.5)
        self.assertEqual(loose["weak_thresholds"]["defect"]["recall"], 1.0)
        self.assertEqual(loose["default_precision_threshold"], 0.0)


class EndToEndDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.kpi_images = self.root / "kpi_images"
        self.test_images = self.root / "test_images"
        self.source_images = self.root / "source_images"
        self.clean_images = self.root / "clean_images"
        for directory in (
            self.kpi_images,
            self.test_images,
            self.source_images,
            self.clean_images,
        ):
            directory.mkdir()

        make_image(self.kpi_images / "kpi.png", 1)
        make_image(self.test_images / "test.png", 2)
        for index in range(1, 9):
            make_image(self.source_images / f"source{index}.png", 10 + index)
        for index in range(1, 7):
            make_image(self.clean_images / f"clean{index}.png", 30 + index)

        self.kpi_coco = self.root / "kpi.json"
        self.test_coco = self.root / "test.json"
        self.source_coco = self.root / "source.json"
        self.clean_coco = self.root / "clean.json"
        write_json(
            self.kpi_coco,
            coco(
                [
                    {
                        **image_record(1, "kpi.png"),
                        "source_path": str(self.kpi_images / "kpi.png"),
                    }
                ],
                [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
            ),
        )
        write_json(
            self.test_coco,
            coco(
                [
                    {
                        **image_record(1, "test.png"),
                        "source_path": str(self.test_images / "test.png"),
                    }
                ],
                [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 18, 18]}],
            ),
        )
        source_records = []
        source_annotations = []
        for index in range(1, 9):
            source_records.append(
                {
                    **image_record(index, f"source{index}.png"),
                    "source_path": str(self.source_images / f"source{index}.png"),
                }
            )
            source_annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [10 + index % 3, 11, 18, 17],
                }
            )
        write_json(self.source_coco, coco(source_records, source_annotations))
        clean_records = []
        for index in range(1, 7):
            record = image_record(index, f"clean{index}.png")
            record["source_path"] = str(self.clean_images / f"clean{index}.png")
            record["deft_od_aoi"].pop("defect_type")
            clean_records.append(record)
        write_json(self.clean_coco, coco(clean_records, []))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validate_route_and_assemble(self) -> None:
        validation = validate_deft_od_aoi_inputs.run(
            argparse.Namespace(
                kpi_coco=str(self.kpi_coco),
                kpi_images_dir=str(self.kpi_images),
                test_coco=str(self.test_coco),
                test_images_dir=str(self.test_images),
                source_coco=str(self.source_coco),
                source_images_dir=str(self.source_images),
                clean_coco=str(self.clean_coco),
                clean_images_dir=str(self.clean_images),
                output=None,
            )
        )
        self.assertEqual(validation["status"], "valid")

        policy_path = self.root / "policy.json"
        write_json(
            policy_path,
            build_policy(
                profile="configurable",
                max_iterations=3,
                uniform_mine_per_pocket=0,
                synthetic_enabled=True,
            ),
        )
        loose_path = self.root / "loose.parquet"
        strict_path = self.root / "strict.parquet"
        pd.DataFrame(
            [
                {
                    "gap_type": "FP",
                    "filepath": str(self.kpi_images / "kpi.png"),
                    "bbox": [1.0, 1.0, 5.0, 5.0],
                    "best_iou": 0.01,
                },
                {
                    "gap_type": "FP",
                    "filepath": str(self.kpi_images / "kpi.png"),
                    "bbox": [7.0, 7.0, 20.0, 20.0],
                    "best_iou": 0.2,
                },
            ]
        ).to_parquet(loose_path)
        pd.DataFrame(
            [
                {
                    "gap_type": "FN",
                    "filepath": str(self.kpi_images / "kpi.png"),
                    "bbox": [10.0, 10.0, 30.0, 30.0],
                    "best_iou": 0.4,
                }
            ]
        ).to_parquet(strict_path)
        route_dir = self.root / "route"
        valid_generators = self.root / "valid_generators.json"
        write_json(valid_generators, ["metal+scratch"])
        report = route_deft_od_aoi.route(
            argparse.Namespace(
                policy=str(policy_path),
                iteration=1,
                loose_gaps=str(loose_path),
                strict_gaps=str(strict_path),
                kpi_coco=str(self.kpi_coco),
                kpi_images_dir=str(self.kpi_images),
                source_coco=str(self.source_coco),
                source_images_dir=str(self.source_images),
                clean_coco=str(self.clean_coco),
                clean_images_dir=str(self.clean_images),
                output_dir=str(route_dir),
                previous_defect_ledger=None,
                previous_clean_ledger=None,
                previous_admission_index=None,
                conversion_old_strict=None,
                conversion_new_strict=None,
                prior_admitted_synthetic=0,
                valid_generator_types=str(valid_generators),
            )
        )
        self.assertEqual(report["background_hit_count"], 1)
        self.assertEqual(report["near_miss_count"], 1)
        self.assertGreater(report["selected"]["strict_fn_real"], 0)
        self.assertGreater(report["selected"]["near_miss_real"], 0)
        self.assertGreater(report["selected"]["background_clean"], 0)
        self.assertLessEqual(
            report["cumulative_clean_negatives"],
            report["cumulative_real_defectives"],
        )

        output_coco = self.root / "train" / "annotations.json"
        assembly = assemble_deft_od_aoi_coco.run(
            argparse.Namespace(
                route_manifest=[str(route_dir / "mined_manifest.json")],
                synthetic_source=[],
                output_coco=str(output_coco),
                output_images_dir=str(self.root / "train" / "images"),
                link_mode="copy",
            )
        )
        assembled = json.loads(output_coco.read_text(encoding="utf-8"))
        annotated_ids = {annotation["image_id"] for annotation in assembled["annotations"]}
        clean_ids = {
            image["id"]
            for image in assembled["images"]
            if image["deft_od_aoi_kind"] == "clean_negative"
        }
        self.assertEqual(assembly["images"], len(assembled["images"]))
        self.assertTrue(clean_ids)
        self.assertFalse(clean_ids & annotated_ids)
        self.assertEqual(assembled["categories"], [{"id": 1, "name": "defect", "supercategory": "defect"}])

        base_checkpoint = self.root / "warehouse.pth"
        base_checkpoint.write_bytes(b"test-checkpoint")
        classmap = self.root / "label_map.txt"
        classmap.write_text("defect\n", encoding="utf-8")
        specs_dir = self.root / "specs"
        train_results = self.root / "train_results"
        manifest = write_rtdetr_specs.run(
            argparse.Namespace(
                policy=str(policy_path),
                iteration=3,
                train_coco=str(output_coco),
                train_images_dir=str(self.root / "train" / "images"),
                kpi_coco=str(self.kpi_coco),
                kpi_images_dir=str(self.kpi_images),
                test_images_dir=str(self.test_images),
                classmap=str(classmap),
                base_checkpoint=str(base_checkpoint),
                output_dir=str(specs_dir),
                results_root=str(train_results),
                incumbent_config=None,
                history=None,
            )
        )
        self.assertEqual(len(manifest["probes"]), 3)
        for probe, metric in zip(manifest["probes"], (0.1, 0.3, 0.2)):
            status = Path(probe["results_dir"]) / "train" / "status.json"
            status.parent.mkdir(parents=True)
            status.write_text(
                json.dumps({"epoch": 1, "kpi": {"val_mAP50": metric}}) + "\n",
                encoding="utf-8",
            )
        winner = select_deft_od_aoi_probe.run(
            argparse.Namespace(
                policy=str(policy_path),
                probe_manifest=str(specs_dir / "probe_manifest.json"),
                train_spec=str(specs_dir / "train.yaml"),
                winner_output=str(specs_dir / "winner.json"),
                best_config_output=str(specs_dir / "best_config.json"),
                history_output=str(specs_dir / "history.json"),
            )
        )
        self.assertEqual(winner["name"], "scaled")
        self.assertEqual(winner["num_epochs"], 48)
        train_spec_text = (specs_dir / "train.yaml").read_text(encoding="utf-8")
        self.assertNotIn("train.optim.lr:", train_spec_text)
        self.assertIn("num_epochs: 48", train_spec_text)

        checkpoint_dir = self.root / "main_checkpoints"
        checkpoint_dir.mkdir()
        for epoch in (45, 47):
            (checkpoint_dir / f"model_epoch_{epoch:03d}.pth").write_bytes(b"checkpoint")
        main_status = self.root / "main_status.json"
        main_status.write_text(
            json.dumps({"epoch": 45, "kpi": {"val_mAP50": 0.4}})
            + "\n"
            + json.dumps({"epoch": 47, "kpi": {"val_mAP50": 0.5}})
            + "\n",
            encoding="utf-8",
        )
        selection = select_deft_od_aoi_checkpoint.run(
            argparse.Namespace(
                policy=str(policy_path),
                status=str(main_status),
                checkpoint_dir=str(checkpoint_dir),
                planned_epochs=48,
                extension_applied=False,
                output=str(self.root / "selection.json"),
            )
        )
        self.assertEqual(selection["action"], "extend")
        self.assertEqual(selection["extended_num_epochs"], 60)

        disabled_policy = self.root / "policy_no_probes.json"
        write_json(
            disabled_policy,
            build_policy(
                profile="configurable",
                max_iterations=3,
                uniform_mine_per_pocket=0,
                synthetic_enabled=False,
                probes_enabled=False,
            ),
        )
        skipped = write_rtdetr_specs.run(
            argparse.Namespace(
                policy=str(disabled_policy),
                iteration=3,
                train_coco=str(output_coco),
                train_images_dir=str(self.root / "train" / "images"),
                kpi_coco=str(self.kpi_coco),
                kpi_images_dir=str(self.kpi_images),
                test_images_dir=str(self.test_images),
                classmap=str(classmap),
                base_checkpoint=str(base_checkpoint),
                output_dir=str(self.root / "specs_no_probes"),
                results_root=str(self.root / "train_results_no_probes"),
                incumbent_config=None,
                history=None,
            )
        )
        self.assertEqual(skipped["probes"], [])
        skipped_train = (self.root / "specs_no_probes" / "train.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("num_epochs: 48", skipped_train)


if __name__ == "__main__":
    unittest.main()
