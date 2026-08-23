#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import assemble_deft_od_aoi_coco  # noqa: E402
import inspect_deft_od_aoi_sources  # noqa: E402
import normalize_deft_od_aoi_pools  # noqa: E402
import prepare_deft_od_aoi_siglip_candidates  # noqa: E402
import prepare_deft_od_aoi_siglip_queries  # noqa: E402
import prepare_deft_od_aoi_sources  # noqa: E402
import prepare_rtdetr_extension_spec  # noqa: E402
import prepare_rtdetr_measurement_specs  # noqa: E402
import route_deft_od_aoi  # noqa: E402
import route_deft_od_aoi_iteration  # noqa: E402
import route_deft_od_aoi_siglip  # noqa: E402
import select_deft_od_aoi_checkpoint  # noqa: E402
import select_deft_od_aoi_probe  # noqa: E402
import validate_deft_od_aoi_inputs  # noqa: E402
import write_gap_specs  # noqa: E402
import write_rtdetr_specs  # noqa: E402
from deft_od_aoi_policy import (  # noqa: E402
    build_policy,
    probes_active,
    uniform_mine_for_iteration,
    validate_policy,
)


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


class CumulativeAssemblyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for seed, name in enumerate(
            (
                "real1.png",
                "synth1.png",
                "real2.png",
                "clean2.png",
                "synth2.png",
                "real3.png",
                "clean3.png",
            ),
            start=1,
        ):
            make_image(self.source / name, seed)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _route(
        self,
        iteration: int,
        records: list[dict],
        real: int,
        clean: int,
    ) -> tuple[Path, Path]:
        route = self.root / f"route{iteration}.json"
        report = self.root / f"routing_report{iteration}.json"
        write_json(route, records)
        write_json(
            report,
            {
                "cumulative_real_defectives": real,
                "cumulative_clean_negatives": clean,
            },
        )
        return route, report

    def _synthetic(self, iteration: int, name: str) -> str:
        path = self.root / f"synthetic{iteration}.json"
        write_json(
            path,
            coco(
                [
                    {
                        "id": 1,
                        "file_name": name,
                        "source_path": str(self.source / name),
                        "width": 64,
                        "height": 64,
                    }
                ],
                [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 8, 9]}],
            ),
        )
        return f"{path}::{self.source}"

    def _assemble(
        self,
        iteration: int,
        route: Path,
        report: Path,
        *,
        previous: Path | None,
        synthetic: list[str],
    ) -> tuple[Path, dict]:
        output = self.root / f"train{iteration}" / "annotations.json"
        result = assemble_deft_od_aoi_coco.run(
            argparse.Namespace(
                previous_assembled_coco=str(previous) if previous else None,
                route_manifest=[str(route)],
                current_routing_report=str(report),
                synthetic_source=synthetic,
                output_coco=str(output),
                output_images_dir=str(output.parent / "images"),
                link_mode="copy",
            )
        )
        return output, result

    def test_three_iterations_retain_every_previous_image(self) -> None:
        route1, report1 = self._route(
            1,
            [
                {
                    "source_path": str(self.source / "real1.png"),
                    "width": 64,
                    "height": 64,
                    "boxes": [[1, 1, 10, 10]],
                    "kind": "real_defect",
                }
            ],
            real=1,
            clean=0,
        )
        train1, result1 = self._assemble(
            1,
            route1,
            report1,
            previous=None,
            synthetic=[self._synthetic(1, "synth1.png")],
        )

        route2, report2 = self._route(
            2,
            [
                {
                    "source_path": str(self.source / "real2.png"),
                    "width": 64,
                    "height": 64,
                    "boxes": [[4, 4, 11, 12]],
                    "kind": "real_defect",
                },
                {
                    "source_path": str(self.source / "clean2.png"),
                    "width": 64,
                    "height": 64,
                    "boxes": [],
                    "kind": "clean_negative",
                    "allow_empty_annotations": True,
                },
            ],
            real=2,
            clean=1,
        )
        train2, result2 = self._assemble(
            2,
            route2,
            report2,
            previous=train1,
            synthetic=[self._synthetic(2, "synth2.png")],
        )

        route3, report3 = self._route(
            3,
            [
                {
                    "source_path": str(self.source / "real3.png"),
                    "width": 64,
                    "height": 64,
                    "boxes": [[7, 8, 9, 10]],
                    "kind": "real_defect",
                },
                {
                    "source_path": str(self.source / "clean3.png"),
                    "width": 64,
                    "height": 64,
                    "boxes": [],
                    "kind": "clean_negative",
                    "allow_empty_annotations": True,
                },
            ],
            real=3,
            clean=2,
        )
        train3, result3 = self._assemble(
            3,
            route3,
            report3,
            previous=train2,
            synthetic=[],
        )

        image_sets = []
        for path in (train1, train2, train3):
            assembled = json.loads(path.read_text(encoding="utf-8"))
            image_sets.append({image["source_path"] for image in assembled["images"]})
        self.assertTrue(image_sets[0] < image_sets[1] < image_sets[2])
        self.assertEqual(result1["by_kind"], {"real_defect": 1, "synthetic_defect": 1})
        self.assertEqual(
            result2["by_kind"],
            {"clean_negative": 1, "real_defect": 2, "synthetic_defect": 2},
        )
        self.assertEqual(
            result3["by_kind"],
            {"clean_negative": 2, "real_defect": 3, "synthetic_defect": 2},
        )
        self.assertEqual(result2["retained_previous_images"], 2)
        self.assertEqual(result3["retained_previous_images"], 5)

    def test_cumulative_ledger_gate_rejects_missing_previous_coco(self) -> None:
        route, report = self._route(
            2,
            [
                {
                    "source_path": str(self.source / "real2.png"),
                    "width": 64,
                    "height": 64,
                    "boxes": [[4, 4, 11, 12]],
                    "kind": "real_defect",
                }
            ],
            real=2,
            clean=0,
        )
        with self.assertRaisesRegex(ValueError, "include the immediately previous"):
            self._assemble(2, route, report, previous=None, synthetic=[])
        self.assertFalse((self.root / "train2").exists())


class PolicyTest(unittest.TestCase):
    def test_policy_requires_explicit_run_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "synthetic_enabled"):
            build_policy(
                max_iterations=3,
                synthetic_enabled=None,
            )
        policy = build_policy(
            max_iterations=3,
            synthetic_enabled=False,
        )
        self.assertEqual(policy["schema_version"], 4)
        self.assertEqual(policy["retrieval"]["mode"], "siglip_only")
        self.assertEqual(uniform_mine_for_iteration(policy, 2), 0)
        self.assertTrue(policy["training"]["probes_enabled"])
        self.assertTrue(policy["model_soup"]["enabled"])
        self.assertEqual(policy["model_soup"]["method"], "greedy")
        self.assertTrue(probes_active(policy, 3))
        ipc_safe = build_policy(
            max_iterations=3,
            synthetic_enabled=False,
            training_workers=0,
            inference_workers=0,
        )
        self.assertEqual(ipc_safe["training"]["workers"], 0)
        self.assertEqual(ipc_safe["inference"]["workers"], 0)
        with self.assertRaisesRegex(ValueError, "workers"):
            build_policy(max_iterations=1, synthetic_enabled=False, training_workers=-1)
        disabled = build_policy(
            max_iterations=3,
            synthetic_enabled=False,
            probes_enabled=False,
        )
        self.assertFalse(disabled["training"]["probes_enabled"])
        self.assertFalse(probes_active(disabled, 3))
        soup_disabled = build_policy(
            max_iterations=3,
            synthetic_enabled=False,
            model_soup_enabled=False,
        )
        self.assertFalse(soup_disabled["model_soup"]["enabled"])
        legacy = dict(policy)
        legacy["schema_version"] = 3
        legacy.pop("model_soup")
        validate_policy(legacy)

    def test_dual_gap_specs(self) -> None:
        policy = build_policy(
            max_iterations=3,
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

    def test_gap_writer_projects_binary_coco_to_kitti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "kpi.json"
            write_json(
                source,
                coco(
                    [
                        {"id": 1, "file_name": "defect.png", "width": 64, "height": 64},
                        {"id": 2, "file_name": "clean.png", "width": 64, "height": 64},
                    ],
                    [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 10, 12]}],
                ),
            )
            report = write_gap_specs.project_coco_to_kitti(
                str(source), str(root / "labels"), "defect"
            )
            self.assertEqual(report["images"], 2)
            self.assertIn("2.000000 3.000000 12.000000 15.000000", (root / "labels" / "defect.txt").read_text())
            self.assertEqual((root / "labels" / "clean.txt").read_text(), "")

    def test_binary_gap_analysis_writes_loose_and_strict_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); predictions = root / "predictions"; predictions.mkdir()
            source = root / "kpi.json"
            write_json(source, coco([{"id": 1, "file_name": "one.png", "source_path": "/durable/one.png", "width": 64, "height": 64}], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10]}]))
            (predictions / "one.txt").write_text(
                "defect 0 0 0 10 10 20 20 0 0 0 0 0 0 0 0.7\n"
                "defect 0 0 0 40 40 50 50 0 0 0 0 0 0 0 0.4\n"
            )
            policy = build_policy(max_iterations=1, synthetic_enabled=False)
            args = argparse.Namespace(
                ground_truth_coco=str(source), inference_ann_path=str(predictions),
                output_dir=str(root / "gaps"), kpi="iter1"
            )
            for kind in ("loose", "strict"):
                (root / "gaps" / kind).mkdir(parents=True)
            report = write_gap_specs.analyze_binary(args, policy)
            self.assertEqual(report["loose"]["gap_counts"], {"FP": 1})
            self.assertEqual(report["strict"]["gap_counts"], {"FN": 1})
            strict = pd.read_parquet(root / "gaps" / "strict" / "box_gaps.parquet")
            self.assertEqual(strict.iloc[0]["filepath"], "/durable/one.png")


class GenericPoolNormalizationTest(unittest.TestCase):
    def test_mixed_shards_normalize_to_strict_pools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kpi_images = root / "kpi_images"
            test_images = root / "test_images"
            pool_images = root / "pool_images"
            shard = root / "mixed_bench"
            normalized = root / "normalized"
            for directory in (kpi_images, test_images, pool_images, shard, normalized):
                directory.mkdir()
            make_image(kpi_images / "kpi.png", 1)
            make_image(test_images / "test.png", 2)
            make_image(pool_images / "defect.png", 3)
            make_image(pool_images / "clean.png", 4)

            kpi_coco = root / "kpi.json"
            test_coco = root / "test.json"
            write_json(
                kpi_coco,
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "kpi.png",
                            "width": 64,
                            "height": 64,
                            "benchmark": "eval_bench",
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 7,
                            "bbox": [1, 1, 10, 10],
                        }
                    ],
                    "categories": [{"id": 7, "name": "scratch"}],
                },
            )
            write_json(
                test_coco,
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "test.png",
                            "width": 64,
                            "height": 64,
                            "benchmark": "eval_bench",
                        }
                    ],
                    "annotations": [],
                    "categories": [{"id": 7, "name": "scratch"}],
                },
            )
            write_json(
                shard / "train.json",
                {"images": [], "annotations": [], "categories": []},
            )
            write_json(
                shard / "mine.json",
                {
                    "images": [
                        {
                            "id": 10,
                            "file_name": "000001.png",
                            "source_path": str(pool_images / "defect.png"),
                            "width": 64,
                            "height": 64,
                        },
                        {
                            "id": 11,
                            "file_name": "000002.png",
                            "source_path": str(pool_images / "clean.png"),
                            "width": 64,
                            "height": 64,
                        },
                    ],
                    "annotations": [
                        {
                            "id": 5,
                            "image_id": 10,
                            "category_id": 9,
                            "bbox": [2, 2, 12, 12],
                            "defect_label": "scratch",
                        }
                    ],
                    "categories": [{"id": 9, "name": "defect"}],
                    "info": {"bench": "mixed_bench"},
                },
            )
            documents, report = normalize_deft_od_aoi_pools.normalize(
                argparse.Namespace(
                    kpi_coco=str(kpi_coco),
                    kpi_images_dir=str(kpi_images),
                    test_coco=str(test_coco),
                    test_images_dir=str(test_images),
                    pool_root=str(root),
                    pool_coco=None,
                    pool_splits="train,mine",
                    pool_images_dir=None,
                    output_dir=str(normalized),
                    check_only=False,
                )
            )
            self.assertEqual(report["outputs"]["source"], {"images": 1, "annotations": 1})
            self.assertEqual(report["outputs"]["clean"], {"images": 1, "annotations": 0})
            source_meta = documents["source"]["images"][0]["deft_od_aoi"]
            self.assertEqual(source_meta["benchmark"], "mixed_bench")
            self.assertEqual(source_meta["texture"], "mixed_bench")
            self.assertEqual(source_meta["defect_type"], "scratch")
            for name, value in documents.items():
                write_json(normalized / f"{name}.json", value)
            validation = validate_deft_od_aoi_inputs.run(
                argparse.Namespace(
                    kpi_coco=str(normalized / "kpi.json"),
                    kpi_images_dir=str(normalized),
                    test_coco=str(normalized / "test.json"),
                    test_images_dir=str(normalized),
                    source_coco=str(normalized / "source.json"),
                    source_images_dir=str(normalized),
                    clean_coco=str(normalized / "clean.json"),
                    clean_images_dir=str(normalized),
                    output=None,
                )
            )
            self.assertEqual(validation["status"], "valid")

            stage_report = route_deft_od_aoi_iteration._stage_coco(
                argparse.Namespace(
                    coco=str(normalized / "source.json"),
                    output_images=str(root / "staged_source"),
                    report=str(root / "stage_report.json"),
                )
            )
            self.assertEqual(stage_report["images"], 1)


class SourceManifestPreparationTest(unittest.TestCase):
    def test_strict_manifest_preserves_curated_clean_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "PlantA" / "housing"
            paths = {
                "kpi": dataset / "test" / "broken" / "kpi.png",
                "test": dataset / "test" / "broken" / "test.png",
                "train_defect": dataset / "test" / "broken" / "train_defect.png",
                "train_boxless": dataset / "test" / "good" / "train_good.png",
                "mine_defect": dataset / "test" / "broken" / "mine_defect.png",
                "mine_boxless": dataset / "test" / "good" / "mine_good.png",
                "external_clean": dataset / "train" / "good" / "external_good.png",
            }
            for seed, path in enumerate(paths.values(), start=1):
                path.parent.mkdir(parents=True, exist_ok=True)
                make_image(path, seed)

            def record(image_id: int, path: Path) -> dict:
                return {
                    "id": image_id,
                    "file_name": path.name,
                    "source_path": str(path),
                    "width": 64,
                    "height": 64,
                }

            box = {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 12, 12],
            }
            kpi_coco = root / "kpi.json"
            test_coco = root / "test.json"
            train_coco = root / "train.json"
            mine_coco = root / "mine.json"
            write_json(
                kpi_coco,
                coco([record(1, paths["kpi"])], [box]),
            )
            write_json(
                test_coco,
                coco([record(1, paths["test"])], [box]),
            )
            write_json(
                train_coco,
                coco(
                    [
                        record(1, paths["train_defect"]),
                        record(2, paths["train_boxless"]),
                    ],
                    [box],
                ),
            )
            write_json(
                mine_coco,
                coco(
                    [
                        record(1, paths["mine_defect"]),
                        record(2, paths["mine_boxless"]),
                    ],
                    [box],
                ),
            )
            manifest_path = root / "dataset_sources.json"
            write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "strict_metadata": True,
                    "metadata_rules": {
                        "plant_a": {
                            "defect_path_regex": (
                                r"/PlantA/(?P<texture>[^/]+)/test/"
                                r"(?P<defect_type>[^/]+)/"
                            ),
                            "clean_path_regex": (
                                r"/PlantA/(?P<texture>[^/]+)/(?:train|test)/good/"
                            ),
                            "generator_type_template": (
                                "{benchmark}_{texture}+{defect_type}"
                            ),
                        }
                    },
                    "inputs": {
                        "kpi": [{"benchmark": "plant_a", "coco": str(kpi_coco)}],
                        "test": [{"benchmark": "plant_a", "coco": str(test_coco)}],
                        "mining": [
                            {
                                "benchmark": "plant_a",
                                "coco": [str(train_coco), str(mine_coco)],
                                "boxless_clean_coco": [str(mine_coco)],
                            }
                        ],
                        "clean": [
                            {
                                "benchmark": "plant_a",
                                "glob": str(dataset / "train" / "good" / "*.png"),
                            }
                        ],
                    },
                },
            )
            documents, report = prepare_deft_od_aoi_sources.prepare(
                argparse.Namespace(
                    manifest=str(manifest_path),
                    output_dir=None,
                    check_only=True,
                    link_mode="symlink",
                )
            )
            self.assertEqual(report["outputs"]["source"], {"images": 2, "annotations": 2})
            self.assertEqual(report["outputs"]["clean"], {"images": 2, "annotations": 0})
            self.assertEqual(
                sum(item.get("boxless_ignored", 0) for item in report["mining_sources"]),
                1,
            )
            source_metadata = documents["source"]["images"][0]["deft_od_aoi"]
            self.assertEqual(source_metadata["texture"], "housing")
            self.assertEqual(source_metadata["defect_type"], "broken")
            self.assertEqual(source_metadata["generator_type"], "plant_a_housing+broken")
            self.assertFalse(
                any("fallback" in key for key in report["metadata_sources"])
            )
            clean_view = root / "anomalygen_clean"
            clean_counts = prepare_deft_od_aoi_sources._materialize_synthesis_clean_view(
                documents["clean"], clean_view, "symlink"
            )
            self.assertEqual(clean_counts, {"plant_a_housing": 2})
            self.assertEqual(
                len(list((clean_view / "plant_a_housing" / "clean_image").iterdir())),
                2,
            )
            prepare_deft_od_aoi_sources._materialize_synthesis_clean_view(
                documents["clean"], clean_view, "symlink"
            )

            inspection = inspect_deft_od_aoi_sources.inspect(
                argparse.Namespace(
                    dataset_path=[str(root)], max_depth=2, sample_paths=2, output=None
                )
            )
            discovered = inspection["datasets"][0]["coco_files"]
            self.assertEqual(len(discovered), 4)
            mine_summary = next(
                item for item in discovered if Path(item["path"]).name == "mine.json"
            )
            self.assertEqual(mine_summary["annotated_images"], 1)
            self.assertEqual(mine_summary["boxless_images"], 1)


class SiglipOnlyRoutingTest(unittest.TestCase):
    def test_admission_quarantines_nonpositive_boxes_before_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.png"
            make_image(image_path, 91)
            policy = build_policy(max_iterations=2, synthetic_enabled=False)
            admission = route_deft_od_aoi.Admission(policy, None)
            candidates = [
                {
                    "source_path": str(image_path),
                    "boxes": [[30, 30, -10, 12]],
                },
                {
                    "source_path": str(image_path),
                    "boxes": [[20, 100, 10, 10]],
                },
                {
                    "source_path": str(image_path),
                    "boxes": [[10, 10, 20, 20]],
                },
            ]

            admitted = admission.admit(candidates, 3, clean=False)

            self.assertEqual(len(admitted), 1)
            self.assertEqual(admitted[0]["boxes"], [[10.0, 10.0, 20.0, 20.0]])
            self.assertEqual(admission.report["boxes_quarantined"], 2)
            self.assertEqual(admission.report["rejected_no_valid_boxes"], 2)

    def test_admission_does_not_report_roundoff_as_boundary_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.png"
            make_image(image_path, 92)
            policy = build_policy(max_iterations=2, synthetic_enabled=False)
            admission = route_deft_od_aoi.Admission(policy, None)
            box = [0.1, 0.2, 20.3, 21.4]

            admitted = admission.admit(
                [{"source_path": str(image_path), "boxes": [box]}],
                1,
                clean=False,
            )

            self.assertEqual(admitted[0]["boxes"], [box])
            self.assertEqual(admission.report["boxes_clipped_to_image"], 0)

    def test_candidate_crops_apply_exif_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            source_image = images / "portrait.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (40, 20), (20, 40, 60)).save(source_image, exif=exif)
            source_coco = root / "source.json"
            clean_coco = root / "clean.json"
            record = image_record(1, source_image.name)
            record.update({"source_path": str(source_image), "width": 20, "height": 40})
            write_json(
                source_coco,
                coco(
                    [record],
                    [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 25, 8, 8]}],
                ),
            )
            write_json(clean_coco, coco([], []))
            frame, report = prepare_deft_od_aoi_siglip_candidates.prepare(
                argparse.Namespace(
                    source_coco=str(source_coco),
                    source_images_dir=str(images),
                    clean_coco=str(clean_coco),
                    clean_images_dir=str(images),
                    output_crops_dir=str(root / "crops"),
                    output_parquet=str(root / "candidate.parquet"),
                    report_json=str(root / "report.json"),
                    defect_context_scale=1.5,
                    clean_grids="1,2",
                    output_size=224,
                )
            )
            self.assertEqual(len(frame), 1)
            self.assertEqual(report["rows"], {"defect": 1})

    def test_global_role_separated_retrieval_crosses_dataset_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_images = root / "source"
            clean_images = root / "clean"
            kpi_images = root / "kpi"
            for directory in (source_images, clean_images, kpi_images):
                directory.mkdir()
            paths = [
                source_images / "plant_a_defect.png",
                source_images / "poolx_defect.png",
                clean_images / "plant_a_clean.png",
                clean_images / "poolx_clean.png",
                kpi_images / "kpi.png",
            ]
            for seed, path in enumerate(paths, start=1):
                make_image(path, seed)

            source_coco = root / "source.json"
            clean_coco = root / "clean.json"
            kpi_coco = root / "kpi.json"
            source_rows = [
                {**image_record(1, "plant_a_defect.png"), "source_path": str(paths[0])},
                {**image_record(2, "poolx_defect.png"), "source_path": str(paths[1])},
            ]
            source_rows[0]["deft_od_aoi"]["benchmark"] = "plant_a"
            source_rows[1]["deft_od_aoi"]["benchmark"] = "poolx_neu"
            write_json(
                source_coco,
                coco(
                    source_rows,
                    [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 20, 20]},
                        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [8, 8, 20, 20]},
                    ],
                ),
            )
            clean_rows = [
                {**image_record(1, "plant_a_clean.png"), "source_path": str(paths[2])},
                {**image_record(2, "poolx_clean.png"), "source_path": str(paths[3])},
            ]
            for row, benchmark in zip(clean_rows, ("plant_a", "poolx_solder")):
                row["deft_od_aoi"]["benchmark"] = benchmark
                row["deft_od_aoi"].pop("defect_type")
                row["deft_od_aoi"].pop("generator_type")
            write_json(clean_coco, coco(clean_rows, []))
            kpi_row = {**image_record(1, "kpi.png"), "source_path": str(paths[4])}
            kpi_row["deft_od_aoi"]["benchmark"] = "plant_a"
            write_json(
                kpi_coco,
                coco(
                    [kpi_row],
                    [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 20, 20]}],
                ),
            )

            candidate_frame, candidate_report = prepare_deft_od_aoi_siglip_candidates.prepare(
                argparse.Namespace(
                    source_coco=str(source_coco),
                    source_images_dir=str(source_images),
                    clean_coco=str(clean_coco),
                    clean_images_dir=str(clean_images),
                    output_crops_dir=str(root / "candidate_crops"),
                    output_parquet=str(root / "candidate_rows.parquet"),
                    report_json=str(root / "candidate_report.json"),
                    defect_context_scale=1.5,
                    clean_grids="1,2",
                    output_size=224,
                )
            )
            self.assertEqual(candidate_report["rows"], {"clean": 10, "defect": 2})

            strict_path = root / "strict.parquet"
            loose_path = root / "loose.parquet"
            pd.DataFrame(
                [{"gap_type": "FN", "filepath": "kpi.png", "bbox": [8, 8, 28, 28], "best_iou": 0.0}]
            ).to_parquet(strict_path)
            pd.DataFrame(
                [{"gap_type": "FP", "filepath": "kpi.png", "bbox": [35, 35, 55, 55], "best_iou": 0.01}]
            ).to_parquet(loose_path)
            query_frame, query_report = prepare_deft_od_aoi_siglip_queries.prepare(
                argparse.Namespace(
                    strict_gaps=str(strict_path),
                    loose_gaps=str(loose_path),
                    kpi_coco=str(kpi_coco),
                    kpi_images_dir=str(kpi_images),
                    output_crops_dir=str(root / "query_crops"),
                    output_parquet=str(root / "query_rows.parquet"),
                    report_json=str(root / "query_report.json"),
                    context_scale=1.5,
                    output_size=224,
                    background_iou_upper=0.05,
                    near_miss_iou_upper=0.5,
                )
            )
            self.assertEqual(query_report["queries"], {"strict_fn": 1, "background_fp": 1})

            def candidate_embedding(row: pd.Series) -> list[float]:
                if row["role"] == "defect":
                    return [1.0, 0.0] if row["benchmark"] == "poolx_neu" else [-1.0, 0.0]
                return [0.0, 1.0] if row["benchmark"] == "poolx_solder" else [0.0, -1.0]

            candidate_frame["embedding"] = candidate_frame.apply(candidate_embedding, axis=1)
            query_frame["embedding"] = query_frame["role"].map(
                {"defect": [1.0, 0.0], "clean": [0.0, 1.0]}
            )
            candidate_embeddings = root / "candidate_embeddings.parquet"
            query_embeddings = root / "query_embeddings.parquet"
            candidate_frame.to_parquet(candidate_embeddings, index=False)
            query_frame.to_parquet(query_embeddings, index=False)
            policy_path = root / "policy.json"
            write_json(policy_path, build_policy(max_iterations=3, synthetic_enabled=False))
            routing_root = root / "iteration_routing"
            state = route_deft_od_aoi_iteration._prepare(
                argparse.Namespace(
                    policy=str(policy_path),
                    iteration=1,
                    output_root=str(routing_root),
                    strict_gaps=str(strict_path),
                    loose_gaps=str(loose_path),
                    kpi_coco=str(kpi_coco),
                    kpi_images_dir=str(kpi_images),
                    embedding_batch_size=8,
                    runtime_model_path=None,
                )
            )
            self.assertEqual(state["status"], "PREPARED")
            wrapper_queries = pd.read_parquet(state["query_inputs"])
            wrapper_queries["embedding"] = wrapper_queries["role"].map(
                {"defect": [1.0, 0.0], "clean": [0.0, 1.0]}
            )
            wrapper_queries.to_parquet(state["query_embeddings"], index=False)
            finalized_embeddings = routing_root / "queries" / "query_embeddings_final.parquet"
            finalize_report = route_deft_od_aoi_iteration._finalize_embeddings(
                argparse.Namespace(
                    embedding_parquet=state["query_embeddings"],
                    durable_output=str(finalized_embeddings),
                    published_output=None,
                    report=str(routing_root / "queries" / "embedding_finalize_report.json"),
                    routing_state=str(routing_root / "routing_state.json"),
                )
            )
            self.assertEqual(finalize_report["rows"], 2)
            self.assertNotIn("filepath", pd.read_parquet(finalized_embeddings).columns)
            report = route_deft_od_aoi_iteration._commit(
                argparse.Namespace(
                    policy=str(policy_path),
                    iteration=1,
                    output_root=str(routing_root),
                    candidate_embeddings=str(candidate_embeddings),
                    query_embeddings=str(finalized_embeddings),
                    kpi_coco=str(kpi_coco),
                    source_coco=str(source_coco),
                    source_images_dir=str(source_images),
                    clean_coco=str(clean_coco),
                    clean_images_dir=str(clean_images),
                    previous_defect_ledger=None,
                    previous_clean_ledger=None,
                    previous_admission_index=None,
                    conversion_old_strict=None,
                    conversion_new_strict=None,
                    prior_admitted_synthetic=0,
                    valid_generator_types=None,
                )
            )
            self.assertEqual(report["uniform_mine_per_pocket"], 0)
            self.assertEqual(report["selected_by_benchmark"]["poolx_neu"], 1)
            self.assertEqual(report["selected_by_benchmark"]["poolx_solder"], 1)
            self.assertTrue((routing_root / "routing" / "retrieval_audit.parquet").is_file())
            committed = json.loads((routing_root / "routing_state.json").read_text())
            self.assertEqual(committed["status"], "COMPLETE")


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
                max_iterations=3,
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
                previous_assembled_coco=None,
                route_manifest=[str(route_dir / "mined_manifest.json")],
                current_routing_report=str(route_dir / "routing_report.json"),
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
                published_output_dir=None,
                results_root=str(train_results),
                incumbent_config=None,
                history=None,
                training_workers=0,
                inference_workers=0,
            )
        )
        self.assertEqual(manifest["runtime_overrides"], {"training_workers": 0, "inference_workers": 0})
        main_spec = yaml.safe_load((specs_dir / "train.yaml").read_text(encoding="utf-8"))
        self.assertEqual(main_spec["dataset"]["workers"], 0)
        self.assertEqual(len(manifest["probes"]), 3)
        for split in ("kpi", "test"):
            inference_spec = yaml.safe_load(
                (specs_dir / f"{split}_infer.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(inference_spec["dataset"]["eval_class_ids"], [1])
            inference_classmap = Path(
                inference_spec["dataset"]["infer_data_sources"]["classmap"]
            )
            self.assertEqual(
                inference_classmap.read_text(encoding="utf-8"),
                "background\ndefect\n",
            )
        staged_specs = self.root / "staged_specs"
        published_specs = self.root / "published_specs"
        write_rtdetr_specs.run(
            argparse.Namespace(
                policy=str(policy_path), iteration=2, train_coco=str(output_coco),
                train_images_dir=str(self.root / "train" / "images"),
                kpi_coco=str(self.kpi_coco), kpi_images_dir=str(self.kpi_images),
                test_images_dir=str(self.test_images), classmap=str(classmap),
                base_checkpoint=str(base_checkpoint), output_dir=str(staged_specs),
                published_output_dir=str(published_specs), results_root=str(train_results),
                incumbent_config=None, history=None, training_workers=0,
                inference_workers=0,
            )
        )
        staged_infer = yaml.safe_load((staged_specs / "kpi_infer.yaml").read_text())
        self.assertEqual(
            staged_infer["dataset"]["infer_data_sources"]["classmap"],
            os.path.abspath(published_specs / "inference_classmap.txt"),
        )
        self.assertNotIn(str(staged_specs.resolve()), json.dumps(staged_infer))
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

        extension_status = self.root / "extension_status.json"
        extension_status.write_text(
            json.dumps({"epoch": 48, "kpi": {"val_mAP50": 0.45}}) + "\n",
            encoding="utf-8",
        )
        combined_selection = select_deft_od_aoi_checkpoint.run(
            argparse.Namespace(
                policy=str(policy_path),
                status=[str(main_status), str(extension_status)],
                checkpoint_dir=str(checkpoint_dir),
                planned_epochs=60,
                extension_applied=True,
                output=str(self.root / "combined_selection.json"),
            )
        )
        self.assertEqual(combined_selection["action"], "select")
        self.assertEqual(combined_selection["best_epoch"], 47)
        self.assertEqual(len(combined_selection["status_files"]), 2)

        terminal = checkpoint_dir / "model_epoch_047.pth"
        extension_path = specs_dir / "train_extension.yaml"
        extension = prepare_rtdetr_extension_spec.run(
            argparse.Namespace(
                input=str(specs_dir / "train.yaml"),
                output=str(extension_path),
                resume_checkpoint=str(terminal),
                expected_epochs=48,
                extended_epochs=60,
            )
        )
        self.assertEqual(extension["extended_num_epochs"], 60)
        extension_spec = yaml.safe_load(extension_path.read_text(encoding="utf-8"))
        self.assertEqual(extension_spec["train"]["num_epochs"], 60)
        self.assertEqual(
            extension_spec["train"]["resume_training_checkpoint_path"],
            str(terminal.resolve()),
        )
        self.assertEqual(extension_spec["dataset"]["workers"], 0)
        with self.assertRaisesRegex(ValueError, "terminal checkpoint"):
            prepare_rtdetr_extension_spec.run(
                argparse.Namespace(
                    input=str(specs_dir / "train.yaml"),
                    output=str(extension_path),
                    resume_checkpoint=str(checkpoint_dir / "model_epoch_045.pth"),
                    expected_epochs=48,
                    extended_epochs=60,
                )
            )

        measurement_dir = self.root / "measurement_staging"
        published_measurement_dir = specs_dir
        measurement_selection = self.root / "measurement_selection.json"
        write_json(
            measurement_selection,
            {
                "action": "select",
                "best_epoch": 47,
                "best_kpi_mAP50": 0.5,
                "selected_checkpoint": str(terminal.resolve()),
                "planned_epochs": 60,
                "extension_applied": True,
            },
        )
        measurement = prepare_rtdetr_measurement_specs.run(
            argparse.Namespace(
                train_spec=str(specs_dir / "train.yaml"),
                kpi_infer_spec=str(specs_dir / "kpi_infer.yaml"),
                test_infer_spec=str(specs_dir / "test_infer.yaml"),
                selection=str(measurement_selection),
                kpi_coco=str(self.kpi_coco),
                test_coco=str(self.test_coco),
                output_dir=str(measurement_dir),
                published_output_dir=str(published_measurement_dir),
                results_root=str(train_results),
            )
        )
        self.assertNotIn(str(measurement_dir.resolve()), json.dumps(measurement))
        self.assertEqual(
            measurement["specs"]["kpi"]["evaluate"],
            os.path.abspath(published_measurement_dir / "kpi_evaluate_selected.yaml"),
        )

        disabled_policy = self.root / "policy_no_probes.json"
        write_json(
            disabled_policy,
            build_policy(
                max_iterations=3,
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
                training_workers=None,
                inference_workers=None,
            )
        )
        self.assertEqual(skipped["probes"], [])
        skipped_train = (self.root / "specs_no_probes" / "train.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("num_epochs: 48", skipped_train)


if __name__ == "__main__":
    unittest.main()
