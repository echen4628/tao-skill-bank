#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for AnomalyGenNext input preparation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "prepare_anomalygennext_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_anomalygennext_inputs", SCRIPT)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def write_image(path: Path, value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def write_mask(path: Path, offset: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[offset : offset + 8, offset : offset + 8] = 255
    Image.fromarray(mask).save(path)


def write_full_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(path)


class PipelineContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "prepared-inputs-run"
        self.generation_run = self.root / "generation-run"
        self.source = self.root / "source" / "Toy" / "widget"
        self.fn_image = self.source / "test" / "scratch" / "001.png"
        self.fn_mask = self.source / "ground_truth" / "scratch" / "001_mask.png"
        write_image(self.fn_image)
        write_mask(self.fn_mask)

        pool = self.root / "pool" / "toy_widget"
        for index, value in enumerate((10, 20, 30)):
            write_image(pool / "clean_image" / f"clean{index}.png", value)
        for index in range(2):
            write_mask(pool / "mask" / "scratch" / f"sample{index}.png", 4 + index)

        split = self.root / "splits" / "toy"
        split.mkdir(parents=True)
        (split / "split_manifest.json").write_text(
            json.dumps(
                {
                    "1": {
                        "bucket": "kpi",
                        "source_path": str(self.fn_image),
                        "category": "widget",
                        "defective": True,
                    }
                }
            )
        )
        gaps = pd.DataFrame(
            [
                {
                    "image_id": "0001",
                    "filepath": str(self.fn_image),
                    "gap_type": "FN",
                    "bbox": np.asarray([8, 8, 15, 15], dtype=float),
                    "class": "defect",
                }
            ]
        )
        gaps.to_parquet(self.root / "gaps.parquet", index=False)
        recipe = {
            "dataset_name": "toy",
            "anomaly_types": [["toy_widget", "scratch"]],
        }
        (self.root / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        (self.root / "defect_spec.jsonl").write_text(
            json.dumps(
                {
                    "defect_type": "toy_widget+scratch",
                    "spatial_dependency": "text",
                    "roi_prompt_defect_location": "on the widget",
                }
            )
            + "\n"
        )
        checkpoint = self.root / "iter_000001000.pt"
        checkpoint.write_bytes(b"checkpoint")
        self.config = self.root / "config.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "source_tag": "test_fixture",
                    "gap_parquet": str(self.root / "gaps.parquet"),
                    "split_root": str(self.root / "splits"),
                    "pool_dataset_root": str(self.root / "pool"),
                    "defect_spec": str(self.root / "defect_spec.jsonl"),
                    "datasets": {
                        "toy": {
                            "path_marker": "Toy",
                            "texture_prefix": "toy_",
                            "mask_style": "stem_mask_png",
                            "checkpoint": str(checkpoint),
                            "recipe": str(self.root / "recipe.yaml"),
                        }
                    },
                    "selection": {
                        "datasets": ["toy"],
                        "split": "kpi",
                        "per_dataset": 1,
                        "mask_sample_seed": 42,
                    },
                    "embedding": {
                        "model": "SigLIP",
                        "model_path": "google/siglip-base-patch16-224",
                        "batch_size": 8,
                    },
                    "retrieval": {
                        "metric": "cosine",
                        "candidate_topn": 3,
                        "max_neighbors_per_fn": 2,
                        "min_similarity": 0.9,
                        "prior_clean_exclusion_manifest": "",
                    },
                },
                sort_keys=False,
            )
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run_preparation(self, full_mask_clean_stem: str | None = None) -> None:
        pipeline.prepare_inputs(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )
        clean = pd.read_parquet(self.run / "manifests" / "clean_pool.parquet")
        query = pd.read_parquet(self.run / "manifests" / "fn_embedding_inputs.parquet")
        clean["embedding"] = [
            np.asarray([1.0, 0.00]),
            np.asarray([0.99, 0.01]),
            np.asarray([0.98, 0.02]),
        ]
        query["embedding"] = [np.asarray([1.0, 0.0])]
        clean.to_parquet(self.run / "embeddings" / "clean_embeddings.parquet", index=False)
        query.to_parquet(self.run / "embeddings" / "fn_embeddings.parquet", index=False)
        pipeline.build_knn_and_amp(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )
        amp_inputs = json.loads((self.run / "amp" / "amp_samples.json").read_text())
        amp_rows = []
        for row in amp_inputs:
            output = (
                self.run
                / "amp"
                / Path(row["clean_image"]).stem
                / row["defect_type"]
                / f"{Path(row['submask']).stem}__seed0.png"
            )
            if Path(row["clean_image"]).stem == full_mask_clean_stem:
                write_full_mask(output)
            else:
                write_mask(output)
            amp_rows.append(
                {
                    "image_filename": row["clean_image"],
                    "mask_filename": str(output),
                    "anomaly_type": row["defect_type"],
                    "crop_and_paste": True,
                    "num_generated_images": 1,
                    "poisson_blend": False,
                    "iteration_generation_max_instance": 5,
                    "guidance": 6.0,
                    "num_steps": 35,
                    "crop_ratio": 2.0,
                }
            )
        pipeline._write_jsonl(self.run / "amp" / "testcase.jsonl", amp_rows)
        pipeline.finalize_inputs(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )

    def test_prepared_inputs_freezes_exact_generator_inputs(self) -> None:
        self._run_preparation()
        frozen_config = (
            self.run / "prepared_anomalygennext_inputs" / "filtering_config.yaml"
        )
        self.assertTrue(frozen_config.is_file())
        contract = json.loads(
            (self.run / "prepared_anomalygennext_inputs" / "input_contract.json").read_text()
        )
        self.assertEqual(contract["filtering_config"], str(frozen_config))
        manifest = json.loads((self.run / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json").read_text())
        self.assertEqual(manifest["source_tag"], "test_fixture")
        self.assertEqual(manifest["selected_fn_count"], 1)
        self.assertEqual(manifest["selected_pair_count"], 2)
        self.assertEqual(manifest["generator_row_count"], 4)
        self.assertEqual(manifest["generator_groups"][0]["anomaly_types"], ["toy_widget+scratch"])
        rows = pipeline._read_jsonl(
            self.run / "prepared_anomalygennext_inputs" / "anomalygen_inputs" / "toy" / "testcase.jsonl"
        )
        self.assertEqual(len(rows), 4)
        pipeline.validate_prepared_inputs(type("Args", (), {"prepared_inputs_root": str(self.run)})())

    def test_full_image_amp_mask_rejects_candidate_and_uses_next_neighbor(self) -> None:
        self._run_preparation(full_mask_clean_stem="clean0")
        selected = pd.read_parquet(self.run / "manifests" / "selected_pairs.parquet")
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {Path(path).stem for path in selected["clean_filepath"]},
            {"clean1", "clean2"},
        )
        status = pd.read_parquet(self.run / "manifests" / "knn_roi_status.parquet")
        rejected = status[status["clean_filepath"].map(lambda path: Path(path).stem == "clean0")]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected.iloc[0]["selection_reason"],
            "invalid_aligned_mask:fn_mask:full_image",
        )

    def test_source_tag_is_optional_provenance(self) -> None:
        config = yaml.safe_load(self.config.read_text())
        config.pop("source_tag")
        self.config.write_text(yaml.safe_dump(config))
        loaded = pipeline._load_config(self.config)
        self.assertEqual(loaded["source_tag"], "user_provided")

    def test_text_defect_requires_placement_prompt(self) -> None:
        (self.root / "defect_spec.jsonl").write_text(
            json.dumps(
                {
                    "defect_type": "toy_widget+scratch",
                    "spatial_dependency": "text",
                    "roi_prompt_defect_location": "",
                }
            )
            + "\n"
        )
        with self.assertRaisesRegex(ValueError, "roi_prompt_defect_location"):
            pipeline.prepare_inputs(
                type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
            )

    def test_prepared_inputs_all_eligible_selects_across_split_buckets(self) -> None:
        second_image = self.source / "test" / "scratch" / "002.png"
        second_mask = self.source / "ground_truth" / "scratch" / "002_mask.png"
        write_image(second_image, 96)
        write_mask(second_mask, 10)
        split_path = self.root / "splits" / "toy" / "split_manifest.json"
        split_rows = json.loads(split_path.read_text())
        split_rows["2"] = {
            "bucket": "train",
            "source_path": str(second_image),
            "category": "widget",
            "defective": True,
        }
        split_path.write_text(json.dumps(split_rows))
        gaps = pd.read_parquet(self.root / "gaps.parquet")
        gaps = pd.concat(
            [
                gaps,
                pd.DataFrame(
                    [
                        {
                            "image_id": "0002",
                            "filepath": str(second_image),
                            "gap_type": "FN",
                            "bbox": np.asarray([10, 10, 17, 17], dtype=float),
                            "class": "defect",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        gaps.to_parquet(self.root / "gaps.parquet", index=False)
        config = yaml.safe_load(self.config.read_text())
        config["selection"] = {
            "mode": "all_eligible",
            "datasets": ["toy"],
            "mask_sample_seed": 42,
        }
        self.config.write_text(yaml.safe_dump(config, sort_keys=False))

        pipeline.prepare_inputs(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )

        ledger = pd.read_parquet(self.run / "manifests" / "fn_queries.parquet")
        selected = ledger[ledger["selected_for_smoke"]]
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["split"]), {"kpi", "train"})

    def test_multiple_fn_boxes_on_one_image_keep_distinct_queries(self) -> None:
        gaps = pd.read_parquet(self.root / "gaps.parquet")
        second = gaps.iloc[0].copy()
        second["bbox"] = np.asarray([9, 9, 15, 15], dtype=float)
        gaps = pd.concat([gaps, pd.DataFrame([second])], ignore_index=True)
        gaps.to_parquet(self.root / "gaps.parquet", index=False)
        config = yaml.safe_load(self.config.read_text())
        config["selection"] = {
            "mode": "all_eligible",
            "datasets": ["toy"],
            "mask_sample_seed": 42,
        }
        self.config.write_text(yaml.safe_dump(config, sort_keys=False))

        pipeline.prepare_inputs(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )
        embedding_inputs = pd.read_parquet(
            self.run / "manifests" / "fn_embedding_inputs.parquet"
        )
        selected_queries = pd.read_parquet(
            self.run / "manifests" / "selected_fn_queries.parquet"
        )
        self.assertEqual(len(embedding_inputs), 1)
        self.assertEqual(selected_queries["fn_id"].nunique(), 2)

        clean = pd.read_parquet(self.run / "manifests" / "clean_pool.parquet")
        clean["embedding"] = [
            np.asarray([1.0, 0.00]),
            np.asarray([0.99, 0.01]),
            np.asarray([0.98, 0.02]),
        ]
        embedding_inputs["embedding"] = [np.asarray([1.0, 0.0])]
        clean.to_parquet(self.run / "embeddings" / "clean_embeddings.parquet", index=False)
        embedding_inputs.to_parquet(
            self.run / "embeddings" / "fn_embeddings.parquet", index=False
        )
        pipeline.build_knn_and_amp(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )
        candidates = pd.read_parquet(self.run / "manifests" / "knn_candidates.parquet")
        self.assertEqual(candidates["fn_id"].nunique(), 2)
        self.assertEqual(set(candidates["od_category"]), {"defect"})

        amp_rows = []
        for row in json.loads((self.run / "amp" / "amp_samples.json").read_text()):
            output = (
                self.run
                / "amp"
                / Path(row["clean_image"]).stem
                / row["defect_type"]
                / f"{Path(row['submask']).stem}__seed0.png"
            )
            write_mask(output)
            amp_rows.append(
                {
                    "image_filename": row["clean_image"],
                    "mask_filename": str(output),
                    "anomaly_type": row["defect_type"],
                }
            )
        pipeline._write_jsonl(self.run / "amp" / "testcase.jsonl", amp_rows)
        pipeline.finalize_inputs(
            type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
        )
        selected = pd.read_parquet(self.run / "manifests" / "selected_pairs.parquet")
        clean_by_fn = {
            fn_id: set(group["clean_filepath"])
            for fn_id, group in selected.groupby("fn_id")
        }
        self.assertEqual(len(clean_by_fn), 2)
        self.assertTrue(set.intersection(*clean_by_fn.values()))
        provenance = pipeline._read_jsonl(
            self.run / "prepared_anomalygennext_inputs" / "anomalygen_inputs" / "toy" / "provenance.jsonl"
        )
        self.assertEqual({row["od_category"] for row in provenance}, {"defect"})

    def test_prepared_inputs_rejects_unknown_selection_mode(self) -> None:
        config = yaml.safe_load(self.config.read_text())
        config["selection"]["mode"] = "surprise"
        self.config.write_text(yaml.safe_dump(config, sort_keys=False))
        with self.assertRaisesRegex(ValueError, "unsupported selection.mode"):
            pipeline.prepare_inputs(
                type("Args", (), {"config": str(self.config), "run_root": str(self.run)})()
            )

    def test_type_and_mask_supports_component_replacement_and_fixed_class(self) -> None:
        image = self.root / "TextureSet" / "images" / "ClassA" / "Train" / "0034.PNG"
        expected_mask = (
            self.root / "TextureSet" / "masks" / "ClassA" / "Train" / "0034_label.PNG"
        )
        write_image(image)
        write_mask(expected_mask)
        texture, defect_class, mask = pipeline._type_and_mask(
            str(image),
            "texture_set",
            {
                "path_marker": "TextureSet",
                "texture_offset": 2,
                "texture_prefix": "texture_set_",
                "split_components": ["Test", "Train"],
                "defect_class_fixed": "defect",
                "mask_style": "configurable",
                "mask_component_replacements": {"images": "masks"},
                "mask_suffix": "_label",
                "mask_extensions": [".PNG"],
            },
        )
        self.assertEqual(texture, "texture_set_ClassA")
        self.assertEqual(defect_class, "defect")
        self.assertEqual(mask, expected_mask)

    def test_prepared_inputs_gate_rejects_changed_anomalygen_next_generation_plan(self) -> None:
        self._run_preparation()
        plan_path = self.run / "prepared_anomalygennext_inputs" / "anomalygen_next_generation_plan.json"
        plan_path.write_text(plan_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            pipeline.validate_prepared_inputs(
                type("Args", (), {"prepared_inputs_root": str(self.run)})()
            )

    def test_embedding_stage_and_restore_preserve_durable_paths(self) -> None:
        source = self.root / "embedding_input.parquet"
        pd.DataFrame([{"filepath": str(self.fn_image), "token": "keep"}]).to_parquet(source)
        source_spec = self.root / "embedding.yaml"
        source_spec.write_text(
            yaml.safe_dump(
                {
                    "input_parquet": str(source),
                    "output_parquet": str(self.root / "durable.parquet"),
                    "model": "SigLIP",
                    "model_path": "remote-model",
                    "batch_size": 1,
                }
            )
        )
        stage = self.root / "scratch"
        pipeline.stage_embedding(
            type(
                "Args",
                (),
                {
                    "source_spec": str(source_spec),
                    "stage_root": str(stage),
                    "model_path": str(self.root / "local-model"),
                    "local_output": str(stage / "embeddings.parquet"),
                    "output_spec": str(stage / "spec.yaml"),
                    "mapping": str(stage / "mapping.parquet"),
                },
            )()
        )
        local = pd.read_parquet(stage / "input.parquet")
        local["embedding"] = [np.asarray([1.0, 0.0])]
        local.to_parquet(stage / "embeddings.parquet")
        restored = self.root / "restored.parquet"
        pipeline.restore_embedding(
            type(
                "Args",
                (),
                {
                    "embedding_parquet": str(stage / "embeddings.parquet"),
                    "mapping": str(stage / "mapping.parquet"),
                    "durable_output": str(restored),
                },
            )()
        )
        self.assertEqual(pd.read_parquet(restored).iloc[0]["filepath"], str(self.fn_image))

    def test_materialize_aoi_plan_selects_fn_rows_from_complete_gap_table(self) -> None:
        image = self.root / "MVTec-AD" / "widget" / "test" / "scratch" / "001.png"
        write_image(image)
        coco = self.root / "kpi.json"
        coco.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "source_path": str(image),
                            "deft_od_aoi": {"generator_type": "mvtec_widget+scratch"},
                        }
                    ],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "defect"}],
                }
            )
        )
        strict = self.root / "strict.parquet"
        pd.DataFrame(
            [
                {"image_id": 1, "gap_type": "TP", "bbox": [1, 1, 4, 4], "class": "defect"},
                {"image_id": 1, "gap_type": "FN", "bbox": [8, 8, 15, 15], "class": "defect"},
                {"image_id": 1, "gap_type": "FP", "bbox": [20, 20, 4, 4], "class": "defect"},
            ]
        ).to_parquet(strict, index=False)
        plan = self.root / "synthetic_plan.json"
        plan.write_text(json.dumps({"mvtec_widget+scratch": 2}))
        output = self.root / "materialized"
        pipeline.materialize_aoi_plan(
            type(
                "Args",
                (),
                {
                    "iteration": 2,
                    "kpi_coco": str(coco),
                    "strict_gaps": str(strict),
                    "synthetic_plan": str(plan),
                    "output_root": str(output),
                    "pool_root": str(self.root / "pool"),
                    "defect_spec": str(self.root / "defect_spec.jsonl"),
                    "checkpoint": str(self.root / "iter_000001000.pt"),
                    "recipe": str(self.root / "recipe.yaml"),
                    "embedding_model_path": str(self.root / "model"),
                    "embedding_batch_size": 256,
                    "candidate_topn": 3,
                    "min_similarity": -1.0,
                    "mask_sample_seed": 43,
                },
            )()
        )
        report = json.loads((output / "selection_report.json").read_text())
        subset = pd.read_parquet(output / "strict_fn_synthesis_subset.parquet")
        self.assertEqual(report["input_gap_rows"], 3)
        self.assertEqual(report["input_fn_rows"], 1)
        self.assertEqual(len(subset), 1)
        self.assertTrue(subset["gap_type"].astype(str).str.upper().eq("FN").all())

    def test_restore_amp_rewrites_runtime_paths_without_hidden_state(self) -> None:
        local_clean = self.root / "scratch" / "clean.png"
        durable_clean = self.root / "durable" / "clean.png"
        local_amp = self.root / "scratch" / "amp"
        durable_amp = self.root / "durable" / "amp"
        local_mask = local_amp / "toy" / "mask.png"
        durable_mask = durable_amp / "toy" / "mask.png"
        write_image(local_clean)
        write_image(durable_clean)
        write_mask(local_mask)
        write_mask(durable_mask)
        source = self.root / "runtime_testcase.jsonl"
        source.write_text(
            json.dumps(
                {
                    "image_filename": str(local_clean.resolve()),
                    "mask_filename": str(local_mask.resolve()),
                    "anomaly_type": "toy_widget+scratch",
                }
            )
            + "\n"
        )
        mapping = self.root / "amp_mapping.json"
        mapping.write_text(
            json.dumps(
                {
                    "clean": {
                        str(local_clean.resolve()): str(durable_clean.resolve())
                    },
                    "rows": 1,
                }
            )
        )
        output = self.root / "durable_testcase.jsonl"
        pipeline.restore_amp(
            type(
                "Args",
                (),
                {
                    "mapping_json": str(mapping),
                    "local_amp_root": str(local_amp),
                    "durable_amp_root": str(durable_amp),
                    "source_testcase": str(source),
                    "durable_testcase": str(output),
                },
            )()
        )
        row = json.loads(output.read_text())
        self.assertEqual(row["image_filename"], str(durable_clean.resolve()))
        self.assertEqual(row["mask_filename"], str(durable_mask.resolve()))

if __name__ == "__main__":
    unittest.main()
