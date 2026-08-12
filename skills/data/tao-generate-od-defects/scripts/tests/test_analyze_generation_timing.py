#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_generation_timing.py"
SPEC = importlib.util.spec_from_file_location("analyze_generation_timing", SCRIPT)
assert SPEC and SPEC.loader
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


class AnalyzeGenerationTimingTest(unittest.TestCase):
    def test_groups_rows_and_separates_first_row_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "generation.log"
            log.write_text(
                "[08-12 10:00:00|job=|INFO|x:1:generate_samples_from_batch] Using sampler: UniPC\n"
                "[08-12 10:00:09|job=|INFO|x:1:generate_samples_from_batch] Using sampler: UniPC\n"
                "[08-12 10:00:12|job=|INFO|x/generate.py:612:main] "
                "SDG done: 2 images across 1 rank(s) in 12.0s generation wall-time.\n"
            )
            provenance = root / "provenance.jsonl"
            provenance.write_text(
                json.dumps(
                    {
                        "generation_index": 0,
                        "dataset_id": "toy",
                        "anomaly_type": "toy+a",
                        "mask_branch": "fn_mask",
                        "fn_id": "fn-1",
                        "pair_id": "pair-1",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "generation_index": 1,
                        "dataset_id": "toy",
                        "anomaly_type": "toy+b",
                        "mask_branch": "fn_mask",
                        "fn_id": "fn-2",
                        "pair_id": "pair-2",
                    }
                )
                + "\n"
            )
            defect_spec = root / "defect_spec.jsonl"
            defect_spec.write_text(
                json.dumps({"defect_type": "toy+a", "spatial_dependency": "free"})
                + "\n"
                + json.dumps({"defect_type": "toy+b", "spatial_dependency": "text"})
                + "\n"
            )
            generation_root = root / "raw"
            reconstructed = generation_root / "reconstructed_image"
            reconstructed.mkdir(parents=True)
            first = reconstructed / "out0.png"
            second = reconstructed / "out1.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            generation_csv = generation_root / "texture_ft_generation_result.csv"
            generation_csv.write_text(
                "output_filename,index\nout0.png,0\nout1.png,1\n"
            )
            os.utime(first, (97.0, 97.0))
            os.utime(second, (100.0, 100.0))
            os.utime(generation_csv, (100.0, 100.0))

            rows, summary = timing.analyze(log, generation_csv, provenance, defect_spec)

            self.assertEqual([row["seconds_to_output"] for row in rows], [9.0, 3.0])
            self.assertTrue(rows[0]["warmup_row"])
            self.assertFalse(rows[1]["warmup_row"])
            self.assertEqual(
                summary["by_spatial_dependency"]["free"]["including_warmup"]["count"], 1
            )
            self.assertIsNone(
                summary["by_spatial_dependency"]["free"]["excluding_first_job_row"][
                    "mean_seconds"
                ]
            )
            self.assertEqual(
                summary["by_spatial_dependency"]["text"]["excluding_first_job_row"][
                    "mean_seconds"
                ],
                3.0,
            )


if __name__ == "__main__":
    unittest.main()
