#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import tao_model_soup  # noqa: E402


class FakeTensor:
    def __init__(self, values: object, dtype: str) -> None:
        self.array = np.asarray(values, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def is_floating_point(self) -> bool:
        return np.issubdtype(self.array.dtype, np.floating)

    def is_complex(self) -> bool:
        return np.issubdtype(self.array.dtype, np.complexfloating)

    def detach(self) -> "FakeTensor":
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(self.array.copy(), str(self.array.dtype))

    def to(self, *, dtype: np.dtype) -> "FakeTensor":
        return FakeTensor(self.array.astype(dtype), str(np.dtype(dtype)))

    def mul(self, value: float) -> "FakeTensor":
        return FakeTensor(self.array * value, str(self.array.dtype))

    def add_(self, other: "FakeTensor", *, alpha: float) -> "FakeTensor":
        self.array += other.array * alpha
        return self


class FakeTorch:
    float64 = np.dtype("float64")
    complex128 = np.dtype("complex128")

    def __init__(self, payloads: dict[Path, object]) -> None:
        self.payloads = payloads
        self.saved: object | None = None

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, FakeTensor)

    @staticmethod
    def equal(left: FakeTensor, right: FakeTensor) -> bool:
        return bool(np.array_equal(left.array, right.array))

    def load(self, path: Path, **_: object) -> object:
        return copy.deepcopy(self.payloads[Path(path)])

    def save(self, value: object, path: Path) -> None:
        self.saved = copy.deepcopy(value)
        Path(path).write_bytes(pickle.dumps("fake-checkpoint"))


class MetricTest(unittest.TestCase):
    def test_reads_last_jsonl_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            path.write_text(
                json.dumps({"kpi": {"mAP50": 0.4}})
                + "\n"
                + json.dumps({"kpi": {"mAP50": 0.6}})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(tao_model_soup.read_metric_file(path, "mAP50"), 0.6)
            self.assertEqual(tao_model_soup.read_metric_file(path, "kpi.mAP50"), 0.6)

    def test_rejects_nonfinite_metric(self) -> None:
        self.assertEqual(tao_model_soup.metric_values({"mAP50": "nan"}, "mAP50"), [])


class SpecTest(unittest.TestCase):
    def test_evaluation_writes_nested_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            template = root / "evaluate.yaml"
            template.write_text(
                yaml.safe_dump(
                    {
                        "results_dir": "PLACEHOLDER",
                        "evaluate": {"checkpoint": "PLACEHOLDER"},
                        "dataset": {"test_data_sources": {"json_file": "kpi.json"}},
                    }
                ),
                encoding="utf-8",
            )
            evaluator = root / "evaluate.py"
            evaluator.write_text(
                "import json, pathlib, sys\n"
                "spec = pathlib.Path(sys.argv[1])\n"
                "out = pathlib.Path(sys.argv[2])\n"
                "doc = __import__('yaml').safe_load(spec.read_text())\n"
                "assert doc['evaluate']['checkpoint'].endswith('model.pth')\n"
                "assert doc['results_dir'] == str(out)\n"
                "(out / 'status.json').write_text(json.dumps({'mAP50': 0.75}))\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                output_dir=root / "output",
                eval_spec=template,
                checkpoint_spec_key="evaluate.checkpoint",
                results_spec_key="results_dir",
                eval_command_json=json.dumps(
                    [sys.executable, str(evaluator), "{spec}", "{results_dir}"]
                ),
                metric_key="mAP50",
                metric_file_relative="status.json",
            )
            score = tao_model_soup.run_evaluation(checkpoint, "individual-000", args)
            self.assertEqual(score, 0.75)


class GreedyTest(unittest.TestCase):
    def test_greedy_uses_equal_original_ingredients_and_strict_gate(self) -> None:
        checkpoints = [Path("a.pth"), Path("b.pth"), Path("c.pth")]
        individual = {"a.pth": 0.7, "b.pth": 0.9, "c.pth": 0.8}
        proposals: list[list[str]] = []

        def propose(paths: list[Path], index: int) -> Path:
            proposals.append([str(path) for path in paths])
            return Path(f"candidate-{index}.pth")

        def evaluate(path: Path, label: str) -> float:
            if label.startswith("individual"):
                return individual[str(path)]
            return {"candidate-001": 0.91, "candidate-002": 0.90}[label]

        accepted, scores, trials, final = tao_model_soup.greedy_select(
            checkpoints,
            evaluate,
            propose,
            direction="maximize",
            min_improvement=0.0,
        )
        self.assertEqual(accepted, [Path("b.pth"), Path("c.pth")])
        self.assertEqual(proposals, [["b.pth", "c.pth"], ["b.pth", "c.pth", "a.pth"]])
        self.assertEqual([row["checkpoint"] for row in scores], ["b.pth", "c.pth", "a.pth"])
        self.assertEqual([row["accepted"] for row in trials], [True, False])
        self.assertEqual(final, 0.91)


class MergeTest(unittest.TestCase):
    def test_averages_float_and_copies_reference_integer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pth"
            second = root / "second.pth"
            output = root / "soup.pth"
            fake = FakeTorch(
                {
                    first: {
                        "state_dict": {
                            "weight": FakeTensor([1.0, 3.0], "float32"),
                            "counter": FakeTensor([1], "int64"),
                        },
                        "trainer": "reference",
                    },
                    second: {
                        "state_dict": {
                            "weight": FakeTensor([3.0, 5.0], "float32"),
                            "counter": FakeTensor([2], "int64"),
                        },
                        "trainer": "other",
                    },
                }
            )
            report = tao_model_soup.merge_checkpoints(
                [first, second], output, torch_module=fake
            )
            assert isinstance(fake.saved, dict)
            state = fake.saved["state_dict"]
            np.testing.assert_allclose(state["weight"].array, [2.0, 4.0])
            np.testing.assert_array_equal(state["counter"].array, [1])
            self.assertEqual(fake.saved["trainer"], "reference")
            self.assertEqual(report["floating_tensors_averaged"], 1)
            self.assertEqual(report["nonfloating_tensors_copied"], 1)
            self.assertEqual(report["nonfloating_tensors_differing"], 1)


if __name__ == "__main__":
    unittest.main()
