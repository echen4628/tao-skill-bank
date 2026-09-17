# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "reuse_image_embeddings.py"
SPEC = importlib.util.spec_from_file_location("reuse_image_embeddings", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _frame(paths: list[str], embedded: bool = False) -> pd.DataFrame:
    data = {"filepath": paths, "tag": [f"current-{p}" for p in paths]}
    if embedded:
        data["embedding"] = [[float(index), 1.0] for index, _ in enumerate(paths)]
    return pd.DataFrame(data)


def test_materialize_full_reuse_preserves_current_metadata() -> None:
    current = _frame(["b.png", "a.png"])
    cached = pd.DataFrame({
        "filepath": ["a.png", "b.png"],
        "embedding": [[1.0, 1.0], [2.0, 1.0]],
        "tag": ["stale-a", "stale-b"],
    })
    output, reused, fresh = MODULE._materialize(current, cached, None)
    assert output.to_dict("records") == [
        {"filepath": "b.png", "embedding": [2.0, 1.0], "tag": "current-b.png"},
        {"filepath": "a.png", "embedding": [1.0, 1.0], "tag": "current-a.png"},
    ]
    assert (reused, fresh) == (2, 0)


def test_materialize_partial_reuse_requires_every_fresh_miss() -> None:
    current = _frame(["a.png", "b.png", "c.png"])
    cached = _frame(["a.png"], embedded=True)
    with pytest.raises(ValueError, match="cache misses 2"):
        MODULE._materialize(current, cached, None)
    fresh = _frame(["b.png"], embedded=True)
    with pytest.raises(ValueError, match="fresh embeddings miss 1"):
        MODULE._materialize(current, cached, fresh)


def test_reference_verification_rejects_changed_vector() -> None:
    output = pd.DataFrame({"filepath": ["a.png"], "embedding": [[1.0, 2.0]]})
    reference = pd.DataFrame({"filepath": ["a.png"], "embedding": [[1.0, 3.0]]})
    with pytest.raises(ValueError, match="differs from reference"):
        MODULE._verify_reference(output, reference)


def test_reference_verification_accepts_bounded_gpu_drift() -> None:
    output = pd.DataFrame({"filepath": ["a.png"], "embedding": [[1.0, 2.0]]})
    reference = pd.DataFrame({
        "filepath": ["a.png"], "embedding": [[1.0 + 1e-6, 2.0]],
    })
    maximum = MODULE._verify_reference(output, reference, absolute_tolerance=2e-6)
    assert maximum == pytest.approx(1e-6)
