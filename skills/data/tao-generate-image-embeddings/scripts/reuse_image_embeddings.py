#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize current image embeddings from a compatible prior cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ENCODER_FIELDS = ("model", "model_path", "model_config_path")


def _new_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise ValueError(f"{label} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _read_spec(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"embedding spec must be a mapping: {path}")
    return document


def _encoder(spec: dict[str, Any]) -> dict[str, str]:
    return {field: str(spec.get(field, "")) for field in ENCODER_FIELDS}


def _read_unique(path: Path, label: str, require_embedding: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"filepath"} | ({"embedding"} if require_embedding else set())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame["filepath"].isna().any() or frame["filepath"].duplicated().any():
        raise ValueError(f"{label}.filepath must be non-null and unique")
    return frame.reset_index(drop=True)


def _embedding_dimensions(frame: pd.DataFrame, label: str) -> set[int]:
    dimensions = {len(value) for value in frame["embedding"] if value is not None}
    if not dimensions or frame["embedding"].isna().any():
        raise ValueError(f"{label}.embedding must be non-null and nonempty")
    if len(dimensions) != 1:
        raise ValueError(f"{label} contains mixed embedding dimensions: {dimensions}")
    return dimensions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize(current: pd.DataFrame, cached: pd.DataFrame,
                 fresh: pd.DataFrame | None) -> tuple[pd.DataFrame, int, int]:
    cache_map = cached.set_index("filepath")["embedding"]
    fresh_map = (fresh.set_index("filepath")["embedding"]
                 if fresh is not None else pd.Series(dtype=object))
    cached_paths = set(cache_map.index)
    current_paths = current["filepath"].tolist()
    missing = [path for path in current_paths if path not in cached_paths]
    if missing and fresh is None:
        raise ValueError(
            f"cache misses {len(missing)} of {len(current_paths)} current paths; "
            "provide --fresh-embeddings"
        )
    absent_fresh = [path for path in missing if path not in fresh_map.index]
    if absent_fresh:
        raise ValueError(f"fresh embeddings miss {len(absent_fresh)} required paths")
    vectors = [cache_map[path] if path in cached_paths else fresh_map[path]
               for path in current_paths]
    output = current.copy()
    output.insert(1, "embedding", vectors)
    return output, len(current_paths) - len(missing), len(missing)


def _verify_reference(output: pd.DataFrame, reference: pd.DataFrame,
                      absolute_tolerance: float = 0.0) -> float:
    expected = reference.set_index("filepath")["embedding"]
    if set(output["filepath"]) != set(expected.index):
        raise ValueError("reference embeddings do not cover the current filepath set")
    maximum = 0.0
    for path, vector in zip(output["filepath"], output["embedding"], strict=True):
        reference_vector = expected[path]
        if len(vector) != len(reference_vector):
            raise ValueError(f"reused embedding dimension differs for {path}")
        difference = max(
            (abs(float(left) - float(right))
             for left, right in zip(vector, reference_vector, strict=True)),
            default=0.0,
        )
        maximum = max(maximum, difference)
        if difference > absolute_tolerance:
            raise ValueError(
                f"reused embedding differs from reference for {path}: "
                f"max_abs_diff={difference}, tolerance={absolute_tolerance}"
            )
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-input", required=True)
    parser.add_argument("--current-spec", required=True)
    parser.add_argument("--cached-embeddings", required=True)
    parser.add_argument("--cached-spec", required=True)
    parser.add_argument("--fresh-embeddings")
    parser.add_argument("--reference-embeddings")
    parser.add_argument("--reference-atol", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.reference_atol < 0:
        raise ValueError("--reference-atol must be nonnegative")

    current_path = _existing_file(args.current_input, "current input")
    current_spec_path = _existing_file(args.current_spec, "current spec")
    cached_path = _existing_file(args.cached_embeddings, "cached embeddings")
    cached_spec_path = _existing_file(args.cached_spec, "cached spec")
    fresh_path = (_existing_file(args.fresh_embeddings, "fresh embeddings")
                  if args.fresh_embeddings else None)
    reference_path = (_existing_file(args.reference_embeddings, "reference embeddings")
                      if args.reference_embeddings else None)
    output_path = _new_file(args.output, "output")
    report_path = _new_file(args.report, "report")

    current_encoder = _encoder(_read_spec(current_spec_path))
    cached_encoder = _encoder(_read_spec(cached_spec_path))
    if current_encoder != cached_encoder:
        raise ValueError(
            f"encoder mismatch: current={current_encoder}, cached={cached_encoder}"
        )

    current = _read_unique(current_path, "current input", False)
    cached = _read_unique(cached_path, "cached embeddings", True)
    fresh = _read_unique(fresh_path, "fresh embeddings", True) if fresh_path else None
    reference = (_read_unique(reference_path, "reference embeddings", True)
                 if reference_path else None)
    dimensions = _embedding_dimensions(cached, "cached embeddings")
    if fresh is not None and _embedding_dimensions(fresh, "fresh embeddings") != dimensions:
        raise ValueError("fresh and cached embedding dimensions differ")

    output, reused_rows, fresh_rows = _materialize(current, cached, fresh)
    reference_max_abs_diff = None
    if reference is not None:
        if _embedding_dimensions(reference, "reference embeddings") != dimensions:
            raise ValueError("reference and cached embedding dimensions differ")
        reference_max_abs_diff = _verify_reference(
            output, reference, args.reference_atol
        )
    output.to_parquet(output_path, index=False)
    report = {
        "status": "COMPLETE",
        "encoder": current_encoder,
        "embedding_dim": next(iter(dimensions)),
        "rows": len(output),
        "reused_rows": reused_rows,
        "fresh_rows": fresh_rows,
        "reuse_pct": 100.0 * reused_rows / len(output) if len(output) else 0.0,
        "reference_verified": reference is not None,
        "reference_atol": args.reference_atol,
        "reference_max_abs_diff": reference_max_abs_diff,
        "inputs": {
            "current_input": {"path": str(current_path), "sha256": _sha256(current_path)},
            "current_spec": {"path": str(current_spec_path), "sha256": _sha256(current_spec_path)},
            "cached_embeddings": {"path": str(cached_path), "sha256": _sha256(cached_path)},
            "cached_spec": {"path": str(cached_spec_path), "sha256": _sha256(cached_spec_path)},
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    if fresh_path:
        report["inputs"]["fresh_embeddings"] = {
            "path": str(fresh_path), "sha256": _sha256(fresh_path)
        }
    if reference_path:
        report["inputs"]["reference_embeddings"] = {
            "path": str(reference_path), "sha256": _sha256(reference_path)
        }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
