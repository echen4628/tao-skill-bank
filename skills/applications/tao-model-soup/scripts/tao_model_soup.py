#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build uniform or greedy model soups from compatible TAO checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")


def load_torch() -> Any:
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required; run this script inside the resolved TAO PyTorch image"
        ) from exc
    return torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_tensor_mapping(value: Any, torch: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        torch.is_tensor(item) for item in value.values()
    )


def resolve_state_dict(
    payload: Any, torch: Any, explicit_key: str | None = None
) -> tuple[str, Mapping[str, Any]]:
    if explicit_key:
        if not isinstance(payload, Mapping) or explicit_key not in payload:
            raise ValueError(f"checkpoint has no state-dict key {explicit_key!r}")
        state = payload[explicit_key]
        if not is_tensor_mapping(state, torch):
            raise ValueError(f"checkpoint value {explicit_key!r} is not a tensor mapping")
        return explicit_key, state
    if is_tensor_mapping(payload, torch):
        return "<root>", payload
    if isinstance(payload, Mapping):
        matches = [key for key in STATE_DICT_KEYS if is_tensor_mapping(payload.get(key), torch)]
        if len(matches) == 1:
            key = matches[0]
            return key, payload[key]
        if len(matches) > 1:
            raise ValueError(
                f"checkpoint has multiple possible state dictionaries {matches}; "
                "select one with --state-dict-key"
            )
    raise ValueError(
        "could not find a tensor state dictionary at the checkpoint root or under "
        + ", ".join(STATE_DICT_KEYS)
    )


def state_descriptor(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "floating": bool(tensor.is_floating_point() or tensor.is_complex()),
        }
        for key, tensor in state.items()
    ]


def inspect_checkpoints(
    paths: Sequence[Path], state_dict_key: str | None, torch: Any
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    reference_locator: str | None = None
    reference_descriptor: list[dict[str, Any]] | None = None
    seen_hashes: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise ValueError(f"duplicate checkpoint content: {seen_hashes[digest]} and {path}")
        seen_hashes[digest] = path
        payload = torch_load(torch, path)
        locator, state = resolve_state_dict(payload, torch, state_dict_key)
        descriptor = state_descriptor(state)
        if reference_locator is None:
            reference_locator = locator
            reference_descriptor = descriptor
        elif locator != reference_locator:
            raise ValueError(
                f"state-dict location mismatch: {path} uses {locator!r}, "
                f"expected {reference_locator!r}"
            )
        elif descriptor != reference_descriptor:
            expected = {row["key"]: row for row in reference_descriptor or []}
            actual = {row["key"]: row for row in descriptor}
            missing = sorted(set(expected) - set(actual))[:5]
            extra = sorted(set(actual) - set(expected))[:5]
            changed = sorted(
                key
                for key in set(expected) & set(actual)
                if expected[key] != actual[key]
            )[:5]
            raise ValueError(
                f"incompatible state dictionary in {path}; missing={missing}, "
                f"extra={extra}, changed={changed}"
            )
        records.append(
            {
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        del payload, state
    assert reference_locator is not None and reference_descriptor is not None
    return records, reference_locator, reference_descriptor


def merge_checkpoints(
    paths: Sequence[Path],
    output: Path,
    *,
    state_dict_key: str | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one checkpoint is required")
    torch = torch_module or load_torch()
    weight = 1.0 / len(paths)
    payload = torch_load(torch, paths[0])
    locator, reference = resolve_state_dict(payload, torch, state_dict_key)
    merged: dict[str, Any] = {}
    nonfloating_differences: set[str] = set()

    for key, tensor in reference.items():
        if tensor.is_floating_point():
            merged[key] = tensor.detach().to(dtype=torch.float64).mul(weight)
        elif tensor.is_complex():
            merged[key] = tensor.detach().to(dtype=torch.complex128).mul(weight)
        else:
            merged[key] = tensor.detach().clone()

    for path in paths[1:]:
        other_payload = torch_load(torch, path)
        other_locator, other = resolve_state_dict(other_payload, torch, state_dict_key)
        if other_locator != locator or list(other) != list(reference):
            raise ValueError(f"state dictionary changed after preflight: {path}")
        for key, tensor in other.items():
            ref = reference[key]
            if tuple(tensor.shape) != tuple(ref.shape) or tensor.dtype != ref.dtype:
                raise ValueError(f"tensor changed after preflight: {path}:{key}")
            if ref.is_floating_point() or ref.is_complex():
                merged[key].add_(tensor.detach().to(dtype=merged[key].dtype), alpha=weight)
            elif not torch.equal(ref, tensor):
                nonfloating_differences.add(key)
        del other_payload, other

    for key, ref in reference.items():
        if ref.is_floating_point() or ref.is_complex():
            merged[key] = merged[key].to(dtype=ref.dtype)

    if locator == "<root>":
        output_payload = merged
    else:
        payload[locator] = type(reference)(merged)
        output_payload = payload

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(output_payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "floating_tensors_averaged": sum(
            1 for tensor in reference.values() if tensor.is_floating_point() or tensor.is_complex()
        ),
        "nonfloating_tensors_copied": sum(
            1
            for tensor in reference.values()
            if not tensor.is_floating_point() and not tensor.is_complex()
        ),
        "nonfloating_tensors_differing": len(nonfloating_differences),
        "nonfloating_difference_keys": sorted(nonfloating_differences),
    }


def set_nested(document: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = document
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted_key!r}; {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def metric_values(value: Any, metric_key: str) -> list[float]:
    parts = metric_key.split(".")
    found: list[float] = []

    def finite_number(item: Any) -> float | None:
        if isinstance(item, bool):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def dotted(document: Any) -> Any:
        current = document
        for part in parts:
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    direct = finite_number(dotted(value))
    if direct is not None:
        found.append(direct)
        return found

    if len(parts) == 1:
        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    if key == metric_key:
                        number = finite_number(child)
                        if number is not None:
                            found.append(number)
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    return found


def read_metric_file(path: Path, metric_key: str) -> float | None:
    text = path.read_text(encoding="utf-8")
    documents: list[Any] = []
    try:
        documents.append(json.loads(text))
    except json.JSONDecodeError:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    values = [number for document in documents for number in metric_values(document, metric_key)]
    return values[-1] if values else None


def find_metric(
    results_dir: Path, metric_key: str, metric_file_relative: str | None
) -> tuple[float, Path]:
    if metric_file_relative:
        candidates = [results_dir / metric_file_relative]
    else:
        status = sorted(results_dir.rglob("status.json"))
        candidates = status + [
            path for path in sorted(results_dir.rglob("*.json")) if path not in status
        ]
    for path in candidates:
        if path.is_file():
            value = read_metric_file(path, metric_key)
            if value is not None:
                return value, path
    raise ValueError(
        f"metric {metric_key!r} was not found as a finite JSON value under {results_dir}"
    )


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or "evaluation"


def run_evaluation(checkpoint: Path, label: str, args: argparse.Namespace) -> float:
    run_dir = args.output_dir / "evaluations" / safe_label(label)
    if run_dir.exists():
        raise ValueError(f"evaluation directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    spec_path = run_dir / "evaluate.yaml"
    if args.eval_spec:
        document = yaml.safe_load(args.eval_spec.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("evaluation spec must contain a YAML mapping")
        document = copy.deepcopy(document)
        set_nested(document, args.checkpoint_spec_key, str(checkpoint))
        set_nested(document, args.results_spec_key, str(run_dir))
        spec_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    if args.eval_command_json:
        raw_command = json.loads(args.eval_command_json)
        if not isinstance(raw_command, list) or not raw_command or not all(
            isinstance(item, str) and item for item in raw_command
        ):
            raise ValueError("--eval-command-json must be a non-empty JSON array of strings")
    else:
        if not args.eval_spec:
            raise ValueError("greedy mode requires --eval-spec or --eval-command-json")
        raw_command = ["rtdetr", "evaluate", "-e", "{spec}"]
    replacements = {
        "checkpoint": str(checkpoint),
        "results_dir": str(run_dir),
        "spec": str(spec_path),
        "label": label,
    }
    command = [item.format(**replacements) for item in raw_command]
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "evaluation.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluation {label!r} failed with exit code {completed.returncode}; "
            f"see {run_dir / 'evaluation.log'}"
        )
    score, metric_path = find_metric(
        run_dir, args.metric_key, args.metric_file_relative
    )
    record = {
        "checkpoint": str(checkpoint),
        "metric_key": args.metric_key,
        "score": score,
        "metric_file": str(metric_path),
    }
    (run_dir / "score.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return score


def is_improvement(candidate: float, incumbent: float, direction: str, minimum: float) -> bool:
    if direction == "maximize":
        return candidate > incumbent + minimum
    return candidate < incumbent - minimum


def greedy_select(
    checkpoints: Sequence[Path],
    evaluate: Callable[[Path, str], float],
    propose: Callable[[Sequence[Path], int], Path],
    *,
    direction: str,
    min_improvement: float,
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]], float]:
    individual = [
        {"checkpoint": path, "score": evaluate(path, f"individual-{index:03d}")}
        for index, path in enumerate(checkpoints)
    ]
    reverse = direction == "maximize"
    ordered = sorted(individual, key=lambda row: row["score"], reverse=reverse)
    accepted = [ordered[0]["checkpoint"]]
    incumbent = float(ordered[0]["score"])
    trials: list[dict[str, Any]] = []
    for index, row in enumerate(ordered[1:], start=1):
        ingredients = [*accepted, row["checkpoint"]]
        candidate_path = propose(ingredients, index)
        score = evaluate(candidate_path, f"candidate-{index:03d}")
        accepted_candidate = is_improvement(
            score, incumbent, direction, min_improvement
        )
        trials.append(
            {
                "checkpoint": str(row["checkpoint"]),
                "score": score,
                "accepted": accepted_candidate,
                "ingredient_count": len(ingredients),
                "incumbent_before": incumbent,
            }
        )
        if accepted_candidate:
            accepted.append(row["checkpoint"])
            incumbent = score
    serializable_individual = [
        {"checkpoint": str(row["checkpoint"]), "score": row["score"]}
        for row in ordered
    ]
    return accepted, serializable_individual, trials, incumbent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("uniform", "greedy"), required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, help="Repeat for each trusted checkpoint."
    )
    parser.add_argument(
        "--published-checkpoint",
        action="append",
        help=(
            "Optional durable identity for each staged --checkpoint, in the same order. "
            "Used only in the manifest."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--published-output-dir",
        help="Optional durable output directory recorded in the manifest.",
    )
    parser.add_argument("--output-name", default="model_soup.pth")
    parser.add_argument("--state-dict-key")
    parser.add_argument("--eval-spec", help="Nested TAO RT-DETR KPI evaluate YAML.")
    parser.add_argument("--metric-key", default="mAP50")
    parser.add_argument("--metric-file-relative")
    parser.add_argument(
        "--eval-command-json",
        help=(
            "Optional evaluator argv as JSON; supports "
            "checkpoint/results_dir/spec/label placeholders."
        ),
    )
    parser.add_argument("--checkpoint-spec-key", default="evaluate.checkpoint")
    parser.add_argument("--results-spec-key", default="results_dir")
    parser.add_argument("--direction", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint = [Path(value).expanduser().resolve() for value in args.checkpoint]
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.published_checkpoint:
        if len(args.published_checkpoint) != len(args.checkpoint):
            parser.error("--published-checkpoint must be repeated once per --checkpoint")
        args.published_checkpoint = [
            Path(value).expanduser() for value in args.published_checkpoint
        ]
        if not all(path.is_absolute() for path in args.published_checkpoint):
            parser.error("--published-checkpoint values must be absolute paths")
    else:
        args.published_checkpoint = list(args.checkpoint)
    args.published_output_dir = (
        Path(args.published_output_dir).expanduser()
        if args.published_output_dir
        else args.output_dir
    )
    if not args.published_output_dir.is_absolute():
        parser.error("--published-output-dir must be an absolute path")
    args.eval_spec = Path(args.eval_spec).expanduser().resolve() if args.eval_spec else None
    if len(args.checkpoint) < 2:
        parser.error("at least two --checkpoint values are required")
    if args.min_improvement < 0 or not math.isfinite(args.min_improvement):
        parser.error("--min-improvement must be a finite non-negative number")
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a file name, not a path")
    if args.method == "greedy" and not (args.eval_spec or args.eval_command_json):
        parser.error("greedy mode requires --eval-spec or --eval-command-json")
    if args.eval_spec and not args.eval_spec.is_file():
        parser.error(f"evaluation spec does not exist: {args.eval_spec}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        final_path = args.output_dir / args.output_name
        manifest_path = args.output_dir / "soup_manifest.json"
        if not args.overwrite and (final_path.exists() or manifest_path.exists()):
            raise ValueError(
                "final output already exists; choose a new --output-dir or explicitly "
                "pass --overwrite"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for managed in (
                final_path,
                manifest_path,
                args.output_dir / "evaluations",
                args.output_dir / ".candidates",
            ):
                if managed.is_symlink() or managed.is_file():
                    managed.unlink()
                elif managed.is_dir():
                    shutil.rmtree(managed)
        torch = load_torch()
        records, locator, descriptor = inspect_checkpoints(
            args.checkpoint, args.state_dict_key, torch
        )
        work_dir = args.output_dir / ".candidates"
        if work_dir.exists():
            raise ValueError(f"candidate workspace already exists: {work_dir}")
        work_dir.mkdir()
        merge_report: dict[str, Any]
        individual_scores: list[dict[str, Any]] = []
        candidate_trials: list[dict[str, Any]] = []
        final_score: float | None = None

        if args.method == "uniform":
            ingredients = list(args.checkpoint)
        else:
            def evaluate(path: Path, label: str) -> float:
                return run_evaluation(path, label, args)

            def propose(paths: Sequence[Path], index: int) -> Path:
                candidate = work_dir / f"candidate-{index:03d}.pth"
                merge_checkpoints(
                    paths,
                    candidate,
                    state_dict_key=args.state_dict_key,
                    torch_module=torch,
                )
                return candidate

            ingredients, individual_scores, candidate_trials, final_score = greedy_select(
                args.checkpoint,
                evaluate,
                propose,
                direction=args.direction,
                min_improvement=args.min_improvement,
            )

        merge_report = merge_checkpoints(
            ingredients,
            final_path,
            state_dict_key=args.state_dict_key,
            torch_module=torch,
        )
        final_hash = sha256_file(final_path)
        by_path = {record["path"]: record for record in records}
        published_by_path = {
            str(runtime): str(published)
            for runtime, published in zip(args.checkpoint, args.published_checkpoint)
        }
        published_inputs = []
        for record in records:
            published = dict(record)
            published["path"] = published_by_path[record["path"]]
            published_inputs.append(published)
        published_individual_scores = [
            {
                **row,
                "checkpoint": published_by_path.get(row["checkpoint"], row["checkpoint"]),
            }
            for row in individual_scores
        ]
        published_candidate_trials = [
            {
                **row,
                "checkpoint": published_by_path.get(row["checkpoint"], row["checkpoint"]),
            }
            for row in candidate_trials
        ]
        manifest = {
            "schema_version": 1,
            "method": args.method,
            "architecture": "rtdetr",
            "metric_key": args.metric_key if args.method == "greedy" else None,
            "direction": args.direction if args.method == "greedy" else None,
            "min_improvement": args.min_improvement if args.method == "greedy" else None,
            "checkpoint_contract": {
                "state_dict_location": locator,
                "tensor_count": len(descriptor),
                "floating_tensor_count": sum(row["floating"] for row in descriptor),
            },
            "inputs": published_inputs,
            "individual_scores": published_individual_scores,
            "candidate_trials": published_candidate_trials,
            "ingredients": [
                {
                    **by_path[str(path)],
                    "path": published_by_path[str(path)],
                    "weight": 1.0 / len(ingredients),
                }
                for path in ingredients
            ],
            "final_score": final_score,
            "merge": merge_report,
            "output": {
                "path": str(args.published_output_dir / args.output_name),
                "sha256": final_hash,
                "size_bytes": final_path.stat().st_size,
            },
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
        if not args.keep_candidates:
            shutil.rmtree(work_dir)
        print(
            f"COMPLETE method={args.method} ingredients={len(ingredients)} "
            f"output={final_path} sha256={final_hash}"
        )
        if final_score is not None:
            print(f"selection_metric {args.metric_key}={final_score}")
        print(f"manifest={manifest_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
