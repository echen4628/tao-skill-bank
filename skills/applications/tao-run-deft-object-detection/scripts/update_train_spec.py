#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append this iteration's detector-native producers to a DEFT OD train spec.

Grounding DINO entries use ODVG plus ``label_map``. RT-DETR entries use COCO
without ``label_map`` and freeze ``num_classes`` / ``eval_class_ids`` from the
COCO categories. Dataset growth is list-append for both detectors.

This replaces the reference pipeline's ``reformat_train_spec.py``, which exists
only inside an internal container image and cannot be called from a skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def append_source(
    spec: dict[str, Any],
    image_dir: str,
    annotations_file: str,
    label_map: str | None = None,
) -> int:
    dataset = spec.setdefault("dataset", {})
    sources = dataset.get("train_data_sources")
    if sources is None:
        sources = []
    if not isinstance(sources, list):
        raise ValueError(
            "dataset.train_data_sources must be a list for DEFT dataset accumulation; "
            f"found {type(sources).__name__}."
        )

    entry = {"image_dir": image_dir, "json_file": annotations_file}
    if label_map is not None:
        entry["label_map"] = label_map
    # Re-running an iteration must not double-add the same source.
    if entry in sources:
        print(f"NOTE: source already present, not appending again: {image_dir}")
    else:
        sources.append(entry)
    dataset["train_data_sources"] = sources
    return len(sources)


def set_pretrained(spec: dict[str, Any], checkpoint: str) -> str:
    """Overwrite train.pretrained_model_path with the run's resolved checkpoint.

    Always overwritten, never inherited. The run resolved one checkpoint — the
    user's if they supplied it, otherwise the one pulled from NGC — and that is the
    only correct value. A template carrying some other path would otherwise train
    from weights nobody chose, and every iteration after it would be measured
    against a baseline it does not share.

    Every iteration fine-tunes this same BASE checkpoint, never the previous
    iteration's, so the value is identical on every pass.

    Left unset entirely, training starts from nothing, reports success, and emits a
    model that detects nothing — the most expensive silent failure in the loop,
    because it costs a full training run and only surfaces at KPI.
    """
    train = spec.setdefault("train", {})
    previous = train.get("pretrained_model_path")
    train["pretrained_model_path"] = checkpoint
    if previous and str(previous) != checkpoint:
        print(f"NOTE: replacing train.pretrained_model_path\n"
              f"        template: {previous}\n"
              f"        this run: {checkpoint}")
    return checkpoint


def set_max_labels(spec: dict[str, Any], label_map_file: Path) -> int:
    """Derive dataset.max_labels from the staged label map.

    It caps how many class phrases go into a caption, so it has to match the class
    count. Hardcoding it means every run with a different number of target classes
    is silently wrong: too low and classes are dropped from captions, too high and
    the sampler pads with labels that do not exist.
    """
    label_map = json.loads(label_map_file.read_text(encoding="utf-8"))
    count = len(label_map)
    if count < 1:
        raise ValueError(f"{label_map_file} declares no classes")
    dataset = spec.setdefault("dataset", {})
    previous = dataset.get("max_labels")
    dataset["max_labels"] = count
    if isinstance(previous, int) and previous != count:
        print(f"NOTE: dataset.max_labels {previous} -> {count}, from the staged label map.")
    return count


def set_validation(spec: dict[str, Any], image_dir: str, json_file: str) -> None:
    """Point val (and test) at the pool-derived split.

    Both are set: `grounding_dino train` subscripts val_data_sources unconditionally,
    and leaving test unset is a second way to trip the same schema. The members must
    be real paths — the field is Optional as a whole, but its members are not, so
    `{image_dir: null}` is rejected by Hydra while a bare `null` crashes the loader.
    """
    dataset = spec.setdefault("dataset", {})
    sources = {"image_dir": image_dir, "json_file": json_file}
    dataset["val_data_sources"] = dict(sources)
    dataset["test_data_sources"] = dict(sources)


def set_rtdetr_class_contract(
    spec: dict[str, Any], coco_file: Path
) -> tuple[int, list[int], list[str]]:
    """Pin RT-DETR's class-head contract from one staged COCO source."""
    coco = json.loads(coco_file.read_text(encoding="utf-8"))
    categories = coco.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{coco_file} has no COCO categories")
    parsed: list[tuple[int, str]] = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"{coco_file} categories must be objects")
        category_id = category.get("id")
        name = str(category.get("name", "")).strip()
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise ValueError(f"{coco_file} category ids must be integers")
        if not name:
            raise ValueError(f"{coco_file} category names must be non-empty")
        parsed.append((category_id, name))
    parsed.sort()
    ids = [category_id for category_id, _name in parsed]
    names = [name for _category_id, name in parsed]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError(f"{coco_file} category ids and names must be unique")
    if ids not in (list(range(len(ids))), list(range(1, len(ids) + 1))):
        raise ValueError(
            f"RT-DETR DEFT requires dense category ids 0..N-1 or 1..N; got {ids}"
        )
    dataset = spec.setdefault("dataset", {})
    num_classes = max(ids) + 1
    existing_num_classes = dataset.get("num_classes")
    existing_eval_ids = dataset.get("eval_class_ids")
    if existing_num_classes not in (None, num_classes):
        print(
            f"NOTE: dataset.num_classes {existing_num_classes} -> {num_classes}, from staged COCO."
        )
    if existing_eval_ids not in (None, ids):
        print(
            f"NOTE: dataset.eval_class_ids {existing_eval_ids} -> {ids}, from staged COCO."
        )
    dataset["num_classes"] = num_classes
    dataset["eval_class_ids"] = ids
    dataset["remap_mscoco_category"] = False
    return num_classes, ids, names


def validate_source_paths(values: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Resolve one complete ODVG source triplet and prove every path exists."""
    resolved: list[str] = []
    for flag, raw in values:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{flag} must be absolute: {path}")
        if not path.exists():
            raise FileNotFoundError(f"{flag} does not exist: {path}")
        resolved.append(str(path.resolve()))
    return tuple(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        choices=("grounding_dino", "rtdetr"),
        default="grounding_dino",
    )
    parser.add_argument("--previous-spec", required=True, help="Train spec from the previous phase.")
    parser.add_argument("--output-spec", required=True, help="Where to write the new spec.")
    parser.add_argument("--tmm-image-dir", required=True)
    parser.add_argument("--tmm-odvg-file", default=None)
    parser.add_argument("--tmm-label-map-file", default=None)
    parser.add_argument("--tmm-coco-file", default=None)
    parser.add_argument("--synthetic-image-dir", default=None,
                        help="Optional staged AnomalyGenNext image directory. Must be supplied "
                             "with --synthetic-odvg-file and --synthetic-label-map-file.")
    parser.add_argument("--synthetic-odvg-file", default=None)
    parser.add_argument("--synthetic-label-map-file", default=None)
    parser.add_argument("--synthetic-coco-file", default=None)
    parser.add_argument("--pretrained-model-path", required=True,
                        help="Frozen base checkpoint every iteration fine-tunes FROM — the "
                             "user's if supplied, or the Grounding DINO NGC download. "
                             "Always overwrites whatever "
                             "the template carried; inheriting a stale path would train from "
                             "weights nobody chose.")
    parser.add_argument("--val-image-dir", default=None,
                        help="Validation image directory — the pool images. Required together "
                             "with --val-json-file. `grounding_dino train` cannot run without "
                             "a validation source; see references/grounding-dino.md.")
    parser.add_argument("--val-json-file", default=None,
                        help="Validation COCO from make_pool_val_split.py. Use 0-based ids for "
                             "Grounding DINO and --category-id-policy preserve for RT-DETR.")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    try:
        previous = Path(args.previous_spec).expanduser().resolve()
        if not previous.is_file():
            raise FileNotFoundError(f"previous-spec does not exist: {previous}")
        spec = load_yaml(previous)

        if args.detector == "grounding_dino":
            if not args.tmm_odvg_file or not args.tmm_label_map_file:
                raise ValueError(
                    "Grounding DINO requires --tmm-odvg-file and --tmm-label-map-file"
                )
            tmm_image_dir, tmm_annotations, tmm_label_map_file = validate_source_paths(
                (
                    ("--tmm-image-dir", args.tmm_image_dir),
                    ("--tmm-odvg-file", args.tmm_odvg_file),
                    ("--tmm-label-map-file", args.tmm_label_map_file),
                )
            )
            count = append_source(
                spec, tmm_image_dir, tmm_annotations, tmm_label_map_file
            )
            synthetic_values = (
                args.synthetic_image_dir,
                args.synthetic_odvg_file,
                args.synthetic_label_map_file,
            )
            if any(synthetic_values) and not all(synthetic_values):
                raise ValueError(
                    "Grounding DINO requires --synthetic-image-dir, "
                    "--synthetic-odvg-file, and --synthetic-label-map-file together"
                )
            if all(synthetic_values):
                synthetic_image_dir, synthetic_annotations, synthetic_label_map = (
                    validate_source_paths(
                        (
                            ("--synthetic-image-dir", args.synthetic_image_dir),
                            ("--synthetic-odvg-file", args.synthetic_odvg_file),
                            ("--synthetic-label-map-file", args.synthetic_label_map_file),
                        )
                    )
                )
                count = append_source(
                    spec,
                    synthetic_image_dir,
                    synthetic_annotations,
                    synthetic_label_map,
                )
            classes = set_max_labels(spec, Path(tmm_label_map_file))
        else:
            if not args.tmm_coco_file:
                raise ValueError("RT-DETR requires --tmm-coco-file")
            tmm_image_dir, tmm_annotations = validate_source_paths(
                (
                    ("--tmm-image-dir", args.tmm_image_dir),
                    ("--tmm-coco-file", args.tmm_coco_file),
                )
            )
            count = append_source(spec, tmm_image_dir, tmm_annotations)
            synthetic_values = (args.synthetic_image_dir, args.synthetic_coco_file)
            if any(synthetic_values) and not all(synthetic_values):
                raise ValueError(
                    "RT-DETR requires --synthetic-image-dir and --synthetic-coco-file together"
                )
            if all(synthetic_values):
                synthetic_image_dir, synthetic_annotations = validate_source_paths(
                    (
                        ("--synthetic-image-dir", args.synthetic_image_dir),
                        ("--synthetic-coco-file", args.synthetic_coco_file),
                    )
                )
                (
                    synthetic_num_classes,
                    synthetic_eval_ids,
                    synthetic_class_names,
                ) = set_rtdetr_class_contract(spec, Path(synthetic_annotations))
                count = append_source(spec, synthetic_image_dir, synthetic_annotations)
            classes, eval_ids, class_names = set_rtdetr_class_contract(
                spec, Path(tmm_annotations)
            )
            if all(synthetic_values) and (
                synthetic_num_classes != classes
                or synthetic_eval_ids != eval_ids
                or synthetic_class_names != class_names
            ):
                raise ValueError(
                    "synthetic COCO category contract differs from the mined COCO contract"
                )

        set_pretrained(spec, args.pretrained_model_path)

        # Validation is mandatory for `grounding_dino train`, so refuse to emit a spec
        # that cannot train. The template ships val_data_sources unset precisely so this
        # has to be answered here rather than silently inherited.
        if bool(args.val_image_dir) != bool(args.val_json_file):
            raise ValueError("--val-image-dir and --val-json-file must be given together")
        if args.val_image_dir:
            for flag, raw in (("--val-image-dir", args.val_image_dir),
                              ("--val-json-file", args.val_json_file)):
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    raise ValueError(f"{flag} must be absolute: {path}")
                if not path.exists():
                    raise FileNotFoundError(f"{flag} does not exist: {path}")
            set_validation(
                spec,
                str(Path(args.val_image_dir).expanduser().resolve()),
                str(Path(args.val_json_file).expanduser().resolve()),
            )
        else:
            existing = (spec.get("dataset") or {}).get("val_data_sources")
            if not (isinstance(existing, dict) and existing.get("json_file")):
                policy = "0-based" if args.detector == "grounding_dino" else "ID-preserving"
                raise ValueError(
                    f"no validation source for {args.detector}. Pass "
                    "--val-image-dir/--val-json-file; make_pool_val_split.py builds the "
                    f"required {policy} COCO from the prepared pool."
                )

        if args.num_epochs is not None:
            if args.num_epochs < 1:
                raise ValueError("--num-epochs must be at least 1.")
            train = spec.setdefault("train", {})
            train["num_epochs"] = args.num_epochs
            # TAO fails at config-merge time when checkpoint_interval exceeds num_epochs.
            interval = train.get("checkpoint_interval")
            if isinstance(interval, int) and interval > args.num_epochs:
                print(
                    f"NOTE: lowering train.checkpoint_interval {interval} -> {args.num_epochs} "
                    "to satisfy checkpoint_interval <= num_epochs."
                )
                train["checkpoint_interval"] = args.num_epochs
            val_interval = train.get("validation_interval")
            if isinstance(val_interval, int) and val_interval > args.num_epochs:
                print(
                    f"NOTE: lowering train.validation_interval {val_interval} -> {args.num_epochs}."
                )
                train["validation_interval"] = args.num_epochs

        if args.learning_rate is not None:
            spec.setdefault("train", {}).setdefault("optim", {})["lr"] = args.learning_rate

        if args.num_gpus is not None:
            if args.num_gpus < 1:
                raise ValueError("--num-gpus must be at least 1")
            train = spec.setdefault("train", {})
            train["num_gpus"] = args.num_gpus
            train["gpu_ids"] = list(range(args.num_gpus))

        output = Path(args.output_spec).expanduser().resolve()
        write_yaml(output, spec)
        print(f"Wrote train spec: {output}")
        print(f"detector={args.detector} classes={classes}")
        print(f"dataset.train_data_sources now has {count} source(s)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
