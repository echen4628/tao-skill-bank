#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare frozen FN-driven inputs for AnomalyGenNext inference.

The preparation step resolves box-level false negatives, selects two same-type
mask templates, embeds and searches compatible clean images, runs AMP, and
freezes the exact generator testcase files. The separate generation skill
consumes only these hash-validated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
BRANCHES = ("fn_mask", "same_type_sampled_mask")
AOI_DATASET_MARKERS = {
    "/MVTec-AD/": "mvtec",
    "/VisA/": "visa",
    "/DAGM_2007/": "dagm",
    "/BTAD/": "btad",
    "/MPDD/": "mpdd",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any, length: int = 16) -> str:
    raw = "\x1f".join(json.dumps(_jsonable(part), sort_keys=True) for part in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def _images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("pipeline config must be a mapping")
    required = {
        "gap_parquet",
        "split_root",
        "pool_dataset_root",
        "defect_spec",
        "datasets",
        "selection",
        "embedding",
        "retrieval",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"pipeline config missing keys: {missing}")
    config.setdefault("source_tag", "user_provided")
    if not isinstance(config["source_tag"], str) or not config["source_tag"].strip():
        raise ValueError("pipeline config source_tag must be a non-empty provenance label")
    return config


def _recipe_types(recipe_path: Path) -> set[str]:
    recipe = yaml.safe_load(recipe_path.read_text())
    values = recipe.get("anomaly_types") if isinstance(recipe, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"recipe has no anomaly_types list: {recipe_path}")
    result = set()
    for row in values:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"invalid anomaly_types row in {recipe_path}: {row!r}")
        result.add(f"{row[0]}+{row[1]}")
    return result


def _defect_spec(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        anomaly_type = str(row.get("defect_type", ""))
        if not anomaly_type or anomaly_type in result:
            raise ValueError(f"invalid or duplicate defect_type in {path}: {anomaly_type!r}")
        if row.get("spatial_dependency") == "text" and not str(
            row.get("roi_prompt_defect_location", "")
        ).strip():
            raise ValueError(
                f"text-routed defect {anomaly_type!r} requires "
                "roi_prompt_defect_location"
            )
        result[anomaly_type] = row
    return result


def _dataset_for_path(filepath: str, datasets: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    parts = Path(filepath).parts
    for dataset_id, dataset in datasets.items():
        marker = str(dataset["path_marker"])
        if marker in parts:
            return dataset_id, dataset
    return None


def _type_and_mask(filepath: str, dataset_id: str, dataset: dict[str, Any]) -> tuple[str, str, Path]:
    image = Path(filepath)
    parts = list(image.parts)
    marker_index = parts.index(str(dataset["path_marker"]))
    texture_source = parts[marker_index + int(dataset.get("texture_offset", 1))]
    split_components = [str(value) for value in dataset.get("split_components", ["test"])]
    split_index = next(
        (index for index in range(marker_index, len(parts)) if parts[index] in split_components),
        None,
    )
    if split_index is None:
        raise ValueError(f"none of split_components {split_components} found in {filepath}")
    fixed_defect_class = dataset.get("defect_class_fixed")
    defect_class = (
        str(fixed_defect_class)
        if fixed_defect_class is not None
        else parts[split_index + 1]
    )
    texture_id = f"{dataset.get('texture_prefix', dataset_id + '_')}{texture_source}"
    anomaly_type = f"{texture_id}+{defect_class}"
    mask_style = dataset.get("mask_style", "stem_mask_png")
    mask_parts = parts[:]
    component_replacements = dataset.get("mask_component_replacements")
    if component_replacements:
        mask_parts = [str(component_replacements.get(part, part)) for part in mask_parts]
    else:
        mask_parts[split_index] = "ground_truth"
    if mask_style == "stem_png":
        default_suffix = ""
    elif mask_style == "stem_mask_png":
        default_suffix = "_mask"
    elif mask_style == "configurable":
        default_suffix = ""
    else:
        raise ValueError(f"unknown mask_style for {dataset_id}: {mask_style}")
    suffix = str(dataset.get("mask_suffix", default_suffix))
    extensions = [str(value) for value in dataset.get("mask_extensions", [".png"])]
    mask_base = Path(*mask_parts)
    candidates = [mask_base.with_name(image.stem + suffix + extension) for extension in extensions]
    mask = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return texture_id, defect_class, mask


def _mask_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        array = np.asarray(image.convert("L")) > 0
    if not array.any():
        raise ValueError(f"empty mask: {path}")
    return array


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _isolate_fn_mask(mask_path: Path, image_path: Path, bbox: Any, output: Path) -> dict[str, Any]:
    mask = _mask_array(mask_path)
    with Image.open(image_path) as image:
        width, height = image.size
    if mask.shape != (height, width):
        raise ValueError(
            f"FN mask/image size mismatch: mask={mask.shape[::-1]} image={(width, height)} "
            f"mask_path={mask_path}"
        )
    values = np.asarray(bbox, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError(f"invalid FN bbox: {bbox!r}")
    x1, y1, x2, y2 = values.tolist()
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2)) + 1))
    bottom = max(0, min(height, int(math.ceil(y2)) + 1))
    isolated = np.zeros_like(mask)
    isolated[top:bottom, left:right] = mask[top:bottom, left:right]
    if not isolated.any():
        raise ValueError(f"FN mask has no pixels inside bbox {values.tolist()}: {mask_path}")
    _write_mask(output, isolated)
    return {
        "source_mask": str(mask_path),
        "isolated_mask": str(output),
        "pixels": int(isolated.sum()),
        "content_sha256": _sha256(output),
    }


def _copy_sampled_mask(source: Path, output: Path) -> dict[str, Any]:
    mask = _mask_array(source)
    _write_mask(output, mask)
    return {
        "source_mask": str(source),
        "isolated_mask": str(output),
        "pixels": int(mask.sum()),
        "content_sha256": _sha256(output),
    }


def _split_metadata(split_root: Path, datasets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset_id in datasets:
        path = split_root / dataset_id / "split_manifest.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, dict):
            raise ValueError(f"split manifest is not a mapping: {path}")
        for split_id, row in rows.items():
            source = str(row["source_path"])
            if source in result:
                raise ValueError(f"duplicate source path across split manifests: {source}")
            result[source] = {**row, "dataset_id": dataset_id, "split_id": str(split_id)}
    return result


def prepare_inputs(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    source_config_path = Path(args.config).resolve()
    run = Path(args.run_root)
    for path in (
        run / "manifests",
        run / "specs",
        run / "embeddings",
        run / "amp",
        run / "prepared_anomalygennext_inputs" / "source_masks",
        run / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    config_path = run / "prepared_anomalygennext_inputs" / "filtering_config.yaml"
    if source_config_path != config_path.resolve():
        shutil.copy2(source_config_path, config_path)

    gaps_path = Path(config["gap_parquet"])
    gaps = pd.read_parquet(gaps_path)
    required = {"image_id", "filepath", "gap_type", "bbox", "class"}
    missing = sorted(required - set(gaps.columns))
    if missing:
        raise ValueError(f"gap parquet missing columns: {missing}")
    fn = gaps[gaps["gap_type"].astype(str).str.upper().eq("FN")].copy()
    fn = fn.sort_values(["image_id", "filepath"], kind="stable").reset_index(drop=True)

    datasets = config["datasets"]
    split_meta = _split_metadata(Path(config["split_root"]), datasets)
    placement = _defect_spec(Path(config["defect_spec"]))
    pool_root = Path(config["pool_dataset_root"])
    recipe_types: dict[str, set[str]] = {}
    for dataset_id, dataset in datasets.items():
        recipe = dataset.get("recipe")
        if recipe:
            recipe_types[dataset_id] = _recipe_types(Path(recipe))

    ledger_rows: list[dict[str, Any]] = []
    for box_order, row in fn.iterrows():
        filepath = str(row["filepath"])
        match = _dataset_for_path(filepath, datasets)
        dataset_id = match[0] if match else "unknown"
        dataset = match[1] if match else None
        reasons: list[str] = []
        texture_id = defect_class = anomaly_type = ""
        fn_mask = ""
        if dataset is None:
            reasons.append("unmapped_dataset")
        else:
            try:
                texture_id, defect_class, mask = _type_and_mask(filepath, dataset_id, dataset)
                anomaly_type = f"{texture_id}+{defect_class}"
                fn_mask = str(mask)
            except (ValueError, IndexError) as exc:
                reasons.append(f"ambiguous_type:{exc}")
        split = split_meta.get(filepath)
        if split is None:
            reasons.append("missing_split_metadata")
        if dataset_id not in recipe_types:
            reasons.append("missing_completed_checkpoint")
        elif anomaly_type not in recipe_types[dataset_id]:
            reasons.append("unsupported_adapter")
        spec_row = placement.get(anomaly_type)
        if spec_row is None:
            reasons.append("missing_defect_spec")
        elif spec_row.get("spatial_dependency") == "text" and not str(
            spec_row.get("roi_prompt_defect_location", "")
        ).strip():
            reasons.append("missing_text_prompt")
        if fn_mask and not Path(fn_mask).is_file():
            reasons.append("missing_fn_mask")
        clean_pool = pool_root / texture_id / "clean_image" if texture_id else Path()
        sampled_pool = pool_root / texture_id / "mask" / defect_class if texture_id else Path()
        if texture_id and not _images(clean_pool):
            reasons.append("empty_clean_pool")
        if texture_id and not _images(sampled_pool):
            reasons.append("empty_same_type_mask_pool")
        bbox_json = json.dumps(_jsonable(row["bbox"]), separators=(",", ":"))
        fn_id = "fn-" + _stable_id(dataset_id, str(row["image_id"]), filepath, bbox_json)
        ledger_rows.append(
            {
                "fn_id": fn_id,
                "box_order": int(box_order),
                "image_id": str(row["image_id"]),
                "filepath": filepath,
                "bbox_json": bbox_json,
                "od_category": str(row["class"]),
                "dataset_id": dataset_id,
                "texture_id": texture_id,
                "defect_class": defect_class,
                "anomaly_type": anomaly_type,
                "split": split.get("bucket", "") if split else "",
                "split_id": split.get("split_id", "") if split else "",
                "fn_mask_source": fn_mask,
                "clean_pool": str(clean_pool) if texture_id else "",
                "sampled_mask_pool": str(sampled_pool) if texture_id else "",
                "eligibility": not reasons,
                "skip_reason": ";".join(reasons),
                "source_tag": config["source_tag"],
            }
        )
    ledger = pd.DataFrame(ledger_rows)

    selection = config["selection"]
    selected_datasets = list(selection["datasets"])
    selection_mode = str(selection.get("mode", "per_dataset"))
    selected_rows = []
    if selection_mode == "all_eligible":
        eligible = ledger[
            ledger["dataset_id"].isin(selected_datasets) & ledger["eligibility"]
        ].sort_values(
            ["dataset_id", "image_id", "filepath", "bbox_json"], kind="stable"
        )
        if eligible.empty:
            raise ValueError(
                f"no eligible FNs found for selected datasets: {selected_datasets}"
            )
        selected_rows.extend(eligible.to_dict("records"))
    elif selection_mode == "per_dataset":
        selection_bucket = str(selection["split"])
        per_dataset = int(selection.get("per_dataset", 1))
        for dataset_id in selected_datasets:
            eligible = ledger[
                (ledger["dataset_id"] == dataset_id)
                & ledger["eligibility"]
                & (ledger["split"] == selection_bucket)
            ].sort_values(["image_id", "filepath", "bbox_json"], kind="stable")
            if len(eligible) < per_dataset:
                raise ValueError(
                    f"need {per_dataset} eligible {selection_bucket} FNs for "
                    f"{dataset_id}, found {len(eligible)}"
                )
            selected_rows.extend(eligible.head(per_dataset).to_dict("records"))
    else:
        raise ValueError(
            f"unsupported selection.mode {selection_mode!r}; expected per_dataset or all_eligible"
        )
    selected_ids = {row["fn_id"] for row in selected_rows}
    ledger["selected_for_smoke"] = ledger["fn_id"].isin(selected_ids)
    ledger.to_parquet(run / "manifests" / "fn_queries.parquet", index=False)

    seed = int(selection["mask_sample_seed"])
    mask_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    clean_seen: set[str] = set()
    query_rows: list[dict[str, Any]] = []
    for query_order, selected in enumerate(selected_rows):
        fn_id = selected["fn_id"]
        dataset_id = selected["dataset_id"]
        anomaly_type = selected["anomaly_type"]
        source_dir = run / "prepared_anomalygennext_inputs" / "source_masks" / fn_id
        isolated_path = source_dir / f"{fn_id}__fn_mask.png"
        fn_info = _isolate_fn_mask(
            Path(selected["fn_mask_source"]),
            Path(selected["filepath"]),
            json.loads(selected["bbox_json"]),
            isolated_path,
        )
        mask_rows.append(
            {
                "fn_id": fn_id,
                "query_order": query_order,
                "dataset_id": dataset_id,
                "anomaly_type": anomaly_type,
                "branch": "fn_mask",
                "pool_split": selected["split"],
                **fn_info,
            }
        )

        candidates = _images(Path(selected["sampled_mask_pool"]))
        stable_seed = int(_stable_id(seed, fn_id, anomaly_type), 16)
        sampled = candidates[random.Random(stable_seed).randrange(len(candidates))]
        sampled_path = source_dir / f"{fn_id}__same_type_sampled_mask.png"
        sampled_info = _copy_sampled_mask(sampled, sampled_path)
        mask_rows.append(
            {
                "fn_id": fn_id,
                "query_order": query_order,
                "dataset_id": dataset_id,
                "anomaly_type": anomaly_type,
                "branch": "same_type_sampled_mask",
                "pool_split": "train+mine",
                **sampled_info,
            }
        )
        query_rows.append(
            {
                "filepath": selected["filepath"],
                "fn_id": fn_id,
                "query_order": query_order,
                "dataset_id": dataset_id,
                "pool_key": selected["texture_id"],
                "anomaly_type": anomaly_type,
                "od_category": selected["od_category"],
            }
        )
        for image in _images(Path(selected["clean_pool"])):
            key = str(image)
            if key in clean_seen:
                continue
            clean_seen.add(key)
            clean_rows.append(
                {
                    "filepath": key,
                    "dataset_id": dataset_id,
                    "pool_key": selected["texture_id"],
                    "anomaly_type_eligibility": anomaly_type,
                    "source_tag": "verified_clean_train_pool",
                }
            )

    mask_selection = pd.DataFrame(mask_rows)
    clean_pool = pd.DataFrame(clean_rows)
    query_frame = pd.DataFrame(query_rows)
    query_input = query_frame.drop_duplicates("filepath", keep="first")
    mask_selection.to_parquet(run / "manifests" / "mask_selection.parquet", index=False)
    clean_pool.to_parquet(run / "manifests" / "clean_pool.parquet", index=False)
    query_input.to_parquet(run / "manifests" / "fn_embedding_inputs.parquet", index=False)
    query_frame.to_parquet(run / "manifests" / "selected_fn_queries.parquet", index=False)

    embedding = config["embedding"]
    base = {
        "input_parquet": str(run / "manifests" / "clean_pool.parquet"),
        "output_parquet": str(run / "embeddings" / "clean_embeddings.parquet"),
        "model": embedding["model"],
        "model_path": embedding["model_path"],
        "model_config_path": "",
        "batch_size": int(embedding["batch_size"]),
    }
    (run / "specs" / "clean_embeddings.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    query_spec = dict(base)
    query_spec["input_parquet"] = str(run / "manifests" / "fn_embedding_inputs.parquet")
    query_spec["output_parquet"] = str(run / "embeddings" / "fn_embeddings.parquet")
    (run / "specs" / "fn_embeddings.yaml").write_text(
        yaml.safe_dump(query_spec, sort_keys=False)
    )
    selected_types = {row["anomaly_type"] for row in selected_rows}
    _write_jsonl(
        run / "specs" / "defect_spec.jsonl",
        [placement[key] for key in sorted(selected_types)],
    )
    contract = {
        "phase": 1,
        "filtering_config": str(config_path),
        "filtering_config_sha256": _sha256(config_path),
        "source_tag": config["source_tag"],
        "gap_parquet": str(gaps_path),
        "split_root": config["split_root"],
        "selection": selection,
        "embedding": embedding,
        "retrieval": config["retrieval"],
        "selected_fns": [
            {
                key: row[key]
                for key in (
                    "fn_id",
                    "image_id",
                    "dataset_id",
                    "texture_id",
                    "defect_class",
                    "anomaly_type",
                    "split",
                    "filepath",
                )
            }
            for row in selected_rows
        ],
        "selected_fn_count": len(selected_rows),
        "clean_image_count": len(clean_pool),
        "source_mask_count": len(mask_selection),
        "training_pool_mutated": False,
    }
    _write_json(run / "prepared_anomalygennext_inputs" / "input_contract.json", contract)
    print(
        f"prepare inputs PASS: all_fn={len(ledger)} selected_fn={len(selected_rows)} "
        f"clean={len(clean_pool)} masks={len(mask_selection)}"
    )


def _matrix(series: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in series]
    if not rows or any(row.size == 0 or not np.isfinite(row).all() for row in rows):
        raise ValueError("empty or non-finite embedding row")
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(f"embedding widths differ: {sorted(widths)}")
    matrix = np.stack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding row")
    return matrix / norms


def build_knn_and_amp(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    run = Path(args.run_root)
    clean = pd.read_parquet(run / "embeddings" / "clean_embeddings.parquet").reset_index(
        drop=True
    )
    embedded_queries = pd.read_parquet(run / "embeddings" / "fn_embeddings.parquet")
    query_records = pd.read_parquet(run / "manifests" / "selected_fn_queries.parquet")
    mask_selection = pd.read_parquet(run / "manifests" / "mask_selection.parquet")
    for label, frame in (("clean", clean), ("query", embedded_queries)):
        if not {"filepath", "embedding"}.issubset(frame.columns):
            raise ValueError(f"{label} embedding parquet missing filepath/embedding")
    if embedded_queries["filepath"].duplicated().any():
        raise ValueError("FN embedding parquet contains duplicate filepath rows")
    queries = query_records.merge(
        embedded_queries[["filepath", "embedding"]],
        on="filepath",
        how="left",
        validate="many_to_one",
    )
    if queries["embedding"].isna().any():
        missing = sorted(queries.loc[queries["embedding"].isna(), "filepath"].astype(str).unique())
        raise ValueError(f"missing FN embeddings for selected query paths: {missing}")
    clean_matrix = _matrix(clean["embedding"])
    query_matrix = _matrix(queries["embedding"])
    retrieval = config["retrieval"]
    if str(retrieval.get("metric", "cosine")) != "cosine":
        raise ValueError("AnomalyGen input preparation currently requires retrieval.metric=cosine")
    topn = int(retrieval["candidate_topn"])
    min_similarity = float(retrieval["min_similarity"])
    excluded: set[str] = set()
    exclusion_path = retrieval.get("prior_clean_exclusion_manifest")
    if exclusion_path:
        exclusion = Path(exclusion_path)
        if exclusion.is_file():
            frame = pd.read_parquet(exclusion)
            excluded = set(map(str, frame["filepath"]))

    candidates: list[dict[str, Any]] = []
    amp_rows: list[dict[str, Any]] = []
    for query_position, query in queries.reset_index(drop=True).iterrows():
        pool_indices = clean.index[clean["pool_key"] == query["pool_key"]].to_numpy()
        if not len(pool_indices):
            raise ValueError(f"no clean embeddings for pool_key={query['pool_key']}")
        scores = clean_matrix[pool_indices] @ query_matrix[query_position]
        order = np.argsort(-scores, kind="stable")[:topn]
        branch_rows = mask_selection[mask_selection["fn_id"] == query["fn_id"]]
        if set(branch_rows["branch"]) != set(BRANCHES):
            raise ValueError(f"FN {query['fn_id']} does not have exactly the two required mask branches")
        for rank, local_position in enumerate(order, start=1):
            clean_position = int(pool_indices[int(local_position)])
            clean_row = clean.loc[clean_position]
            clean_path = str(clean_row["filepath"])
            similarity = float(scores[int(local_position)])
            gate_reason = ""
            if similarity < min_similarity:
                gate_reason = "below_similarity_floor"
            elif clean_path in excluded:
                gate_reason = "prior_iteration_exclusion"
            candidate_id = "pair-" + _stable_id(query["fn_id"], clean_path)
            candidate = {
                "candidate_id": candidate_id,
                "fn_id": str(query["fn_id"]),
                "query_order": int(query["query_order"]),
                "dataset_id": str(query["dataset_id"]),
                "anomaly_type": str(query["anomaly_type"]),
                "od_category": str(query["od_category"]),
                "pool_key": str(query["pool_key"]),
                "fn_filepath": str(query["filepath"]),
                "clean_filepath": clean_path,
                "neighbor_rank": rank,
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
                "encoder_model": config["embedding"]["model"],
                "encoder_model_path": config["embedding"]["model_path"],
                "eligible_for_amp": not gate_reason,
                "gate_reason": gate_reason,
                "source_tag": config["source_tag"],
            }
            candidates.append(candidate)
            if gate_reason:
                continue
            for _, branch in branch_rows.sort_values("branch").iterrows():
                amp_rows.append(
                    {
                        "clean_image": clean_path,
                        "defect_type": str(query["anomaly_type"]),
                        "submask": str(branch["isolated_mask"]),
                        "name": f"{candidate_id}__{branch['branch']}",
                        "cad_mask": None,
                        "cad_mask_label": None,
                        "n_seeds": 1,
                        "submask_split_largest": False,
                    }
                )
    pd.DataFrame(candidates).to_parquet(
        run / "manifests" / "knn_candidates.parquet", index=False
    )
    _write_json(run / "amp" / "amp_samples.json", amp_rows)
    print(
        f"KNN/AMP input PASS: candidates={len(candidates)} amp_rows={len(amp_rows)} "
        f"embedding_dim={clean_matrix.shape[1]}"
    )


def _validate_aligned_mask(mask_path: Path, image_path: Path) -> dict[str, Any]:
    mask = _mask_array(mask_path)
    with Image.open(image_path) as image:
        width, height = image.size
    if mask.shape != (height, width):
        raise ValueError(f"aligned mask/image dimensions differ: {mask_path} vs {image_path}")
    pixels = int(mask.sum())
    if pixels == width * height:
        raise ValueError(f"aligned mask covers the entire image: {mask_path}")
    return {"aligned_mask_pixels": pixels, "aligned_mask_sha256": _sha256(mask_path)}


def _amp_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        mask_path = Path(row["mask_filename"])
        source_stem = mask_path.name.split("__seed", 1)[0]
        key = (str(row["image_filename"]), source_stem)
        if key in result:
            raise ValueError(f"duplicate AMP output key: {key}")
        result[key] = row
    return result


def finalize_inputs(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    run = Path(args.run_root)
    candidates = pd.read_parquet(run / "manifests" / "knn_candidates.parquet")
    masks = pd.read_parquet(run / "manifests" / "mask_selection.parquet")
    amp_path = run / "amp" / "testcase.jsonl"
    if not amp_path.is_file():
        raise FileNotFoundError(amp_path)
    amp = _amp_index(amp_path)
    keep = int(config["retrieval"]["max_neighbors_per_fn"])
    status_rows: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    generator_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for fn_id in candidates.sort_values("query_order")["fn_id"].drop_duplicates():
        retained = 0
        # Uniqueness is scoped to one FN. Different false-negative boxes may
        # intentionally reuse the same nearest clean image; the pair id, masks,
        # AMP work, provenance, and generated outputs remain FN-specific.
        used_clean: set[str] = set()
        group = candidates[candidates["fn_id"] == fn_id].sort_values("neighbor_rank")
        branch_rows = masks[masks["fn_id"] == fn_id].set_index("branch")
        for _, candidate in group.iterrows():
            status = candidate.to_dict()
            status["selected"] = False
            status["selection_reason"] = str(candidate["gate_reason"])
            if retained >= keep:
                status["selection_reason"] = "quota_reached"
                status_rows.append(status)
                continue
            if not bool(candidate["eligible_for_amp"]):
                status_rows.append(status)
                continue
            clean_path = str(candidate["clean_filepath"])
            if clean_path in used_clean:
                status["selection_reason"] = "duplicate_clean_within_fn"
                status_rows.append(status)
                continue
            placed: dict[str, dict[str, Any]] = {}
            missing = []
            for branch_name in BRANCHES:
                source_path = Path(branch_rows.loc[branch_name, "isolated_mask"])
                row = amp.get((clean_path, source_path.stem))
                if row is None:
                    missing.append(branch_name)
                else:
                    placed[branch_name] = row
            if missing:
                status["selection_reason"] = "amp_failure:" + ",".join(missing)
                status_rows.append(status)
                continue
            pair_id = str(candidate["candidate_id"])
            selected = candidate.to_dict()
            selected["pair_id"] = pair_id
            invalid_aligned_mask = None
            for branch_name, row in placed.items():
                try:
                    info = _validate_aligned_mask(
                        Path(row["mask_filename"]), Path(clean_path)
                    )
                except ValueError as exc:
                    message = str(exc)
                    if message.startswith("aligned mask covers the entire image:"):
                        code = "full_image"
                    elif message.startswith("aligned mask/image dimensions differ:"):
                        code = "dimension_mismatch"
                    else:
                        code = "invalid"
                    invalid_aligned_mask = f"invalid_aligned_mask:{branch_name}:{code}"
                    break
                selected[f"{branch_name}_aligned_mask"] = str(row["mask_filename"])
                selected[f"{branch_name}_aligned_mask_sha256"] = info["aligned_mask_sha256"]
            if invalid_aligned_mask:
                status["selection_reason"] = invalid_aligned_mask
                status_rows.append(status)
                continue
            selected_pairs.append(selected)
            used_clean.add(clean_path)
            retained += 1
            status["selected"] = True
            status["selection_reason"] = "selected"
            status_rows.append(status)
            dataset_id = str(candidate["dataset_id"])
            for branch_name in BRANCHES:
                amp_row = dict(placed[branch_name])
                amp_row["anomaly_type"] = str(candidate["anomaly_type"])
                amp_row["num_generated_images"] = 1
                generation_index = len(generator_rows[dataset_id])
                generator_rows[dataset_id].append(amp_row)
                source = branch_rows.loc[branch_name]
                provenance_rows[dataset_id].append(
                    {
                        "generation_index": generation_index,
                        "pair_id": pair_id,
                        "fn_id": str(fn_id),
                        "dataset_id": dataset_id,
                        "anomaly_type": str(candidate["anomaly_type"]),
                        "od_category": str(candidate["od_category"]),
                        "mask_branch": branch_name,
                        "source_mask": str(source["source_mask"]),
                        "source_mask_copy": str(source["isolated_mask"]),
                        "source_mask_sha256": str(source["content_sha256"]),
                        "fn_filepath": str(candidate["fn_filepath"]),
                        "clean_filepath": clean_path,
                        "aligned_mask": str(amp_row["mask_filename"]),
                        "neighbor_rank": int(candidate["neighbor_rank"]),
                        "cosine_similarity": float(candidate["cosine_similarity"]),
                        "source_tag": config["source_tag"],
                    }
                )

    status_frame = pd.DataFrame(status_rows)
    selected_frame = pd.DataFrame(selected_pairs)
    status_frame.to_parquet(run / "manifests" / "knn_roi_status.parquet", index=False)
    selected_frame.to_parquet(run / "manifests" / "selected_pairs.parquet", index=False)

    plan_rows = []
    artifacts = []
    unified_rows = []
    for dataset_id in sorted(generator_rows):
        dataset = config["datasets"][dataset_id]
        directory = run / "prepared_anomalygennext_inputs" / "anomalygen_inputs" / dataset_id
        testcase = directory / "testcase.jsonl"
        provenance = directory / "provenance.jsonl"
        _write_jsonl(testcase, generator_rows[dataset_id])
        _write_jsonl(provenance, provenance_rows[dataset_id])
        types = sorted({row["anomaly_type"] for row in provenance_rows[dataset_id]})
        plan = {
            "dataset_id": dataset_id,
            "anomaly_type": types[0] if len(types) == 1 else ",".join(types),
            "anomaly_types": types,
            "testcase": str(testcase),
            "provenance": str(provenance),
            "checkpoint": str(dataset["checkpoint"]),
            "recipe": str(dataset["recipe"]),
            "real_root": str(config["pool_dataset_root"]),
            "requested_rows": len(generator_rows[dataset_id]),
        }
        plan_rows.append(plan)
        for path in (testcase, provenance):
            artifacts.append({"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
        for testcase_row, provenance_row in zip(
            generator_rows[dataset_id], provenance_rows[dataset_id], strict=True
        ):
            unified_rows.append(
                {
                    **provenance_row,
                    "checkpoint": str(dataset["checkpoint"]),
                    "recipe": str(dataset["recipe"]),
                    "generator_input": testcase_row,
                }
            )
    unified = run / "prepared_anomalygennext_inputs" / "anomalygen_inputs.jsonl"
    _write_jsonl(unified, unified_rows)
    artifacts.append({"path": str(unified), "sha256": _sha256(unified), "bytes": unified.stat().st_size})
    anomalygen_next_generation_plan_path = (
        run / "prepared_anomalygennext_inputs" / "anomalygen_next_generation_plan.json"
    )
    _write_json(anomalygen_next_generation_plan_path, plan_rows)
    for path in (
        Path(args.config).resolve(),
        run / "prepared_anomalygennext_inputs" / "input_contract.json",
        anomalygen_next_generation_plan_path,
    ):
        artifacts.append(
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    manifest = {
        "schema_version": 2,
        "phase": "prepared_anomalygennext_inputs",
        "status": "COMPLETE",
        "source_tag": config["source_tag"],
        "selected_fn_count": int(selected_frame["fn_id"].nunique()) if not selected_frame.empty else 0,
        "selected_pair_count": len(selected_frame),
        "generator_row_count": len(unified_rows),
        "generator_groups": plan_rows,
        "artifacts": artifacts,
        "skip_counts": dict(Counter(status_frame["selection_reason"])),
        "generation_ready": bool(unified_rows),
        "training_pool_mutated": False,
    }
    _write_json(
        run / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json",
        manifest,
    )
    if not manifest["generation_ready"]:
        raise RuntimeError("prepared inputs contain no AnomalyGenNext generation rows")
    print(
        f"finalize prepared inputs PASS: fn={manifest['selected_fn_count']} "
        f"pairs={manifest['selected_pair_count']} generator_rows={manifest['generator_row_count']}"
    )


def _validate_prepared_inputs_root(root: Path) -> dict[str, Any]:
    manifest_path = (
        root / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or not manifest.get("generation_ready"):
        raise ValueError("prepared-input manifest is not COMPLETE/generation_ready")
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"prepared-input artifact changed or is missing: {path}")
    return manifest


def validate_prepared_inputs(args: argparse.Namespace) -> None:
    root = Path(args.prepared_inputs_root)
    manifest = _validate_prepared_inputs_root(root)
    manifest_path = (
        root / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    )


def _aoi_dataset_for_path(path: str) -> str:
    matches = [dataset for marker, dataset in AOI_DATASET_MARKERS.items() if marker in path]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one supported AOI dataset marker in {path!r}")
    return matches[0]


def materialize_aoi_plan(args: argparse.Namespace) -> None:
    """Turn an AOI synthetic plan into this skill's complete host input config."""
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    split_root = output / "split_manifests"
    split_root.mkdir()
    coco = json.loads(Path(args.kpi_coco).read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    plan = {str(key): int(value) for key, value in json.loads(Path(args.synthetic_plan).read_text()).items()}
    all_gaps = pd.read_parquet(args.strict_gaps).copy()
    if "gap_type" not in all_gaps:
        raise ValueError("strict gap input lacks gap_type")
    gap_types = all_gaps["gap_type"].astype(str).str.upper()
    unknown = sorted(set(gap_types) - {"TP", "FP", "FN"})
    if unknown:
        raise ValueError(f"strict gap input contains unknown gap types: {unknown}")
    gaps = all_gaps[gap_types.eq("FN")].copy()
    if gaps.empty:
        raise ValueError("strict gap input contains no FN rows")
    gaps["generator_type"] = gaps["image_id"].map(
        lambda value: images[int(value)]["deft_od_aoi"]["generator_type"]
    )
    gaps["filepath"] = gaps["image_id"].map(lambda value: images[int(value)]["source_path"])
    gaps["bbox_sort"] = gaps["bbox"].map(
        lambda value: json.dumps([float(item) for item in value], separators=(",", ":"))
    )
    selected_parts: list[pd.DataFrame] = []
    report_rows = []
    for anomaly_type, requested in sorted(plan.items()):
        available = gaps[gaps["generator_type"] == anomaly_type].sort_values(
            ["image_id", "filepath", "bbox_sort"], kind="stable"
        )
        # Every FN produces the two required independent mask branches.
        pair_count = min(len(available), requested // 2)
        selected_parts.append(available.head(pair_count))
        report_rows.append(
            {
                "anomaly_type": anomaly_type,
                "available_fn_boxes": int(len(available)),
                "requested_images": requested,
                "selected_pairs": int(pair_count),
                "frozen_generator_rows": int(pair_count * 2),
                "bounded_shortfall": int(requested - pair_count * 2),
            }
        )
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        raise ValueError("synthetic plan yielded no pair-preserving rows")
    selected = selected.drop(columns=["generator_type", "bbox_sort"])
    frozen_gaps = output / "strict_fn_synthesis_subset.parquet"
    selected.to_parquet(frozen_gaps, index=False)
    split_rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in selected.to_dict("records"):
        source = str(row["filepath"])
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        dataset = _aoi_dataset_for_path(source)
        split_rows[dataset][f"image-{int(row['image_id'])}"] = {"source_path": source, "bucket": "kpi"}
    for dataset, rows in split_rows.items():
        directory = split_root / dataset
        directory.mkdir()
        _write_json(directory / "split_manifest.json", rows)
    checkpoint = str(Path(args.checkpoint).resolve())
    recipe = str(Path(args.recipe).resolve())
    common = {"checkpoint": checkpoint, "recipe": recipe}
    datasets = {
        "mvtec": {**common, "path_marker": "MVTec-AD", "texture_offset": 1, "texture_prefix": "mvtec_", "split_components": ["test"], "mask_style": "stem_mask_png"},
        "visa": {**common, "path_marker": "VisA", "texture_offset": 1, "texture_prefix": "visa_", "split_components": ["test"], "mask_style": "stem_png", "mask_extensions": [".png"]},
        "dagm": {**common, "path_marker": "DAGM_2007", "texture_offset": 2, "texture_prefix": "dagm_", "split_components": ["Test", "Train"], "defect_class_fixed": "defect", "mask_component_replacements": {"images": "masks"}, "mask_style": "configurable", "mask_suffix": "_label", "mask_extensions": [".PNG", ".png"]},
        "btad": {**common, "path_marker": "BTAD", "texture_offset": 2, "texture_prefix": "btad_", "split_components": ["test"], "defect_class_fixed": "ko", "mask_component_replacements": {"images": "masks", "test": "ground_truth"}, "mask_style": "stem_png", "mask_extensions": [".png", ".bmp"]},
        "mpdd": {**common, "path_marker": "MPDD", "texture_offset": 1, "texture_prefix": "mpdd_", "split_components": ["test"], "mask_style": "stem_mask_png"},
    }
    active = sorted(split_rows)
    config = {
        "source_tag": f"deft_od_aoi_iter{args.iteration}_frozen_synthetic_plan",
        "gap_parquet": str(frozen_gaps),
        "split_root": str(split_root),
        "pool_dataset_root": str(Path(args.pool_root).resolve()),
        "defect_spec": str(Path(args.defect_spec).resolve()),
        "datasets": {key: datasets[key] for key in active},
        "selection": {"mode": "all_eligible", "datasets": active, "mask_sample_seed": args.mask_sample_seed},
        "embedding": {"model": "SigLIP", "model_path": str(Path(args.embedding_model_path).resolve()), "batch_size": args.embedding_batch_size},
        "retrieval": {"metric": "cosine", "candidate_topn": args.candidate_topn, "max_neighbors_per_fn": 1, "min_similarity": args.min_similarity, "prior_clean_exclusion_manifest": ""},
    }
    config_path = output / "filtering_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "status": "COMPLETE",
        "iteration": int(args.iteration),
        "selected_fn_pairs": int(len(selected)),
        "input_gap_rows": int(len(all_gaps)),
        "input_fn_rows": int(len(gaps)),
        "frozen_generator_rows": int(len(selected) * 2),
        "bounded_shortfall": int(sum(plan.values()) - len(selected) * 2),
        "active_datasets": active,
        "filtering_config": str(config_path),
        "frozen_gaps": str(frozen_gaps),
        "per_type": report_rows,
    }
    _write_json(output / "selection_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_type"}, indent=2))


def stage_embedding(args: argparse.Namespace) -> None:
    spec = yaml.safe_load(Path(args.source_spec).read_text(encoding="utf-8"))
    frame = pd.read_parquet(spec["input_parquet"]).copy()
    if frame.empty or "filepath" not in frame:
        raise ValueError("embedding input is empty or lacks filepath")
    images = Path(args.stage_root) / "images"
    images.mkdir(parents=True, exist_ok=True)
    mapping = []
    local_paths = []
    for index, durable in enumerate(frame["filepath"].astype(str)):
        source = Path(durable)
        if not source.is_file():
            raise FileNotFoundError(source)
        local = images / f"{index:07d}{source.suffix.lower()}"
        shutil.copy2(source, local)
        mapping.append({"local_filepath": str(local), "durable_filepath": durable})
        local_paths.append(str(local))
    frame["filepath"] = local_paths
    local_input = Path(args.stage_root) / "input.parquet"
    frame.to_parquet(local_input, index=False)
    pd.DataFrame(mapping).to_parquet(args.mapping, index=False)
    spec.update({"input_parquet": str(local_input), "output_parquet": str(Path(args.local_output)), "model_path": str(Path(args.model_path))})
    Path(args.output_spec).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    print(f"staged_embedding_inputs={len(frame)}")


def restore_embedding(args: argparse.Namespace) -> None:
    frame = pd.read_parquet(args.embedding_parquet).copy()
    mapping = pd.read_parquet(args.mapping)
    lookup = dict(zip(mapping["local_filepath"].astype(str), mapping["durable_filepath"].astype(str)))
    missing = sorted(set(frame["filepath"].astype(str)) - set(lookup))
    if missing:
        raise ValueError(f"embedding outputs contain unmapped local paths: {missing[:5]}")
    frame["filepath"] = frame["filepath"].astype(str).map(lookup)
    if frame.empty or "embedding" not in frame:
        raise ValueError("embedding output is empty or lacks embedding")
    target = Path(args.durable_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    if frame["filepath"].astype(str).str.startswith("/raid/scratch/").any():
        raise ValueError("durable embedding parquet still references node-local scratch")
    print(f"restored_embedding_rows={len(frame)}")


def _stable_local_name(path: str) -> str:
    source = Path(path)
    return f"{hashlib.sha256(path.encode()).hexdigest()[:20]}{source.suffix.lower()}"


def stage_amp(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.amp_samples).read_text(encoding="utf-8"))
    durable_root = Path(args.durable_run_root).resolve()
    scratch_root = Path(args.scratch_run_root).resolve()
    hot = Path(args.hot_root).resolve()
    clean_root, mask_root = hot / "clean", hot / "masks"
    clean_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    clean_map: dict[str, str] = {}
    rewritten = []
    for row in rows:
        item = dict(row)
        durable_clean = str(row["clean_image"])
        local_clean = clean_root / _stable_local_name(durable_clean)
        if not local_clean.is_file():
            shutil.copy2(durable_clean, local_clean)
        clean_map[str(local_clean)] = durable_clean
        item["clean_image"] = str(local_clean)
        durable_mask = Path(str(row["submask"])).resolve()
        try:
            local_mask = scratch_root / durable_mask.relative_to(durable_root)
            if not local_mask.is_file():
                raise FileNotFoundError(local_mask)
        except ValueError:
            local_mask = mask_root / _stable_local_name(str(durable_mask))
            if not local_mask.is_file():
                shutil.copy2(durable_mask, local_mask)
        item["submask"] = str(local_mask)
        rewritten.append(item)
    Path(args.output_json).write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
    _write_json(Path(args.mapping_json), {"clean": clean_map, "rows": len(rewritten)})
    print(f"staged_amp_rows={len(rewritten)} unique_clean={len(clean_map)}")


def restore_amp(args: argparse.Namespace) -> None:
    mapping = json.loads(Path(args.mapping_json).read_text(encoding="utf-8"))["clean"]
    local_amp = Path(args.local_amp_root).resolve()
    durable_amp = Path(args.durable_amp_root).resolve()
    restored = []
    for row in _read_jsonl(Path(args.source_testcase)):
        local_image = str(Path(str(row["image_filename"])).resolve())
        if local_image not in mapping:
            raise ValueError(f"AMP testcase has unmapped clean image: {local_image}")
        row["image_filename"] = mapping[local_image]
        local_mask = Path(str(row["mask_filename"])).resolve()
        try:
            relative = local_mask.relative_to(local_amp)
        except ValueError as exc:
            raise ValueError(f"AMP mask is outside local AMP root: {local_mask}") from exc
        durable_mask = durable_amp / relative
        if not durable_mask.is_file():
            raise FileNotFoundError(durable_mask)
        row["mask_filename"] = str(durable_mask)
        restored.append(row)
    if not restored:
        raise ValueError("AMP testcase contains no rows")
    _write_jsonl(Path(args.durable_testcase), restored)
    print(f"restored_amp_testcase_rows={len(restored)}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("prepare-inputs")
    command.add_argument("--config", required=True)
    command.add_argument("--run-root", required=True)
    command.set_defaults(func=prepare_inputs)

    command = commands.add_parser("build-knn-and-amp")
    command.add_argument("--config", required=True)
    command.add_argument("--run-root", required=True)
    command.set_defaults(func=build_knn_and_amp)

    command = commands.add_parser("finalize-inputs")
    command.add_argument("--config", required=True)
    command.add_argument("--run-root", required=True)
    command.set_defaults(func=finalize_inputs)

    command = commands.add_parser("validate-prepared-inputs")
    command.add_argument("--prepared-inputs-root", required=True)
    command.set_defaults(func=validate_prepared_inputs)

    command = commands.add_parser("materialize-aoi-plan")
    command.add_argument("--iteration", type=int, required=True)
    command.add_argument("--kpi-coco", required=True)
    command.add_argument("--strict-gaps", required=True)
    command.add_argument("--synthetic-plan", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--pool-root", required=True)
    command.add_argument("--defect-spec", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--recipe", required=True)
    command.add_argument("--embedding-model-path", required=True)
    command.add_argument("--candidate-topn", type=int, default=3)
    command.add_argument("--min-similarity", type=float, default=-1.0)
    command.add_argument("--mask-sample-seed", type=int, default=43)
    command.add_argument("--embedding-batch-size", type=int, default=64)
    command.set_defaults(func=materialize_aoi_plan)

    command = commands.add_parser("stage-embedding")
    command.add_argument("--source-spec", required=True)
    command.add_argument("--stage-root", required=True)
    command.add_argument("--model-path", required=True)
    command.add_argument("--local-output", required=True)
    command.add_argument("--output-spec", required=True)
    command.add_argument("--mapping", required=True)
    command.set_defaults(func=stage_embedding)

    command = commands.add_parser("restore-embedding")
    command.add_argument("--embedding-parquet", required=True)
    command.add_argument("--mapping", required=True)
    command.add_argument("--durable-output", required=True)
    command.set_defaults(func=restore_embedding)

    command = commands.add_parser("stage-amp")
    command.add_argument("--amp-samples", required=True)
    command.add_argument("--durable-run-root", required=True)
    command.add_argument("--scratch-run-root", required=True)
    command.add_argument("--hot-root", required=True)
    command.add_argument("--output-json", required=True)
    command.add_argument("--mapping-json", required=True)
    command.set_defaults(func=stage_amp)

    command = commands.add_parser("restore-amp")
    command.add_argument("--source-testcase", required=True)
    command.add_argument("--durable-testcase", required=True)
    command.add_argument("--mapping-json", required=True)
    command.add_argument("--local-amp-root", required=True)
    command.add_argument("--durable-amp-root", required=True)
    command.set_defaults(func=restore_amp)

    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
