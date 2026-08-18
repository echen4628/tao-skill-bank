#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route DEFT OD AOI loose-FP and strict-FN gaps into admitted data requests.

The script is intentionally data-only. It selects existing real and clean
images and emits a synthetic dose plan; it does not mutate a training set or
launch generation. Persistent ledgers are emitted into the current output
directory and become inputs only after the caller commits the route stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from deft_od_aoi_policy import load_policy, uniform_mine_for_iteration

DCT_SIZE = 32
DCT_KEEP = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--loose-gaps", required=True)
    parser.add_argument("--strict-gaps", required=True)
    parser.add_argument("--kpi-coco", required=True)
    parser.add_argument("--kpi-images-dir", required=True)
    parser.add_argument("--source-coco", required=True)
    parser.add_argument("--source-images-dir", required=True)
    parser.add_argument("--clean-coco", required=True)
    parser.add_argument("--clean-images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-defect-ledger", default=None)
    parser.add_argument("--previous-clean-ledger", default=None)
    parser.add_argument("--previous-admission-index", default=None)
    parser.add_argument("--conversion-old-strict", default=None)
    parser.add_argument("--conversion-new-strict", default=None)
    parser.add_argument("--prior-admitted-synthetic", type=int, default=0)
    parser.add_argument(
        "--valid-generator-types",
        default=None,
        help="JSON array of configured generator types; required when synthesis is enabled",
    )
    return parser.parse_args()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_ledger(path: str | None) -> set[str]:
    if not path:
        return set()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ledger does not exist: {resolved}")
    values = _read_json(resolved)
    if not isinstance(values, list):
        raise ValueError(f"ledger must be a JSON array: {resolved}")
    return {str(Path(str(value)).expanduser().absolute()) for value in values}


def _metadata(image: dict, *, require_defect: bool) -> dict[str, str]:
    nested = image.get("deft_od_aoi") if isinstance(image.get("deft_od_aoi"), dict) else {}

    def value(key: str) -> str:
        raw = nested.get(key, image.get(key))
        return str(raw).strip() if raw is not None else ""

    result = {
        "benchmark": value("benchmark"),
        "texture": value("texture"),
        "defect_type": value("defect_type"),
        "generator_type": value("generator_type"),
    }
    required = ("benchmark", "texture", "defect_type") if require_defect else (
        "benchmark",
        "texture",
    )
    missing = [key for key in required if not result[key]]
    if missing:
        raise ValueError(
            f"COCO image {image.get('file_name')!r} lacks DEFT OD AOI metadata {missing}"
        )
    return result


def _pocket(meta: dict[str, str]) -> tuple[str, str, str]:
    return meta["benchmark"], meta["texture"], meta["defect_type"]


def _texture(meta: dict[str, str]) -> tuple[str, str]:
    return meta["benchmark"], meta["texture"]


def _pocket_name(pocket: tuple[str, str, str]) -> str:
    return "/".join(pocket)


def _source_path(images_dir: Path, image: dict) -> str:
    raw = image.get("source_path")
    path = Path(str(raw)).expanduser() if raw else images_dir / str(image.get("file_name", ""))
    if not path.is_absolute():
        path = images_dir / path
    return str(path.expanduser().absolute())


def _gap_frame(path: str, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(Path(path).expanduser().resolve())
    required = {"gap_type", "filepath", "bbox", "best_iou"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} gap parquet lacks columns {missing}")
    return frame


def _bbox_xyxy(value: object) -> tuple[float, float, float, float]:
    box = [float(item) for item in value]
    if len(box) != 4:
        raise ValueError(f"expected four box coordinates, got {box}")
    return box[0], box[1], box[2], box[3]


def _bbox_xywh_to_xyxy(value: object) -> tuple[float, float, float, float]:
    x, y, width, height = [float(item) for item in value]
    return x, y, x + width, y + height


def _dct_basis(size: int) -> np.ndarray:
    positions = np.arange(size)
    return np.cos(
        np.pi
        * (2 * positions[None, :] + 1)
        * positions[:, None]
        / (2 * size)
    )


_BASIS = _dct_basis(DCT_SIZE)


def _dct_signature(image: Image.Image) -> np.ndarray:
    array = np.asarray(
        image.convert("L").resize((DCT_SIZE, DCT_SIZE), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    vector = (_BASIS @ array @ _BASIS.T)[:DCT_KEEP, :DCT_KEEP].ravel()
    vector[0] = 0.0
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros(DCT_KEEP * DCT_KEEP)


def _crop_signature(path: str, boxes_xyxy: Iterable[tuple[float, float, float, float]]) -> np.ndarray | None:
    boxes = list(boxes_xyxy)
    if not boxes:
        return None
    try:
        with Image.open(path) as image:
            x0 = min(box[0] for box in boxes)
            y0 = min(box[1] for box in boxes)
            x1 = max(box[2] for box in boxes)
            y1 = max(box[3] for box in boxes)
            pad_x = 0.2 * (x1 - x0)
            pad_y = 0.2 * (y1 - y0)
            crop = image.crop(
                (
                    max(0, x0 - pad_x),
                    max(0, y0 - pad_y),
                    min(image.width, x1 + pad_x),
                    min(image.height, y1 + pad_y),
                )
            )
            if crop.width < 8 or crop.height < 8:
                return None
            return _dct_signature(crop)
    except OSError:
        return None


def _combined_signature(path: str, boxes_xywh: list[list[float]]) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            global_signature = _dct_signature(image)
            if boxes_xywh:
                boxes_xyxy = [_bbox_xywh_to_xyxy(box) for box in boxes_xywh]
                defect_signature = _crop_signature(path, boxes_xyxy)
                if defect_signature is None:
                    defect_signature = global_signature
                x0 = min(box[0] for box in boxes_xyxy)
                y0 = min(box[1] for box in boxes_xyxy)
                x1 = max(box[2] for box in boxes_xyxy)
                y1 = max(box[3] for box in boxes_xyxy)
                position = np.array(
                    [
                        (x0 + x1) / (2 * max(1, image.width)),
                        (y0 + y1) / (2 * max(1, image.height)),
                        (x1 - x0) / max(1, image.width),
                        (y1 - y0) / max(1, image.height),
                    ],
                    dtype=np.float64,
                )
            else:
                defect_signature = global_signature
                position = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float64)
            return np.concatenate([global_signature, defect_signature, position])
    except OSError:
        return None


def _iou_xywh(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    ix = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    iy = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = ix * iy
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


class Admission:
    def __init__(self, policy: dict, previous_index: str | None):
        self.policy = policy["admission"]
        dimensions = 2 * DCT_KEEP * DCT_KEEP + 4
        if previous_index:
            path = Path(previous_index).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"admission index does not exist: {path}")
            self.index = np.load(path)
            if self.index.ndim != 2 or self.index.shape[1] != dimensions:
                raise ValueError(f"invalid admission index shape {self.index.shape}")
        else:
            self.index = np.zeros((0, dimensions), dtype=np.float64)
        self.pending: list[np.ndarray] = []
        self.report = Counter()

    def _pool(self) -> np.ndarray:
        if not self.pending:
            return self.index
        return np.vstack([self.index, *[value[None, :] for value in self.pending]])

    def _duplicate(self, signature: np.ndarray, clean: bool) -> bool:
        pool = self._pool()
        if not len(pool):
            return False
        width = DCT_KEEP * DCT_KEEP
        global_similarity = pool[:, :width] @ signature[:width]
        previous_clean = pool[:, 2 * width] < 0
        if clean:
            return bool(
                np.any(
                    previous_clean
                    & (
                        global_similarity
                        > float(self.policy["clean_duplicate_global_cosine"])
                    )
                )
            )
        defect_similarity = pool[:, width : 2 * width] @ signature[width : 2 * width]
        position_distance = np.abs(
            pool[:, 2 * width : 2 * width + 2]
            - signature[2 * width : 2 * width + 2]
        ).max(axis=1)
        return bool(
            np.any(
                (~previous_clean)
                & (global_similarity > float(self.policy["duplicate_global_cosine"]))
                & (
                    defect_similarity
                    > float(self.policy["duplicate_defect_cosine"])
                )
                & (
                    position_distance
                    < float(self.policy["duplicate_position_delta"])
                )
            )
        )

    def _screen_boxes(self, boxes: list[list[float]]) -> list[list[float]]:
        kept: list[list[float]] = []
        for box in boxes:
            _, _, width, height = [float(value) for value in box]
            bad = width * height < float(self.policy["minimum_box_area_px"])
            bad = bad or max(width, height) / max(1.0e-6, min(width, height)) > float(
                self.policy["maximum_box_aspect"]
            )
            if not bad:
                bad = any(
                    _iou_xywh(box, previous) > float(self.policy["duplicate_gt_iou"])
                    for previous in kept
                )
            if bad:
                self.report["boxes_quarantined"] += 1
            else:
                kept.append([float(value) for value in box])
        return kept or boxes

    def admit(self, candidates: list[dict], quota: int, *, clean: bool) -> list[dict]:
        admitted: list[dict] = []
        clusters: list[list[Any]] = []
        minimum_cap = int(self.policy["clean_cluster_minimum_cap"])
        divisor = int(self.policy["clean_cluster_quota_divisor"])
        cluster_cap = max(minimum_cap, quota // divisor) if quota else minimum_cap
        for candidate in candidates:
            if len(admitted) >= quota:
                break
            self.report["checked"] += 1
            signature = _combined_signature(candidate["source_path"], candidate["boxes"])
            if signature is not None and self._duplicate(signature, clean):
                self.report["rejected_duplicate"] += 1
                continue
            if clean and signature is not None:
                global_signature = signature[: DCT_KEEP * DCT_KEEP]
                cluster = next(
                    (
                        item
                        for item in clusters
                        if float(item[0] @ global_signature)
                        > float(self.policy["clean_cluster_cosine"])
                    ),
                    None,
                )
                if cluster is not None:
                    if cluster[1] >= cluster_cap:
                        self.report["rejected_cluster_cap"] += 1
                        continue
                    cluster[1] += 1
                else:
                    clusters.append([global_signature, 1])
            if not clean:
                candidate = {**candidate, "boxes": self._screen_boxes(candidate["boxes"])}
            admitted.append(candidate)
            if signature is not None:
                self.pending.append(signature)
        self.report["admitted"] += len(admitted)
        return admitted

    def save(self, path: Path) -> None:
        values = self._pool()
        temporary = Path(str(path) + ".tmp.npy")
        np.save(temporary, values)
        os.replace(temporary, path)


def _index_coco(coco: dict, images_dir: Path, *, clean: bool) -> dict:
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO requires images and annotations arrays")
    annotations_by_image: defaultdict[Any, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation.get("image_id")].append(annotation)
    records = []
    for image in images:
        metadata = _metadata(image, require_defect=not clean)
        boxes = [
            [float(value) for value in annotation["bbox"]]
            for annotation in annotations_by_image.get(image.get("id"), [])
        ]
        if clean and boxes:
            raise ValueError(f"clean COCO image {image.get('file_name')!r} has annotations")
        if not clean and not boxes:
            continue
        path = _source_path(images_dir, image)
        if not Path(path).is_file():
            raise FileNotFoundError(f"COCO image is missing: {path}")
        record = {
            "source_path": path,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "boxes": boxes,
            "benchmark": metadata["benchmark"],
            "texture": metadata["texture"],
            "defect_type": metadata["defect_type"],
            "generator_type": metadata["generator_type"],
            "kind": "clean_negative" if clean else "real_defect",
        }
        records.append(record)
    return {"records": records, "annotations_by_image": annotations_by_image}


def _kpi_index(coco: dict) -> dict[str, dict]:
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("KPI COCO requires images and annotations")
    annotated_ids = {annotation.get("image_id") for annotation in annotations}
    result = {}
    for image in images:
        metadata = _metadata(
            image,
            require_defect=image.get("id") in annotated_ids,
        )
        result[Path(str(image.get("file_name", ""))).name] = {
            "image": image,
            "metadata": metadata,
        }
    return result


def _lookup_kpi(row: pd.Series, index: dict[str, dict]) -> dict | None:
    return index.get(Path(str(row["filepath"])).name)


def _strict_identities(
    frame: pd.DataFrame, kpi: dict[str, dict]
) -> dict[tuple[str, str, str], set[tuple[str, tuple[float, ...]]]]:
    output: defaultdict[tuple[str, str, str], set] = defaultdict(set)
    for _, row in frame[frame["gap_type"].astype(str).str.upper() == "FN"].iterrows():
        info = _lookup_kpi(row, kpi)
        if info is None:
            continue
        identity = (
            Path(str(row["filepath"])).stem,
            tuple(round(value, 1) for value in _bbox_xyxy(row["bbox"])),
        )
        output[_pocket(info["metadata"])].add(identity)
    return output


def _conversion_rates(
    old_path: str | None,
    new_path: str | None,
    kpi: dict[str, dict],
    minimum_trackable: int,
) -> dict[tuple[str, str, str], float]:
    if not old_path or not new_path:
        return {}
    old = _strict_identities(_gap_frame(old_path, "conversion-old"), kpi)
    new = _strict_identities(_gap_frame(new_path, "conversion-new"), kpi)
    rates = {}
    for pocket, identities in old.items():
        if len(identities) >= minimum_trackable:
            still_present = len(identities & new.get(pocket, set()))
            rates[pocket] = 1.0 - still_present / len(identities)
    return rates


def _scale_counts(counts: dict[str, int], allowed: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= allowed:
        return counts
    if allowed <= 0:
        return {}
    raw = {key: value * allowed / total for key, value in counts.items()}
    scaled = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = allowed - sum(scaled.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - scaled[key]), key))
    for key in order[:remaining]:
        scaled[key] += 1
    return {key: value for key, value in scaled.items() if value > 0}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def route(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    if not 1 <= args.iteration <= int(policy["max_iterations"]):
        raise ValueError("iteration is outside the frozen policy range")
    if args.prior_admitted_synthetic < 0:
        raise ValueError("prior_admitted_synthetic cannot be negative")

    loose = _gap_frame(args.loose_gaps, "loose")
    strict = _gap_frame(args.strict_gaps, "strict")
    kpi_coco = _read_json(args.kpi_coco)
    kpi = _kpi_index(kpi_coco)
    kpi_images_dir = Path(args.kpi_images_dir).expanduser().resolve()
    positive_index = _index_coco(
        _read_json(args.source_coco),
        Path(args.source_images_dir).expanduser().resolve(),
        clean=False,
    )
    clean_index = _index_coco(
        _read_json(args.clean_coco),
        Path(args.clean_images_dir).expanduser().resolve(),
        clean=True,
    )

    used_defect = _load_ledger(args.previous_defect_ledger)
    used_clean = _load_ledger(args.previous_clean_ledger)
    admission = Admission(policy, args.previous_admission_index)

    positives_by_pocket: defaultdict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in positive_index["records"]:
        positives_by_pocket[(record["benchmark"], record["texture"], record["defect_type"])].append(record)
    clean_by_texture: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in clean_index["records"]:
        clean_by_texture[(record["benchmark"], record["texture"])].append(record)
    for records in [*positives_by_pocket.values(), *clean_by_texture.values()]:
        records.sort(key=lambda record: record["source_path"])

    strict_fns = strict[strict["gap_type"].astype(str).str.upper() == "FN"]
    fn_counts: Counter[tuple[str, str, str]] = Counter()
    fn_refs: defaultdict[tuple[str, str, str], list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    generator_by_pocket: dict[tuple[str, str, str], str] = {}
    unresolved_strict = 0
    for _, row in strict_fns.iterrows():
        info = _lookup_kpi(row, kpi)
        if info is None:
            unresolved_strict += 1
            continue
        pocket = _pocket(info["metadata"])
        fn_counts[pocket] += 1
        fn_refs[pocket].append((str(info["image"]["file_name"]), _bbox_xyxy(row["bbox"])))
        generator_by_pocket[pocket] = info["metadata"]["generator_type"] or _pocket_name(pocket)

    synthetic_policy = policy["synthetic"]
    valid_generator_types: set[str] = set()
    if bool(synthetic_policy["enabled"]):
        if not args.valid_generator_types:
            raise ValueError(
                "valid_generator_types is required when synthesis is enabled"
            )
        raw_generator_types = _read_json(args.valid_generator_types)
        if not isinstance(raw_generator_types, list) or not all(
            isinstance(value, str) and value.strip() for value in raw_generator_types
        ):
            raise ValueError("valid_generator_types must be a JSON array of strings")
        valid_generator_types = {value.strip() for value in raw_generator_types}
        if not valid_generator_types:
            raise ValueError("synthesis is enabled but no generator types are configured")
    conversions = _conversion_rates(
        args.conversion_old_strict,
        args.conversion_new_strict,
        kpi,
        int(synthetic_policy["minimum_trackable_boxes"]),
    )
    routing_policy = policy["routing"]
    global_conversion = (
        sum(conversions.values()) / len(conversions)
        if conversions
        else float(routing_policy["adaptive_conversion_prior"])
    )

    signature_cache: dict[str, np.ndarray] = {}

    def rank_candidates(
        candidates: list[dict], references: list[tuple[str, tuple[float, ...]]]
    ) -> list[dict]:
        reference_signatures = []
        for file_name, box in references[:12]:
            signature = _crop_signature(str(kpi_images_dir / file_name), [box])
            if signature is not None:
                reference_signatures.append(signature)
        if not reference_signatures:
            return candidates
        matrix = np.vstack(reference_signatures)

        def score(candidate: dict) -> float:
            path = candidate["source_path"]
            if path not in signature_cache:
                signature = _crop_signature(
                    path,
                    [_bbox_xywh_to_xyxy(box) for box in candidate["boxes"]],
                )
                signature_cache[path] = (
                    signature
                    if signature is not None
                    else np.zeros(DCT_KEEP * DCT_KEEP)
                )
            return float(np.max(matrix @ signature_cache[path]))

        return sorted(candidates, key=lambda candidate: (-score(candidate), candidate["source_path"]))

    selected_real: list[dict] = []
    synthetic_counts: Counter[str] = Counter()
    factor_report: dict[str, int] = {}
    frozen_pockets: list[str] = []
    freeze = float(synthetic_policy["conversion_freeze_below"])
    for pocket, fn_count in fn_counts.most_common():
        conversion = conversions.get(pocket)
        if conversion is not None and conversion < freeze:
            factor = int(routing_policy["real_mine_factor_min"])
            frozen_pockets.append(_pocket_name(pocket))
        else:
            basis = conversion if conversion is not None else global_conversion
            factor = round(1.0 / max(basis, 1.0 / int(routing_policy["real_mine_factor_max"])))
            factor = max(
                int(routing_policy["real_mine_factor_min"]),
                min(int(routing_policy["real_mine_factor_max"]), factor),
            )
        factor_report[_pocket_name(pocket)] = factor
        available = [
            record
            for record in positives_by_pocket.get(pocket, [])
            if record["source_path"] not in used_defect
        ]
        available = rank_candidates(available, fn_refs.get(pocket, []))
        quota = factor * fn_count
        chosen = admission.admit(available, quota, clean=False)
        for record in chosen:
            used_defect.add(record["source_path"])
            record["branch"] = "strict_fn_real"
            record["trigger_count"] = int(fn_count)
        selected_real.extend(chosen)
        generator_type = generator_by_pocket[pocket]
        if generator_type not in valid_generator_types:
            continue
        if conversion is not None and conversion < freeze:
            continue
        shortfall = max(0, quota - len(chosen))
        fill_cap = max(
            int(float(synthetic_policy["shortfall_fill_multiplier"]) * fn_count),
            int(synthetic_policy["shortfall_fill_minimum"]),
        )
        requested = math.ceil(
            float(synthetic_policy["ratio_per_admitted_real"]) * len(chosen)
        ) + min(shortfall, fill_cap)
        if requested:
            synthetic_counts[generator_type] += requested

    loose_fps = loose[loose["gap_type"].astype(str).str.upper() == "FP"]
    background_hits: Counter[tuple[str, str]] = Counter()
    near_misses: Counter[tuple[str, str, str]] = Counter()
    unresolved_loose = 0
    for _, row in loose_fps.iterrows():
        info = _lookup_kpi(row, kpi)
        if info is None:
            unresolved_loose += 1
            continue
        best_iou = float(row["best_iou"])
        if best_iou < float(policy["gap"]["background_iou_upper"]):
            background_hits[_texture(info["metadata"])] += 1
        elif best_iou < float(policy["gap"]["near_miss_iou_upper"]):
            near_misses[_pocket(info["metadata"])] += 1

    selected_near: list[dict] = []
    for pocket, count in near_misses.most_common():
        available = [
            record
            for record in positives_by_pocket.get(pocket, [])
            if record["source_path"] not in used_defect
        ]
        quota = min(
            int(routing_policy["near_miss_real_factor"]) * count,
            int(routing_policy["near_miss_real_cap_per_pocket"]),
        )
        chosen = admission.admit(available, quota, clean=False)
        for record in chosen:
            used_defect.add(record["source_path"])
            record["branch"] = "near_miss_real"
            record["trigger_count"] = int(count)
        selected_near.extend(chosen)

    selected_uniform: list[dict] = []
    uniform_quota = uniform_mine_for_iteration(policy, args.iteration)
    if uniform_quota:
        for pocket in sorted(positives_by_pocket):
            available = [
                record
                for record in positives_by_pocket[pocket]
                if record["source_path"] not in used_defect
            ]
            chosen = admission.admit(available, uniform_quota, clean=False)
            for record in chosen:
                used_defect.add(record["source_path"])
                record["branch"] = "uniform_real"
                record["trigger_count"] = 0
            selected_uniform.extend(chosen)

    real_cumulative = len(used_defect)
    if bool(synthetic_policy["enabled"]):
        fraction = float(synthetic_policy["cumulative_fraction_of_defective"])
        allowed = max(
            0,
            int(fraction / (1.0 - fraction) * real_cumulative)
            - int(args.prior_admitted_synthetic),
        )
        allowed = min(allowed, int(synthetic_policy["per_iteration_request_cap"]))
        synthetic_counts = Counter(_scale_counts(dict(synthetic_counts), allowed))
    else:
        synthetic_counts = Counter()

    clean_budget = max(
        0,
        int(float(routing_policy["clean_cumulative_cap_per_real"]) * real_cumulative)
        - len(used_clean),
    )
    selected_clean: list[dict] = []
    for texture, count in background_hits.most_common():
        remaining = clean_budget - len(selected_clean)
        if remaining <= 0:
            break
        available = [
            record
            for record in clean_by_texture.get(texture, [])
            if record["source_path"] not in used_clean
        ]
        quota = min(int(routing_policy["clean_factor"]) * count, remaining)
        chosen = admission.admit(available, quota, clean=True)
        for record in chosen:
            used_clean.add(record["source_path"])
            record["branch"] = "background_clean"
            record["trigger_count"] = int(count)
            record["allow_empty_annotations"] = True
        selected_clean.extend(chosen)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = selected_real + selected_near + selected_uniform + selected_clean
    report = {
        "iteration": int(args.iteration),
        "profile": policy["profile"],
        "loose_fp_rows": int(len(loose_fps)),
        "strict_fn_rows": int(len(strict_fns)),
        "unresolved_loose_rows": int(unresolved_loose),
        "unresolved_strict_rows": int(unresolved_strict),
        "fn_pockets": len(fn_counts),
        "background_hit_count": int(sum(background_hits.values())),
        "near_miss_count": int(sum(near_misses.values())),
        "selected": {
            "strict_fn_real": len(selected_real),
            "near_miss_real": len(selected_near),
            "uniform_real": len(selected_uniform),
            "background_clean": len(selected_clean),
        },
        "uniform_mine_per_pocket": int(uniform_quota),
        "conversion_rates": {
            _pocket_name(pocket): rate for pocket, rate in sorted(conversions.items())
        },
        "adaptive_factors": factor_report,
        "frozen_synthetic_pockets": sorted(frozen_pockets),
        "unsupported_synthetic_pockets": (
            sorted(
                _pocket_name(pocket)
                for pocket in fn_counts
                if generator_by_pocket[pocket] not in valid_generator_types
            )
            if bool(synthetic_policy["enabled"])
            else []
        ),
        "synthetic_requested": int(sum(synthetic_counts.values())),
        "synthetic_plan": dict(sorted(synthetic_counts.items())),
        "cumulative_real_defectives": len(used_defect),
        "cumulative_clean_negatives": len(used_clean),
        "admission": dict(admission.report),
    }
    _atomic_json(output / "mined_manifest.json", manifest)
    _atomic_json(output / "synthetic_plan.json", dict(sorted(synthetic_counts.items())))
    _atomic_json(output / "routing_report.json", report)
    _atomic_json(output / "defect_ledger.json", sorted(used_defect))
    _atomic_json(output / "clean_ledger.json", sorted(used_clean))
    admission.save(output / "admission_index.npy")
    return report


def main() -> int:
    try:
        report = route(parse_args())
        print(
            f"DEFT OD AOI route iter{report['iteration']}: selected={report['selected']} "
            f"synthetic_requested={report['synthetic_requested']} "
            f"uniform={report['uniform_mine_per_pocket']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
