#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DEFT OD AOI policy construction and validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def build_policy(
    *,
    max_iterations: int,
    synthetic_enabled: bool | None = None,
    probes_enabled: bool | None = None,
) -> dict[str, Any]:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if synthetic_enabled is None:
        raise ValueError("synthetic_enabled must be explicit")
    synthetic_is_enabled = bool(synthetic_enabled)

    probes_are_enabled = True if probes_enabled is None else bool(probes_enabled)

    policy: dict[str, Any] = {
        "schema_version": 3,
        "max_iterations": int(max_iterations),
        "task": {"class_name": "defect", "binary": True},
        "data": {
            "seed_dataset": False,
            "source_pool_buckets": ["train", "mine"],
            "cumulative_training": True,
        },
        "model": {
            "architecture": "rtdetr",
            "backbone": "resnet_50",
            "num_feature_levels": 3,
            "return_interm_indices": [1, 2, 3],
        },
        "inference": {
            "confidence_threshold": 0.001,
            "num_gpus": 1,
            "batch_size": 32,
            "workers": 8,
        },
        "gap": {
            "loose_confidence": 0.3,
            "strict_confidence": 0.8,
            "match_iou": 0.5,
            "background_iou_upper": 0.05,
            "near_miss_iou_upper": 0.5,
            "weak_recall_threshold": 1.0,
            "weak_ap50_threshold": 0.5,
            "weak_precision_threshold": 0.0,
        },
        "routing": {
            "adaptive_conversion_prior": 0.33,
            "real_mine_factor_min": 1,
            "real_mine_factor_max": 6,
            "near_miss_real_factor": 2,
            "near_miss_real_cap_per_pocket": 20,
            "clean_factor": 2,
            "clean_cumulative_cap_per_real": 1.0,
            # Retained as a fixed compatibility field for the legacy router.
            # The SigLIP-only workflow never performs gap-independent mining.
            "uniform_mine": {"mode": "constant", "per_pocket": 0},
        },
        "retrieval": {
            "mode": "siglip_only",
            "model": "SigLIP",
            "model_path": "google/siglip-base-patch16-224",
            "defect_context_scale": 1.5,
            "clean_grids": [1, 2],
            "output_size": 224,
            # Start without an uncalibrated cutoff. The retrieval audit records
            # score distributions so a later run can freeze a justified value.
            "minimum_similarity": -1.0,
            "candidate_overfetch": 15,
            "audit_top_k_per_query": 20,
        },
        "synthetic": {
            "enabled": synthetic_is_enabled,
            "ratio_per_admitted_real": 0.5,
            "shortfall_fill_multiplier": 2.0,
            "shortfall_fill_minimum": 20,
            "conversion_freeze_below": 0.05,
            "minimum_trackable_boxes": 4,
            "cumulative_fraction_of_defective": 0.25,
            "per_iteration_request_cap": 1500,
        },
        "admission": {
            "duplicate_global_cosine": 0.995,
            "duplicate_defect_cosine": 0.985,
            "duplicate_position_delta": 0.04,
            "clean_duplicate_global_cosine": 0.999,
            "clean_cluster_cosine": 0.97,
            "clean_cluster_minimum_cap": 2,
            "clean_cluster_quota_divisor": 4,
            "minimum_box_area_px": 64,
            "maximum_box_aspect": 25.0,
            "duplicate_gt_iou": 0.9,
        },
        "training": {
            "fresh_base_each_iteration": True,
            "num_gpus": 4,
            "batch_size": 8,
            "base_learning_rate": 1.0e-4,
            "backbone_learning_rate": 1.0e-5,
            "validation_interval": 1,
            "checkpoint_interval": 1,
            "workers": 3,
            "fixed_epoch_iterations": [1, 2],
            "fixed_epochs": 36,
            "probes_enabled": probes_are_enabled,
            "probe_start_iteration": 3,
            "probe_count": 3,
            "probe_epochs": 10,
            "probe_policy": {
                "growth_high": 1.25,
                "growth_low": 0.9,
                "high_growth_lr_factor": 1.15,
                "high_growth_step_increment": 2,
                "high_growth_step_maximum": 33,
                "low_growth_lr_factor": 0.85,
                "jitter_seed_base": 4000,
                "jitter_lr_factor": [0.7, 1.4],
                "jitter_weight_decay_factor": [0.5, 2.0],
                "jitter_lr_decay_factor": [0.6, 1.6],
                "jitter_lr_decay_bounds": [0.05, 0.5],
            },
            "epoch_budget": {
                "formula": "round(36 * sqrt(10000 / training_images))",
                "minimum": 24,
                "maximum": 48,
            },
            "extension_epochs": 12,
            "extension_trigger_last_n_epochs": 3,
            "checkpoint_selection_metric": "kpi_validation_ap50",
            "test_is_report_only": True,
        },
    }
    validate_policy(policy)
    return policy


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != 3:
        raise ValueError("unsupported DEFT OD AOI policy schema_version")
    if int(policy.get("max_iterations", 0)) < 1:
        raise ValueError("max_iterations must be positive")
    training = policy.get("training") or {}
    if not isinstance(training.get("probes_enabled"), bool):
        raise ValueError("training.probes_enabled must be boolean")

    gap = policy.get("gap") or {}
    loose = float(gap.get("loose_confidence", -1))
    strict = float(gap.get("strict_confidence", -1))
    background = float(gap.get("background_iou_upper", -1))
    near = float(gap.get("near_miss_iou_upper", -1))
    match = float(gap.get("match_iou", -1))
    if not 0 <= loose < strict <= 1:
        raise ValueError("require 0 <= loose_confidence < strict_confidence <= 1")
    if not 0 <= background < near == match <= 1:
        raise ValueError(
            "require 0 <= background_iou_upper < near_miss_iou_upper == match_iou <= 1"
        )
    inference = policy.get("inference") or {}
    inference_confidence = float(inference.get("confidence_threshold", -1))
    if not 0 <= inference_confidence < loose:
        raise ValueError("inference confidence must be below the loose gap confidence")

    routing = policy.get("routing") or {}
    factor_min = int(routing.get("real_mine_factor_min", 0))
    factor_max = int(routing.get("real_mine_factor_max", 0))
    if not 1 <= factor_min <= factor_max:
        raise ValueError("invalid adaptive real-mining factor bounds")
    if float(routing.get("clean_cumulative_cap_per_real", -1)) < 0:
        raise ValueError("clean cumulative cap cannot be negative")

    synthetic = policy.get("synthetic") or {}
    if not isinstance(synthetic.get("enabled"), bool):
        raise ValueError("synthetic.enabled must be boolean")
    fraction = float(synthetic.get("cumulative_fraction_of_defective", -1))
    if not 0 <= fraction < 1:
        raise ValueError("synthetic cumulative fraction must be within [0, 1)")
    if int(synthetic.get("minimum_trackable_boxes", 0)) < 1:
        raise ValueError("minimum_trackable_boxes must be positive")
    if int(synthetic.get("per_iteration_request_cap", -1)) < 0:
        raise ValueError("synthetic per-iteration cap cannot be negative")
    if int(synthetic.get("shortfall_fill_minimum", -1)) < 0:
        raise ValueError("synthetic shortfall-fill minimum cannot be negative")

    uniform = routing.get("uniform_mine") or {}
    mode = uniform.get("mode")
    if mode != "constant":
        raise ValueError("uniform_mine.mode must be constant")
    if int(uniform.get("per_pocket", -1)) < 0:
        raise ValueError("uniform per-pocket value cannot be negative")
    if int(uniform["per_pocket"]) != 0:
        raise ValueError("SigLIP-only routing requires uniform mining to be disabled")

    retrieval = policy.get("retrieval") or {}
    if retrieval.get("mode") != "siglip_only":
        raise ValueError("retrieval.mode must be siglip_only")
    if retrieval.get("model") != "SigLIP":
        raise ValueError("retrieval.model must be SigLIP")
    if not str(retrieval.get("model_path", "")).strip():
        raise ValueError("retrieval.model_path is required")
    if float(retrieval.get("defect_context_scale", 0)) < 1:
        raise ValueError("retrieval.defect_context_scale must be at least 1")
    clean_grids = retrieval.get("clean_grids")
    if not isinstance(clean_grids, list) or not clean_grids:
        raise ValueError("retrieval.clean_grids must be a non-empty list")
    if any(int(value) < 1 for value in clean_grids):
        raise ValueError("retrieval.clean_grids values must be positive")
    if int(retrieval.get("output_size", 0)) < 1:
        raise ValueError("retrieval.output_size must be positive")
    similarity = float(retrieval.get("minimum_similarity", -2))
    if not -1 <= similarity <= 1:
        raise ValueError("retrieval.minimum_similarity must be within [-1, 1]")
    if int(retrieval.get("candidate_overfetch", 0)) < 1:
        raise ValueError("retrieval.candidate_overfetch must be positive")
    if int(retrieval.get("audit_top_k_per_query", 0)) < 1:
        raise ValueError("retrieval.audit_top_k_per_query must be positive")


def probes_active(policy: dict[str, Any], iteration: int) -> bool:
    if iteration < 1:
        raise ValueError("iteration must be positive")
    validate_policy(policy)
    training = policy["training"]
    return bool(training["probes_enabled"]) and iteration >= int(
        training["probe_start_iteration"]
    )


def main_epoch_budget(policy: dict[str, Any], iteration: int, train_size: int) -> int:
    if iteration < 1:
        raise ValueError("iteration must be positive")
    if train_size < 1:
        raise ValueError("train_size must be positive")
    validate_policy(policy)
    training = policy["training"]
    fixed = {int(value) for value in training["fixed_epoch_iterations"]}
    if iteration in fixed:
        return int(training["fixed_epochs"])
    epoch = training["epoch_budget"]
    budget = round(36 * math.sqrt(10000 / train_size))
    return max(int(epoch["minimum"]), min(int(epoch["maximum"]), budget))


def uniform_mine_for_iteration(policy: dict[str, Any], iteration: int) -> int:
    if iteration < 1:
        raise ValueError("iteration must be positive")
    validate_policy(policy)
    uniform = policy["routing"]["uniform_mine"]
    return int(uniform["per_pocket"])


def load_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("DEFT OD AOI policy must be a JSON object")
    validate_policy(policy)
    return policy
