#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route DEFT OD AOI gaps by global role-separated SigLIP similarity only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deft_od_aoi_policy import load_policy, uniform_mine_for_iteration
from route_deft_od_aoi import (
    Admission,
    _atomic_json,
    _conversion_rates,
    _index_coco,
    _kpi_index,
    _load_ledger,
    _pocket_name,
    _read_json,
    _scale_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--candidate-embeddings", required=True)
    parser.add_argument("--query-embeddings", required=True)
    parser.add_argument("--kpi-coco", required=True)
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
    parser.add_argument("--valid-generator-types", default=None)
    return parser.parse_args()


def _vectors(frame: pd.DataFrame, name: str) -> np.ndarray:
    if "embedding" not in frame:
        raise ValueError(f"{name} lacks embedding column")
    try:
        matrix = np.stack(
            [np.asarray(value, dtype=np.float32).reshape(-1) for value in frame["embedding"]]
        )
    except ValueError as exc:
        raise ValueError(f"{name} contains inconsistent embedding dimensions") from exc
    if matrix.ndim != 2 or not matrix.shape[1]:
        raise ValueError(f"{name} embedding matrix is invalid")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(matrix)) or np.any(norms <= 0):
        raise ValueError(f"{name} contains non-finite or zero embeddings")
    return matrix / norms


def _required(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks columns {missing}")


class Retriever:
    def __init__(
        self,
        candidates: pd.DataFrame,
        queries: pd.DataFrame,
        records: dict[str, dict[str, dict[str, Any]]],
        policy: dict[str, Any],
    ) -> None:
        _required(
            candidates,
            {"candidate_id", "role", "parent_filepath", "embedding"},
            "candidate embeddings",
        )
        _required(
            queries,
            {"query_id", "role", "branch", "embedding"},
            "query embeddings",
        )
        if set(candidates["role"].astype(str)) - {"defect", "clean"}:
            raise ValueError("candidate roles must be defect or clean")
        if set(queries["role"].astype(str)) - {"defect", "clean"}:
            raise ValueError("query roles must be defect or clean")
        if candidates["candidate_id"].astype(str).duplicated().any():
            raise ValueError("candidate_id values must be unique")
        if queries["query_id"].astype(str).duplicated().any():
            raise ValueError("query_id values must be unique")
        allowed_branches = {
            "defect": {"strict_fn", "near_miss_fp"},
            "clean": {"background_fp"},
        }
        for role, branches in allowed_branches.items():
            observed = set(
                queries.loc[queries["role"].astype(str) == role, "branch"].astype(str)
            )
            if observed - branches:
                raise ValueError(
                    f"query role {role!r} has incompatible branches {sorted(observed - branches)}"
                )
        self.candidates = candidates.reset_index(drop=True).copy()
        self.queries = queries.reset_index(drop=True).copy()
        self.candidate_vectors = _vectors(self.candidates, "candidate embeddings")
        self.query_vectors = _vectors(self.queries, "query embeddings")
        if self.candidate_vectors.shape[1] != self.query_vectors.shape[1]:
            raise ValueError("candidate and query embedding dimensions differ")
        self.records = records
        self.minimum = float(policy["minimum_similarity"])
        self.overfetch = int(policy["candidate_overfetch"])
        self.audit_top_k = int(policy["audit_top_k_per_query"])
        self.audit_rows: list[dict[str, Any]] = []
        self.role_indices = {
            role: np.flatnonzero(self.candidates["role"].astype(str).to_numpy() == role)
            for role in ("defect", "clean")
        }
        for role, indices in self.role_indices.items():
            known = set(records[role])
            parents = set(self.candidates.iloc[indices]["parent_filepath"].astype(str))
            missing = sorted(parents - known)
            if missing:
                raise ValueError(
                    f"{role} embedding parents are absent from canonical COCO: {missing[:5]}"
                )

    def query_rows(self, branch: str) -> pd.DataFrame:
        return self.queries[self.queries["branch"].astype(str) == branch].copy()

    def rank(
        self,
        *,
        role: str,
        query_rows: pd.DataFrame,
        excluded: set[str],
        quota: int,
        branch: str,
    ) -> list[dict[str, Any]]:
        if quota <= 0 or query_rows.empty:
            return []
        query_positions = query_rows.index.to_numpy(dtype=int)
        candidate_positions = self.role_indices[role]
        if excluded:
            keep = ~self.candidates.iloc[candidate_positions]["parent_filepath"].astype(str).isin(excluded).to_numpy()
            candidate_positions = candidate_positions[keep]
        if not len(candidate_positions):
            return []
        scores = self.query_vectors[query_positions] @ self.candidate_vectors[candidate_positions].T
        per_query_limit = min(
            scores.shape[1],
            max(self.audit_top_k, math.ceil(quota / len(query_positions)) * self.overfetch + 10),
        )
        orders = []
        for local_query, query_position in enumerate(query_positions):
            row_scores = scores[local_query]
            if per_query_limit < len(row_scores):
                positions = np.argpartition(-row_scores, per_query_limit - 1)[:per_query_limit]
                positions = positions[np.argsort(-row_scores[positions], kind="stable")]
            else:
                positions = np.argsort(-row_scores, kind="stable")
            ranked: list[tuple[int, float]] = []
            seen_parent: set[str] = set()
            query = self.queries.iloc[query_position]
            for local_candidate in positions:
                score = float(row_scores[local_candidate])
                if score < self.minimum:
                    continue
                candidate_position = int(candidate_positions[local_candidate])
                candidate = self.candidates.iloc[candidate_position]
                parent = str(candidate["parent_filepath"])
                if parent in seen_parent:
                    continue
                seen_parent.add(parent)
                ranked.append((candidate_position, score))
                if len(ranked) <= self.audit_top_k:
                    self.audit_rows.append(
                        {
                            "branch": branch,
                            "role": role,
                            "query_id": str(query["query_id"]),
                            "query_benchmark": str(query.get("benchmark", "")),
                            "candidate_id": str(candidate["candidate_id"]),
                            "candidate_parent": parent,
                            "candidate_benchmark": str(candidate.get("benchmark", "")),
                            "similarity": score,
                            "query_rank": len(ranked),
                        }
                    )
            orders.append(ranked)

        target = max(quota * self.overfetch, quota + 25)
        output: list[dict[str, Any]] = []
        selected_parents: set[str] = set()
        depth = 0
        while len(output) < target and any(depth < len(order) for order in orders):
            for local_query, order in enumerate(orders):
                if depth >= len(order):
                    continue
                candidate_position, score = order[depth]
                candidate = self.candidates.iloc[candidate_position]
                parent = str(candidate["parent_filepath"])
                if parent in selected_parents:
                    continue
                selected_parents.add(parent)
                record = dict(self.records[role][parent])
                query = query_rows.iloc[local_query]
                record["retrieval"] = {
                    "backend": "siglip",
                    "similarity": score,
                    "query_id": str(query["query_id"]),
                    "candidate_id": str(candidate["candidate_id"]),
                }
                output.append(record)
                if len(output) >= target:
                    break
            depth += 1
        return output


def _record_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["source_path"]): record for record in index["records"]}


def _pocket_from_query(row: pd.Series) -> tuple[str, str, str]:
    values = tuple(str(row.get(key, "")).strip() for key in ("benchmark", "texture", "defect_type"))
    if any(not value for value in values):
        raise ValueError(f"defect query {row.get('query_id')!r} lacks pocket metadata")
    return values  # type: ignore[return-value]


def _score_summary(matches: pd.DataFrame, selected: set[str]) -> dict[str, Any]:
    if matches.empty:
        return {}
    output = {}
    for branch, frame in matches.groupby("branch"):
        values = frame["similarity"].astype(float)
        output[str(branch)] = {
            "audit_rows": int(len(frame)),
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
            "selected_parents_in_audit": int(frame["candidate_parent"].isin(selected).sum()),
        }
    return output


def route(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    retrieval_policy = policy.get("retrieval") or {}
    if retrieval_policy.get("mode") != "siglip_only":
        raise ValueError("policy retrieval.mode must be siglip_only")
    if uniform_mine_for_iteration(policy, args.iteration) != 0:
        raise ValueError("SigLIP-only routing requires uniform mining to be zero")
    if not 1 <= args.iteration <= int(policy["max_iterations"]):
        raise ValueError("iteration is outside the frozen policy range")
    if int(args.prior_admitted_synthetic) < 0:
        raise ValueError("prior_admitted_synthetic cannot be negative")
    candidates = pd.read_parquet(Path(args.candidate_embeddings).expanduser().resolve())
    queries = pd.read_parquet(Path(args.query_embeddings).expanduser().resolve())
    source = _index_coco(
        _read_json(args.source_coco),
        Path(args.source_images_dir).expanduser().resolve(),
        clean=False,
    )
    clean = _index_coco(
        _read_json(args.clean_coco),
        Path(args.clean_images_dir).expanduser().resolve(),
        clean=True,
    )
    records = {"defect": _record_map(source), "clean": _record_map(clean)}
    retriever = Retriever(candidates, queries, records, retrieval_policy)
    used_defect = _load_ledger(args.previous_defect_ledger)
    used_clean = _load_ledger(args.previous_clean_ledger)
    admission = Admission(policy, args.previous_admission_index)

    kpi = _kpi_index(_read_json(args.kpi_coco))
    conversions = _conversion_rates(
        args.conversion_old_strict,
        args.conversion_new_strict,
        kpi,
        int(policy["synthetic"]["minimum_trackable_boxes"]),
    )
    routing_policy = policy["routing"]
    global_conversion = (
        sum(conversions.values()) / len(conversions)
        if conversions
        else float(routing_policy["adaptive_conversion_prior"])
    )
    strict = retriever.query_rows("strict_fn")
    strict_by_pocket: dict[tuple[str, str, str], pd.DataFrame] = {
        pocket: frame
        for pocket, frame in strict.groupby(
            strict.apply(_pocket_from_query, axis=1), sort=False
        )
    }
    synthetic_policy = policy["synthetic"]
    valid_generator_types: set[str] = set()
    if bool(synthetic_policy["enabled"]):
        if not args.valid_generator_types:
            raise ValueError("valid_generator_types is required when synthesis is enabled")
        values = _read_json(args.valid_generator_types)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("valid_generator_types must be a JSON string array")
        valid_generator_types = {value.strip() for value in values if value.strip()}

    selected_real: list[dict[str, Any]] = []
    synthetic_counts: Counter[str] = Counter()
    factor_report: dict[str, int] = {}
    frozen_pockets: list[str] = []
    freeze = float(synthetic_policy["conversion_freeze_below"])
    for pocket, query_rows in strict_by_pocket.items():
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
        quota = factor * len(query_rows)
        ranked = retriever.rank(
            role="defect",
            query_rows=query_rows,
            excluded=used_defect,
            quota=quota,
            branch="strict_fn_real",
        )
        chosen = admission.admit(ranked, quota, clean=False)
        for record in chosen:
            used_defect.add(record["source_path"])
            record["branch"] = "strict_fn_real"
            record["trigger_count"] = int(len(query_rows))
        selected_real.extend(chosen)
        generator = str(query_rows.iloc[0].get("generator_type", "")).strip() or _pocket_name(pocket)
        if generator not in valid_generator_types or (conversion is not None and conversion < freeze):
            continue
        shortfall = max(0, quota - len(chosen))
        fill_cap = max(
            int(float(synthetic_policy["shortfall_fill_multiplier"]) * len(query_rows)),
            int(synthetic_policy["shortfall_fill_minimum"]),
        )
        requested = math.ceil(
            float(synthetic_policy["ratio_per_admitted_real"]) * len(chosen)
        ) + min(shortfall, fill_cap)
        if requested:
            synthetic_counts[generator] += requested

    selected_near: list[dict[str, Any]] = []
    near = retriever.query_rows("near_miss_fp")
    if not near.empty:
        near_groups = near.groupby(near.apply(_pocket_from_query, axis=1), sort=False)
        for _, query_rows in near_groups:
            quota = min(
                int(routing_policy["near_miss_real_factor"]) * len(query_rows),
                int(routing_policy["near_miss_real_cap_per_pocket"]),
            )
            ranked = retriever.rank(
                role="defect",
                query_rows=query_rows,
                excluded=used_defect,
                quota=quota,
                branch="near_miss_real",
            )
            chosen = admission.admit(ranked, quota, clean=False)
            for record in chosen:
                used_defect.add(record["source_path"])
                record["branch"] = "near_miss_real"
                record["trigger_count"] = int(len(query_rows))
            selected_near.extend(chosen)

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
    selected_clean: list[dict[str, Any]] = []
    background = retriever.query_rows("background_fp")
    if not background.empty and clean_budget:
        quota = min(int(routing_policy["clean_factor"]) * len(background), clean_budget)
        ranked = retriever.rank(
            role="clean",
            query_rows=background,
            excluded=used_clean,
            quota=quota,
            branch="background_clean",
        )
        chosen = admission.admit(ranked, quota, clean=True)
        for record in chosen:
            used_clean.add(record["source_path"])
            record["branch"] = "background_clean"
            record["trigger_count"] = int(len(background))
            record["allow_empty_annotations"] = True
        selected_clean.extend(chosen)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = selected_real + selected_near + selected_clean
    audit = pd.DataFrame(retriever.audit_rows)
    selected_paths = {record["source_path"] for record in manifest}
    if not audit.empty:
        audit["selected_parent"] = audit["candidate_parent"].isin(selected_paths)
        audit.to_parquet(output / "retrieval_audit.parquet", index=False)
    selected_benchmarks = Counter(record["benchmark"] for record in manifest)
    report = {
        "iteration": int(args.iteration),
        "retrieval": {
            "mode": "siglip_only",
            "model": retrieval_policy["model"],
            "model_path": retrieval_policy["model_path"],
            "embedding_dimension": int(retriever.candidate_vectors.shape[1]),
            "minimum_similarity": float(retrieval_policy["minimum_similarity"]),
            "score_summary": _score_summary(audit, selected_paths),
        },
        "queries": {
            "strict_fn": int(len(strict)),
            "near_miss_fp": int(len(near)),
            "background_fp": int(len(background)),
        },
        "selected": {
            "strict_fn_real": len(selected_real),
            "near_miss_real": len(selected_near),
            "uniform_real": 0,
            "background_clean": len(selected_clean),
        },
        "selected_by_benchmark": dict(sorted(selected_benchmarks.items())),
        "uniform_mine_per_pocket": 0,
        "conversion_rates": {
            _pocket_name(pocket): rate for pocket, rate in sorted(conversions.items())
        },
        "adaptive_factors": factor_report,
        "frozen_synthetic_pockets": sorted(frozen_pockets),
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
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
