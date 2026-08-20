#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare exact DEFT OD AOI pools from a dataset source manifest."""

from __future__ import annotations

import argparse
import filecmp
import glob
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from normalize_deft_od_aoi_pools import (
    CATEGORY,
    _annotation,
    _image,
    _read_coco,
    _slug,
    _source_path,
    _unique_name,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    return parser.parse_args()


def _resolve(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def _expand_paths(value: Any, base: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    output: list[Path] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        pattern = str(candidate if candidate.is_absolute() else base / candidate)
        if glob.has_magic(pattern):
            output.extend(
                Path(path).resolve()
                for path in sorted(glob.glob(pattern, recursive=True))
            )
        else:
            output.append(Path(pattern).resolve())
    return sorted(set(output))


def _labels(annotations: list[dict[str, Any]]) -> list[str]:
    labels = set()
    for annotation in annotations:
        for key in ("defect_label", "defect_type", "category_name", "label"):
            value = str(annotation.get(key, "")).strip()
            if value:
                labels.add(value)
                break
    return sorted(labels)


def _existing(image: dict[str, Any], key: str) -> str:
    nested = image.get("deft_od_aoi")
    nested = nested if isinstance(nested, dict) else {}
    value = nested.get(key, image.get(key, ""))
    return str(value).strip() if value is not None else ""


def _derive_metadata(
    *,
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    source_path: Path,
    entry: dict[str, Any],
    info: dict[str, Any],
    rules: dict[str, Any],
    strict: bool,
    sources: Counter[str],
) -> dict[str, str]:
    benchmark = _existing(image, "benchmark")
    if benchmark:
        sources["benchmark:input"] += 1
    else:
        benchmark = str(
            entry.get("benchmark")
            or info.get("bench")
            or info.get("source_dataset")
            or ""
        ).strip()
        if benchmark:
            sources["benchmark:source"] += 1
    if not benchmark:
        if strict:
            raise ValueError(f"cannot resolve benchmark for {source_path}")
        benchmark = _slug(source_path.parent.name)
        sources["benchmark:fallback"] += 1

    rule = rules.get(benchmark)
    if not isinstance(rule, dict):
        if strict:
            raise ValueError(f"no metadata rule for benchmark {benchmark!r}")
        rule = {}
        sources["rule:missing_fallback"] += 1

    has_defect = bool(annotations)
    regex_text = str(
        entry.get("defect_path_regex" if has_defect else "clean_path_regex")
        or rule.get("defect_path_regex" if has_defect else "clean_path_regex")
        or (rule.get("defect_path_regex") if not has_defect else "")
        or ""
    )
    match = re.search(regex_text, str(source_path)) if regex_text else None
    if regex_text and match is None and strict:
        raise ValueError(
            f"path does not match {benchmark} {'defect' if has_defect else 'clean'} rule: "
            f"{source_path}"
        )
    groups = match.groupdict() if match else {}

    texture = _existing(image, "texture")
    if texture:
        sources["texture:input"] += 1
    else:
        texture = str(groups.get("texture") or "").strip()
        if texture:
            sources["texture:path_regex"] += 1
        elif strict:
            raise ValueError(f"cannot resolve texture for {source_path}")
        else:
            texture = benchmark
            sources["texture:benchmark_fallback"] += 1

    metadata = {"benchmark": benchmark, "texture": texture}
    if has_defect:
        defect_type = _existing(image, "defect_type")
        if defect_type:
            sources["defect_type:input"] += 1
        else:
            defect_type = str(groups.get("defect_type") or "").strip()
            if defect_type:
                sources["defect_type:path_regex"] += 1
            else:
                defect_type = str(rule.get("default_defect_type") or "").strip()
                if defect_type:
                    sources["defect_type:rule_default"] += 1
                else:
                    values = _labels(annotations)
                    if values:
                        defect_type = "+".join(values)
                        sources["defect_type:annotation_label"] += 1
        if not defect_type:
            if strict:
                raise ValueError(f"cannot resolve defect_type for {source_path}")
            defect_type = "defect"
            sources["defect_type:fallback"] += 1
        renames = rule.get("defect_type_renames") or {}
        defect_type = str(renames.get(defect_type, defect_type))
        metadata["defect_type"] = defect_type

        generator_type = _existing(image, "generator_type")
        if generator_type:
            sources["generator_type:input"] += 1
        else:
            template = str(
                entry.get("generator_type_template")
                or rule.get("generator_type_template")
                or ""
            ).strip()
            if template:
                generator_type = template.format(
                    benchmark=benchmark, texture=texture, defect_type=defect_type
                )
                sources["generator_type:template"] += 1
        if generator_type:
            metadata["generator_type"] = generator_type
    return metadata


def _entries(value: Any, role: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"manifest inputs.{role} must be a non-empty array")
    if not all(isinstance(entry, dict) for entry in value):
        raise ValueError(f"manifest inputs.{role} entries must be objects")
    return value


def _entry_cocos(entry: dict[str, Any], base: Path) -> list[Path]:
    paths = _expand_paths(entry.get("coco"), base)
    if not paths:
        raise ValueError(f"source entry has no COCO files: {entry}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"COCO files are missing: {missing}")
    return paths


def _images_dir(entry: dict[str, Any], coco_path: Path, base: Path) -> Path:
    raw = str(entry.get("images_dir") or "").strip()
    return _resolve(raw, base) if raw else coco_path.parent


def _normalize_eval(
    role: str,
    entries: list[dict[str, Any]],
    base: Path,
    rules: dict[str, Any],
    strict: bool,
    sources: Counter[str],
) -> tuple[dict[str, Any], set[str]]:
    output = {"images": [], "annotations": [], "categories": CATEGORY}
    identities: set[str] = set()
    names: set[str] = set()
    for entry in entries:
        for coco_path in _entry_cocos(entry, base):
            coco = _read_coco(coco_path)
            info = coco.get("info") if isinstance(coco.get("info"), dict) else {}
            by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
            for annotation in coco["annotations"]:
                by_image[annotation.get("image_id")].append(annotation)
            images_dir = _images_dir(entry, coco_path, base)
            for source_image in coco["images"]:
                image_annotations = by_image.get(source_image.get("id"), [])
                path = _source_path(source_image, images_dir, coco_path)
                identity = str(path)
                if identity in identities:
                    raise ValueError(f"duplicate {role} image: {identity}")
                identities.add(identity)
                metadata = _derive_metadata(
                    image=source_image,
                    annotations=image_annotations,
                    source_path=path,
                    entry=entry,
                    info=info,
                    rules=rules,
                    strict=strict,
                    sources=sources,
                )
                file_name = _unique_name(
                    str(source_image.get("file_name") or path.name), names
                )
                image_id = len(output["images"]) + 1
                output["images"].append(
                    _image(
                        source_image,
                        image_id=image_id,
                        file_name=file_name,
                        source_path=path,
                        metadata=metadata,
                        source=coco_path,
                    )
                )
                for annotation in image_annotations:
                    output["annotations"].append(
                        _annotation(
                            annotation,
                            image_id,
                            len(output["annotations"]) + 1,
                            coco_path,
                        )
                    )
    return output, identities


def _add_pool_image(
    *,
    kind: str,
    source_image: dict[str, Any],
    annotations: list[dict[str, Any]],
    path: Path,
    metadata: dict[str, str],
    source: Path,
    outputs: dict[str, dict[str, Any]],
    names: dict[str, set[str]],
) -> None:
    output = outputs[kind]
    image_id = len(output["images"]) + 1
    preferred = f"{_slug(metadata['benchmark'])}__{Path(str(source_image.get('file_name') or path.name)).name}"
    file_name = _unique_name(preferred, names[kind])
    output["images"].append(
        _image(
            source_image,
            image_id=image_id,
            file_name=file_name,
            source_path=path,
            metadata=metadata,
            source=source,
        )
    )
    for annotation in annotations:
        output["annotations"].append(
            _annotation(annotation, image_id, len(output["annotations"]) + 1, source)
        )


def _normalize_mining(
    entries: list[dict[str, Any]],
    base: Path,
    rules: dict[str, Any],
    strict: bool,
    sources: Counter[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    outputs = {
        "source": {"images": [], "annotations": [], "categories": CATEGORY},
        "clean": {"images": [], "annotations": [], "categories": CATEGORY},
    }
    names = {"source": set(), "clean": set()}
    identities: dict[str, str] = {}
    reports: list[dict[str, Any]] = []
    for entry in entries:
        clean_cocos = set(_expand_paths(entry.get("boxless_clean_coco") or [], base))
        all_boxless = bool(entry.get("boxless_as_clean", False))
        coco_paths = _entry_cocos(entry, base)
        unknown_clean_cocos = sorted(clean_cocos - set(coco_paths))
        if unknown_clean_cocos:
            raise ValueError(
                "boxless_clean_coco must also be listed by the mining entry's "
                f"coco field: {[str(path) for path in unknown_clean_cocos]}"
            )
        for coco_path in coco_paths:
            coco = _read_coco(coco_path)
            info = coco.get("info") if isinstance(coco.get("info"), dict) else {}
            by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
            for annotation in coco["annotations"]:
                by_image[annotation.get("image_id")].append(annotation)
            images_dir = _images_dir(entry, coco_path, base)
            counts = Counter()
            for source_image in coco["images"]:
                image_annotations = by_image.get(source_image.get("id"), [])
                if not image_annotations and not (all_boxless or coco_path in clean_cocos):
                    counts["boxless_ignored"] += 1
                    continue
                kind = "source" if image_annotations else "clean"
                path = _source_path(source_image, images_dir, coco_path)
                identity = str(path)
                previous = identities.get(identity)
                if previous:
                    if previous != kind:
                        raise ValueError(
                            f"image is both defective and clean across sources: {identity}"
                        )
                    counts["duplicates_skipped"] += 1
                    continue
                metadata = _derive_metadata(
                    image=source_image,
                    annotations=image_annotations,
                    source_path=path,
                    entry=entry,
                    info=info,
                    rules=rules,
                    strict=strict,
                    sources=sources,
                )
                identities[identity] = kind
                _add_pool_image(
                    kind=kind,
                    source_image=source_image,
                    annotations=image_annotations,
                    path=path,
                    metadata=metadata,
                    source=coco_path,
                    outputs=outputs,
                    names=names,
                )
                counts[kind] += 1
                counts["boxes"] += len(image_annotations)
            reports.append(
                {
                    "path": str(coco_path),
                    "benchmark": str(entry.get("benchmark") or info.get("bench") or ""),
                    "boxless_policy": (
                        "all"
                        if all_boxless
                        else ("selected" if coco_path in clean_cocos else "ignore")
                    ),
                    **dict(sorted(counts.items())),
                }
            )
    return outputs, identities, reports


def _clean_paths(entry: dict[str, Any], base: Path) -> list[Path]:
    values: list[Path] = []
    if entry.get("paths"):
        values.extend(_expand_paths(entry["paths"], base))
    if entry.get("glob"):
        values.extend(_expand_paths(entry["glob"], base))
    paths = sorted(set(path for path in values if path.is_file()))
    if not paths:
        raise ValueError(f"clean source resolves to no image files: {entry}")
    return paths


def _add_external_clean(
    entries: list[dict[str, Any]],
    base: Path,
    rules: dict[str, Any],
    strict: bool,
    sources: Counter[str],
    outputs: dict[str, dict[str, Any]],
    identities: dict[str, str],
) -> list[dict[str, Any]]:
    reports = []
    names = {Path(image["file_name"]).name for image in outputs["clean"]["images"]}
    for entry in entries:
        has_coco = bool(entry.get("coco"))
        has_files = bool(entry.get("glob") or entry.get("paths"))
        if has_coco == has_files:
            raise ValueError(
                "each clean entry must provide exactly one of coco or glob/paths"
            )
        if entry.get("coco"):
            for coco_path in _entry_cocos(entry, base):
                coco = _read_coco(coco_path)
                if coco["annotations"]:
                    raise ValueError(
                        f"explicit clean COCO must have zero annotations: {coco_path}"
                    )
                info = coco.get("info") if isinstance(coco.get("info"), dict) else {}
                images_dir = _images_dir(entry, coco_path, base)
                counts = Counter()
                for source_image in coco["images"]:
                    path = _source_path(source_image, images_dir, coco_path)
                    metadata = _derive_metadata(
                        image=source_image,
                        annotations=[],
                        source_path=path,
                        entry=entry,
                        info=info,
                        rules=rules,
                        strict=strict,
                        sources=sources,
                    )
                    identity = str(path)
                    previous = identities.get(identity)
                    if previous:
                        if previous != "clean":
                            raise ValueError(
                                f"explicit clean image is also defective: {identity}"
                            )
                        counts["duplicates_skipped"] += 1
                        continue
                    identities[identity] = "clean"
                    _add_pool_image(
                        kind="clean",
                        source_image=source_image,
                        annotations=[],
                        path=path,
                        metadata=metadata,
                        source=coco_path,
                        outputs=outputs,
                        names={"source": set(), "clean": names},
                    )
                    counts["clean"] += 1
                reports.append(
                    {
                        "benchmark": str(
                            entry.get("benchmark") or info.get("bench") or ""
                        ),
                        "coco": str(coco_path),
                        **dict(sorted(counts.items())),
                    }
                )
            continue
        counts = Counter()
        for path in _clean_paths(entry, base):
            source_image = {
                "file_name": path.name,
                "source_path": str(path),
                "width": 0,
                "height": 0,
            }
            with Image.open(path) as image:
                source_image["width"], source_image["height"] = image.size
            metadata = _derive_metadata(
                image=source_image,
                annotations=[],
                source_path=path,
                entry=entry,
                info={},
                rules=rules,
                strict=strict,
                sources=sources,
            )
            exclude = str(entry.get("exclude_if_exists") or "").strip()
            if exclude:
                rendered = exclude.format(
                    benchmark=metadata["benchmark"],
                    texture=metadata["texture"],
                    stem=path.stem,
                    name=path.name,
                )
                if _resolve(rendered, base).exists():
                    counts["excluded"] += 1
                    continue
            identity = str(path)
            previous = identities.get(identity)
            if previous:
                if previous != "clean":
                    raise ValueError(f"external clean image is also defective: {identity}")
                counts["duplicates_skipped"] += 1
                continue
            identities[identity] = "clean"
            _add_pool_image(
                kind="clean",
                source_image=source_image,
                annotations=[],
                path=path,
                metadata=metadata,
                source=path,
                outputs=outputs,
                names={"source": set(), "clean": names},
            )
            counts["clean"] += 1
        reports.append(
            {
                "benchmark": str(entry.get("benchmark") or ""),
                "glob": str(entry.get("glob") or ""),
                "paths": [str(path) for path in entry.get("paths", [])]
                if isinstance(entry.get("paths"), list)
                else str(entry.get("paths") or ""),
                "exclude_if_exists": str(entry.get("exclude_if_exists") or ""),
                **dict(sorted(counts.items())),
            }
        )
    return reports


def prepare(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if args.check_only and args.output_dir:
        raise ValueError("use either --check-only or --output-dir, not both")
    if not args.check_only and not args.output_dir:
        raise ValueError("provide --check-only or --output-dir")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset source manifest schema_version")
    base = manifest_path.parent
    strict = bool(manifest.get("strict_metadata", True))
    rules = manifest.get("metadata_rules")
    if not isinstance(rules, dict):
        raise ValueError("metadata_rules must be an object")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object")
    sources: Counter[str] = Counter()
    kpi, kpi_ids = _normalize_eval(
        "kpi", _entries(inputs.get("kpi"), "kpi"), base, rules, strict, sources
    )
    test, test_ids = _normalize_eval(
        "test", _entries(inputs.get("test"), "test"), base, rules, strict, sources
    )
    pools, pool_identity_kinds, mining_reports = _normalize_mining(
        _entries(inputs.get("mining"), "mining"), base, rules, strict, sources
    )
    clean_entries = inputs.get("clean") or []
    if not isinstance(clean_entries, list):
        raise ValueError("inputs.clean must be an array")
    clean_reports = _add_external_clean(
        clean_entries,
        base,
        rules,
        strict,
        sources,
        pools,
        pool_identity_kinds,
    )
    pool_ids = set(pool_identity_kinds)
    overlaps = {
        "kpi:test": len(kpi_ids & test_ids),
        "kpi:pool": len(kpi_ids & pool_ids),
        "test:pool": len(test_ids & pool_ids),
    }
    overlaps = {key: value for key, value in overlaps.items() if value}
    if overlaps:
        raise ValueError(f"DEFT OD AOI pools overlap: {overlaps}")
    documents = {"kpi": kpi, "test": test, **pools}
    texture_counts = {
        name: dict(
            sorted(Counter(image["deft_od_aoi"]["texture"] for image in value["images"]).items())
        )
        for name, value in documents.items()
    }
    defect_counts = {
        name: dict(
            sorted(
                Counter(
                    image["deft_od_aoi"].get("defect_type", "")
                    for image in value["images"]
                    if image["deft_od_aoi"].get("defect_type")
                ).items()
            )
        )
        for name, value in documents.items()
    }
    report = {
        "status": "valid",
        "manifest": str(manifest_path),
        "strict_metadata": strict,
        "outputs": {
            name: {"images": len(value["images"]), "annotations": len(value["annotations"])}
            for name, value in documents.items()
        },
        "metadata_sources": dict(sorted(sources.items())),
        "texture_counts": texture_counts,
        "defect_type_counts": defect_counts,
        "mining_sources": mining_reports,
        "clean_sources": clean_reports,
        "overlaps": {},
    }
    return documents, report


def _materialize_file(source: Path, destination: Path, mode: str, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        same = (
            destination.resolve() == source.resolve()
            if destination.is_symlink()
            else filecmp.cmp(source, destination, shallow=False)
        )
        if not same:
            raise ValueError(f"{label} collision: {destination}")
        return
    if mode == "symlink":
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def _materialize_view(document: dict[str, Any], output: Path, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for image in document["images"]:
        source = Path(image["source_path"])
        destination = output / image["file_name"]
        _materialize_file(source, destination, mode, "image-view")


def _materialize_synthesis_clean_view(
    document: dict[str, Any], output: Path, mode: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for image in document["images"]:
        metadata = image["deft_od_aoi"]
        texture_key = f"{_slug(metadata['benchmark'])}_{_slug(metadata['texture'])}"
        directory = output / texture_key / "clean_image"
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(image["source_path"])
        destination = directory / image["file_name"]
        _materialize_file(source, destination, mode, "synthesis clean-view")
        counts[texture_key] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    try:
        args = parse_args()
        documents, report = prepare(args)
        if args.output_dir:
            output = Path(args.output_dir).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            manifest = json.loads(
                Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8")
            )
            _write_json(output / "dataset_sources.json", manifest)
            for name, value in documents.items():
                _write_json(output / f"{name}.json", value)
            _materialize_view(documents["kpi"], output / "kpi_images", args.link_mode)
            _materialize_view(documents["test"], output / "test_images", args.link_mode)
            clean_view = output / "anomalygen_clean"
            clean_view_counts = _materialize_synthesis_clean_view(
                documents["clean"], clean_view, args.link_mode
            )
            report["image_views"] = {
                "kpi": str(output / "kpi_images"),
                "test": str(output / "test_images"),
                "anomalygen_clean": str(clean_view),
                "link_mode": args.link_mode,
            }
            report["anomalygen_clean_texture_counts"] = clean_view_counts
            _write_json(output / "source_preparation_report.json", report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
