#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained visual gallery for an AnomalyGenNext OD run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def representative_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["anomaly_type"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (str(row["fn_id"]), int(row["generation_index"])))
    selected: list[dict[str, Any]] = []
    selected_fn_ids: set[str] = set()
    anomaly_types = sorted(buckets)
    while len(selected) < limit:
        made_progress = False
        for anomaly_type in anomaly_types:
            bucket = buckets[anomaly_type]
            if not bucket:
                continue
            unseen_index = next(
                (i for i, row in enumerate(bucket) if str(row["fn_id"]) not in selected_fn_ids),
                0,
            )
            row = bucket.pop(unseen_index)
            selected.append(row)
            selected_fn_ids.add(str(row["fn_id"]))
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break
    return sorted(selected, key=lambda row: int(row["generation_index"]))


def generation_outputs(phase2: Path, dataset_id: str) -> dict[tuple[str, str], str]:
    ledger = (
        phase2
        / "generation"
        / dataset_id
        / "raw"
        / "texture_ft_generation_result.csv"
    )
    result: dict[tuple[str, str], str] = {}
    with ledger.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["image_filename"]), str(row["mask_filename"]))
            if key in result:
                raise ValueError(f"duplicate generation input in {ledger}: {key}")
            result[key] = str(row["output_filename"])
    return result


def copy_png(source: Path, destination: Path, *, mask: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        converted = image.convert("L" if mask else "RGB")
        converted.thumbnail((960, 960), Image.Resampling.LANCZOS)
        converted.save(destination, format="PNG", optimize=True)


def asset(
    output: Path,
    dataset_id: str,
    generation_index: int,
    stage: str,
    source: Path,
    *,
    mask: bool = False,
) -> str:
    relative = Path("assets") / safe_name(dataset_id) / f"{generation_index:05d}" / f"{stage}.png"
    copy_png(source, output / relative, mask=mask)
    return relative.as_posix()


def build(args: argparse.Namespace) -> None:
    if args.max_cards_per_dataset is not None and args.max_cards_per_dataset <= 0:
        raise ValueError("--max-cards-per-dataset must be positive")
    phase1 = Path(args.phase1_root)
    phase2 = Path(args.phase2_root)
    output = Path(args.output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    phase1_manifest_path = phase1 / "phase1" / "phase1_manifest.json"
    phase1_manifest = json.loads(phase1_manifest_path.read_text())
    validation_path = phase2 / "validation_summary.json"
    validation = json.loads(validation_path.read_text())
    if phase1_manifest.get("status") != "COMPLETE" or validation.get("status") != "COMPLETE":
        raise ValueError("both Phase 1 and Phase 2 must be COMPLETE")
    if validation.get("phase1_manifest_sha256") != sha256(phase1_manifest_path):
        raise ValueError("Phase 2 does not match the current Phase 1 manifest hash")

    selected_datasets = {str(group["dataset_id"]) for group in validation["groups"]}
    rows = [
        row
        for row in read_jsonl(phase1 / "phase1" / "anomalygen_inputs.jsonl")
        if str(row["dataset_id"]) in selected_datasets
    ]
    output_ledgers = {
        dataset_id: generation_outputs(phase2, dataset_id)
        for dataset_id in selected_datasets
    }
    available_rows = []
    missing_generation_rows = 0
    for row in rows:
        dataset_id = str(row["dataset_id"])
        anomaly_type = str(row["anomaly_type"])
        generation_index = int(row["generation_index"])
        generation_key = (str(row["clean_filepath"]), str(row["aligned_mask"]))
        output_filename = output_ledgers[dataset_id].get(generation_key)
        if output_filename is None:
            missing_generation_rows += 1
            continue
        row = dict(row)
        row["_output_filename"] = output_filename
        available_rows.append(row)

    expected_generated = int(validation["generated_images"])
    expected_blocked = int(validation.get("guardrail_blocked", 0))
    if len(available_rows) != expected_generated:
        raise ValueError(
            f"gallery/generated mismatch: available={len(available_rows)} "
            f"validation={expected_generated}"
        )
    if missing_generation_rows != expected_blocked:
        raise ValueError(
            f"missing/guardrail-blocked mismatch: missing={missing_generation_rows} "
            f"validation={expected_blocked}"
        )

    display_rows = []
    for dataset_id in sorted(selected_datasets):
        dataset_rows = [row for row in available_rows if str(row["dataset_id"]) == dataset_id]
        display_rows.extend(representative_rows(dataset_rows, args.max_cards_per_dataset))

    cards = []
    gallery_manifest = []
    for row in display_rows:
        dataset_id = str(row["dataset_id"])
        anomaly_type = str(row["anomaly_type"])
        generation_index = int(row["generation_index"])
        output_filename = str(row["_output_filename"])
        generated_stem = Path(output_filename).stem
        searched = phase2 / "generation" / dataset_id / "searched"
        reconstructed = searched / "reconstructed_image" / f"{generated_stem}.png"
        pseudo_label_overlay = (
            searched / "pseudo_labels" / "visualization" / f"{generated_stem}.png"
        )

        stage_assets = {
            "fn_image": asset(
                output, dataset_id, generation_index, "01_fn_image", Path(row["fn_filepath"])
            ),
            "source_mask": asset(
                output,
                dataset_id,
                generation_index,
                "02_source_mask",
                Path(row["source_mask_copy"]),
                mask=True,
            ),
            "clean_image": asset(
                output, dataset_id, generation_index, "03_clean_neighbor", Path(row["clean_filepath"])
            ),
            "aligned_mask": asset(
                output,
                dataset_id,
                generation_index,
                "04_aligned_mask",
                Path(row["aligned_mask"]),
                mask=True,
            ),
            "generated": asset(
                output, dataset_id, generation_index, "05_generated", reconstructed
            ),
            "pseudo_label_overlay": asset(
                output,
                dataset_id,
                generation_index,
                "06_pseudo_label",
                pseudo_label_overlay,
            ),
        }
        branch = str(row["mask_branch"])
        similarity = float(row["cosine_similarity"])
        stages = (
            ("FN image", stage_assets["fn_image"]),
            (f"Source mask · {branch}", stage_assets["source_mask"]),
            (f"Clean neighbor · rank {int(row['neighbor_rank'])}", stage_assets["clean_image"]),
            ("AMP-aligned mask", stage_assets["aligned_mask"]),
            ("Generated defect", stage_assets["generated"]),
            ("Pseudo-label overlay · generated image", stage_assets["pseudo_label_overlay"]),
        )
        stage_html = "".join(
            f'<figure><a href="{html.escape(path)}" target="_blank">'
            f'<img loading="lazy" src="{html.escape(path)}" alt="{html.escape(label)}"></a>'
            f'<figcaption>{html.escape(label)}</figcaption></figure>'
            for label, path in stages
        )
        cards.append(
            f'<article class="sample" data-dataset="{html.escape(dataset_id)}" '
            f'data-branch="{html.escape(branch)}">'
            f'<div class="sample-head"><div><span class="type">{html.escape(anomaly_type)}</span>'
            f'<span class="badge">{html.escape(branch)}</span></div>'
            f'<div class="score">cosine {similarity:.4f}</div></div>'
            f'<div class="chain">{stage_html}</div>'
            f'<details><summary>Provenance</summary><dl>'
            f'<dt>FN ID</dt><dd>{html.escape(str(row["fn_id"]))}</dd>'
            f'<dt>Pair ID</dt><dd>{html.escape(str(row["pair_id"]))}</dd>'
            f'<dt>OD category</dt><dd>{html.escape(str(row["od_category"]))}</dd>'
            f'<dt>Clean image</dt><dd>{html.escape(str(row["clean_filepath"]))}</dd>'
            f'<dt>Source mask</dt><dd>{html.escape(str(row["source_mask"]))}</dd>'
            f'</dl></details></article>'
        )
        gallery_manifest.append(
            {
                "dataset_id": dataset_id,
                "generation_index": generation_index,
                "anomaly_type": anomaly_type,
                "od_category": row["od_category"],
                "mask_branch": branch,
                "fn_id": row["fn_id"],
                "pair_id": row["pair_id"],
                "cosine_similarity": similarity,
                "neighbor_rank": int(row["neighbor_rank"]),
                "assets": stage_assets,
            }
        )

    datasets = sorted({str(row["dataset_id"]) for row in rows})
    filters = '<button class="active" data-filter="all">All</button>' + "".join(
        f'<button data-filter="{html.escape(dataset)}">{html.escape(dataset.upper())}</button>'
        for dataset in datasets
    )
    group_summary = "".join(
        f'<div class="metric"><strong>{int(group["generated"])}</strong>'
        f'<span>{html.escape(str(group["anomaly_type"]))}</span></div>'
        for group in validation["groups"]
    )
    checkpoint = phase1_manifest["generator_groups"][0]["checkpoint"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnomalyGenNext synthetic gallery</title>
<style>
:root{{--bg:#090c0a;--panel:#121713;--panel2:#192019;--line:#2a342b;--ink:#edf4ee;--muted:#9daf9f;--green:#76b900;--blue:#66b4ff}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#090c0a,#101611);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
.wrap{{max-width:1800px;margin:auto;padding:32px 24px 80px}}header{{border-left:4px solid var(--green);padding-left:18px}}h1{{margin:0;font-size:28px}}header p{{margin:6px 0;color:var(--muted)}}code{{color:#bde57d;overflow-wrap:anywhere}}
.summary{{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px;margin:24px 0}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}}.metric strong{{display:block;font-size:24px;color:white}}.metric span{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin:28px 0 16px;position:sticky;top:0;padding:12px 0;background:#0d120edd;backdrop-filter:blur(8px);z-index:3}}button{{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:999px;padding:8px 14px;cursor:pointer}}button.active{{background:var(--green);color:#081000;border-color:var(--green);font-weight:700}}
.sample{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:0 0 18px;overflow:hidden}}.sample-head{{display:flex;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid var(--line)}}.type{{font-weight:750;color:white}}.badge{{margin-left:10px;padding:3px 8px;border-radius:999px;background:#243423;color:#bde57d;font-size:11px}}.score{{color:var(--blue);font-variant-numeric:tabular-nums}}
.chain{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:1px;background:var(--line)}}figure{{margin:0;background:#0c100d;min-width:0}}figure a{{display:block;aspect-ratio:1/1;overflow:hidden}}figure img{{width:100%;height:100%;object-fit:contain;background:#050705;transition:transform .2s}}figure img:hover{{transform:scale(1.025)}}figcaption{{padding:8px 10px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}}details{{padding:10px 16px;color:var(--muted)}}summary{{cursor:pointer;color:#cbd8cc}}dl{{display:grid;grid-template-columns:120px 1fr;gap:4px 12px}}dt{{color:#7f9181}}dd{{margin:0;overflow-wrap:anywhere}}
.notice{{padding:12px 16px;border:1px solid #456322;background:#172414;border-radius:10px;color:#cfe9b1;margin-top:20px}}footer{{margin-top:30px;color:var(--muted)}}
@media(max-width:1100px){{.chain{{grid-template-columns:repeat(3,1fr)}}.summary{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{.chain{{grid-template-columns:1fr 1fr}}.wrap{{padding:20px 10px}}}}
</style></head><body><div class="wrap">
<header><h1>AnomalyGenNext synthetic gallery</h1><p>Exact defect-type routing · frozen provenance</p></header>
<div class="notice">The native labels preserve each specific AnomalyGenNext type. <code>defect</code> is only the optional downstream binary-OD projection.<br>
<b>Overlay legend:</b> the final panel is the generated defective image with the pseudo-label's translucent blue instance mask and red COCO bounding box. <b>Cosine</b> is whole-image SigLIP similarity from the FN image to the selected clean neighbor; it is a retrieval score, not model confidence or generation quality.</div>
<section class="summary"><div class="metric"><strong>{validation['generated_images']}/{validation['requested_rows']}</strong><span>generated / requested</span></div><div class="metric"><strong>{validation['annotations']}</strong><span>pseudo-label annotations</span></div>{group_summary}</section>
<p><b>Checkpoint:</b> <code>{html.escape(str(checkpoint))}</code><br><b>Phase 1 SHA-256:</b> <code>{html.escape(validation['phase1_manifest_sha256'])}</code><br><b>Training pool mutated:</b> <code>false</code></p>
<nav class="filters" id="gallery">{filters}<button data-branch="fn_mask">FN masks</button><button data-branch="same_type_sampled_mask">Sampled masks</button></nav>
<main>{''.join(cards)}</main><footer>Source tag: <code>{html.escape(validation['source_tag'])}</code> · Gallery contains {len(cards)} immutable provenance chains.</footer>
</div><script>
const buttons=[...document.querySelectorAll('button')],cards=[...document.querySelectorAll('.sample')];
buttons.forEach(b=>b.addEventListener('click',()=>{{buttons.forEach(x=>x.classList.remove('active'));b.classList.add('active');const f=b.dataset.filter,br=b.dataset.branch;cards.forEach(c=>c.hidden=!(f==='all'||(f&&c.dataset.dataset===f)||(br&&c.dataset.branch===br)));}}));
</script></body></html>"""
    (output / "index.html").write_text(page)
    manifest_path = output / "gallery_manifest.json"
    manifest_path.write_text(json.dumps(gallery_manifest, indent=2) + "\n")
    print(
        f"gallery PASS: cards={len(cards)} output={output} "
        f"manifest_sha256={sha256(manifest_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--phase2-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cards-per-dataset", type=int)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
