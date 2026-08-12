#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append a completed gallery to a self-contained base gallery."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


ARTICLE = re.compile(r'<article class="sample".*?</article>', re.DOTALL)


def merge(args: argparse.Namespace) -> None:
    base = Path(args.base_gallery).resolve()
    addition = Path(args.addition_gallery).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gallery: {output}")
    base_manifest = json.loads((base / "gallery_manifest.json").read_text())
    addition_manifest = json.loads((addition / "gallery_manifest.json").read_text())
    namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.asset_namespace).strip("_")
    if not namespace:
        raise ValueError("--asset-namespace is empty after normalization")

    base_html = (base / "index.html").read_text()
    addition_html = (addition / "index.html").read_text()
    cards = ARTICLE.findall(addition_html)
    if len(cards) != len(addition_manifest):
        raise ValueError("addition HTML/manifest card count mismatch")
    datasets = {str(row["dataset_id"]) for row in addition_manifest}
    if len(datasets) != 1:
        raise ValueError("addition gallery must contain exactly one dataset asset directory")
    source_dataset = next(iter(datasets))
    old_prefix = f"assets/{source_dataset}/"
    new_prefix = f"assets/{namespace}/"
    cards_html = "".join(card.replace(old_prefix, new_prefix) for card in cards)

    shutil.copytree(base, output)
    shutil.copytree(addition / "assets" / source_dataset, output / "assets" / namespace)
    for row in addition_manifest:
        row["assets"] = {
            key: value.replace(old_prefix, new_prefix)
            for key, value in row["assets"].items()
        }
        row["gallery_source"] = str(addition)
    merged_manifest = base_manifest + addition_manifest
    (output / "gallery_manifest.json").write_text(json.dumps(merged_manifest, indent=2) + "\n")

    total = len(merged_manifest)
    generated_match = re.search(
        r'<strong>(\d+)/(\d+)</strong><span>generated / requested</span>', base_html
    )
    if generated_match is None:
        raise ValueError("base gallery is missing generated/requested accounting")
    base_generated, base_requested = map(int, generated_match.groups())
    addition_generated = args.addition_generated or len(addition_manifest)
    if addition_generated < len(addition_manifest):
        raise ValueError("--addition-generated cannot be smaller than displayed cards")
    addition_requested = args.addition_requested or addition_generated
    if addition_requested < addition_generated:
        raise ValueError("--addition-requested cannot be smaller than generated cards")
    base_annotations = int(
        re.search(r'<strong>(\d+)</strong><span>pseudo-label annotations</span>', base_html).group(1)
    )
    base_html = re.sub(
        r'<strong>\d+/\d+</strong><span>generated / requested</span>',
        f'<strong>{base_generated + addition_generated}/{base_requested + addition_requested}</strong>'
        '<span>generated / requested</span>',
        base_html,
        count=1,
    )
    base_html = re.sub(
        r'<strong>\d+</strong><span>pseudo-label annotations</span>',
        f'<strong>{base_annotations + args.addition_annotations}</strong><span>pseudo-label annotations</span>',
        base_html,
        count=1,
    )
    type_counts = Counter(str(row["anomaly_type"]) for row in addition_manifest)
    metrics = "".join(
        f'<div class="metric"><strong>{count}</strong><span>{name}</span></div>'
        for name, count in sorted(type_counts.items())
    )
    if args.addition_blocked:
        blocked_pattern = r'<strong>(\d+)</strong><span>guardrail blocked</span>'
        blocked_match = re.search(blocked_pattern, base_html)
        if blocked_match:
            blocked_total = int(blocked_match.group(1)) + args.addition_blocked
            base_html = re.sub(
                blocked_pattern,
                f'<strong>{blocked_total}</strong><span>guardrail blocked</span>',
                base_html,
                count=1,
            )
        else:
            metrics += (
                f'<div class="metric"><strong>{args.addition_blocked}</strong>'
                '<span>guardrail blocked</span></div>'
            )
    displayed_pattern = r'<strong>(\d+)</strong><span>gallery cards displayed</span>'
    if re.search(displayed_pattern, base_html):
        base_html = re.sub(
            displayed_pattern,
            f'<strong>{total}</strong><span>gallery cards displayed</span>',
            base_html,
            count=1,
        )
    else:
        metrics += (
            f'<div class="metric"><strong>{total}</strong>'
            '<span>gallery cards displayed</span></div>'
        )
    base_html = base_html.replace("</section>", metrics + "</section>", 1)
    for dataset in sorted(datasets):
        button = f'<button data-filter="{dataset}">{dataset.upper()}</button>'
        if button not in base_html:
            base_html = base_html.replace(
                '<button data-branch="fn_mask">',
                button + '<button data-branch="fn_mask">',
                1,
            )
    base_html = base_html.replace("</main>", cards_html + "</main>", 1)
    base_html = re.sub(
        r'Gallery contains \d+ immutable provenance chains',
        f'Gallery contains {total} immutable provenance chains',
        base_html,
        count=1,
    )
    extra_sha = args.addition_phase1_sha256
    addition_label = args.addition_label or namespace
    base_html = base_html.replace(
        "<br><b>Training pool mutated:</b>",
        f"<br><b>{addition_label} Phase 1 SHA-256:</b> <code>{extra_sha}</code>"
        "<br><b>Training pool mutated:</b>",
        1,
    )
    (output / "index.html").write_text(base_html)
    print(
        f"gallery merge PASS: base={len(base_manifest)} addition={len(addition_manifest)} "
        f"total={total} output={output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gallery", required=True)
    parser.add_argument("--addition-gallery", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-namespace", required=True)
    parser.add_argument("--addition-annotations", type=int, required=True)
    parser.add_argument("--addition-phase1-sha256", required=True)
    parser.add_argument("--addition-requested", type=int)
    parser.add_argument("--addition-generated", type=int)
    parser.add_argument("--addition-blocked", type=int, default=0)
    parser.add_argument("--addition-label")
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
