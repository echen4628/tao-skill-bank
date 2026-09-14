---
name: tao-prepare-anomalygennext-inputs
description: >-
  Prepare normalized object-detection false negatives, source masks, and clean-image
  embedding inputs for AnomalyGenNext. Use when turning box-level FN gaps into a
  frozen, pair-preserving preparation plan. Do not use for training or generation.
license: Apache-2.0
compatibility: Requires the AnomalyGenNext 1.1 image, pandas, pyarrow, NumPy, Pillow, and PyYAML.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags: [tao, data, anomalygen-next, object-detection, input-preparation]
---

# Prepare AnomalyGenNext Inputs

The first action freezes eligible false-negative identities, isolates each FN
mask to its bounding box, selects one deterministic same-type mask, and emits
the clean and FN embedding specs. The second action consumes those embedding
results, preserves every FN-to-clean pair, and runs native automatic mask
placement (AMP) from the AnomalyGenNext container. The final action accepts
only pairs for which both mask branches passed AMP, then freezes the exact
generation inputs and their hashes.

## Input contract

Start from `assets/default_filtering.yaml`. The gap parquet must contain:

```text
image_id filepath gap_type bbox class split
dataset_id texture_id defect_class anomaly_type fn_mask_source
```

Identity is explicit: `anomaly_type` must equal
`texture_id+defect_class`. Preparation intentionally has no filename or
directory-layout inference. A caller adapting an unfamiliar dataset must
normalize these fields before this boundary.

Each selected dataset maps to an existing fine-tuned checkpoint and recipe.
The recipe must declare the exact anomaly type, and `defect_spec.jsonl` must
contain its placement definition. Text-routed definitions require a nonempty
`roi_prompt_defect_location`. The pool layout is:

```text
POOL/TEXTURE/clean_image/*
POOL/TEXTURE/mask/DEFECT/*
```

## Action

Run through the selected platform after the common launch review:

```bash
scripts/prepare_anomalygennext_inputs.py \
  --config /path/to/filtering.yaml \
  --output-dir /new/result/root
```

The output root must not exist. The action emits `fn_queries.parquet`,
`selected_fn_queries.parquet`, `mask_selection.parquet`, `clean_pool.parquet`,
two `tao-generate-image-embeddings` specs, the copied filtering config, and
`input_contract.json`.

Run both emitted specs through `tao-generate-image-embeddings`, preserving the
same encoder. Place their outputs under `embeddings/` as named by the specs,
then submit the `run_amp` action:

```bash
scripts/run_anomalygennext_amp.py \
  --config /path/to/filtering.yaml \
  --prepared-root /runwork/prepared \
  --published-root /durable/result/root \
  --sam2-checkpoint /models/sam2.1_hiera_large.pt
```

This writes `knn_candidates.parquet`, the native AMP request, and
`amp/testcase.jsonl`. Finalize it in the same durable result root:

The public 1.1 image includes AMP code but not the SAM2.1 payload. Expose the
checkpoint as a read-only runtime input; the wrapper injects it into upstream
AMP without modifying the source image or the frozen filtering config.
The image-owned AnomalyGenNext checkout lives at `/workspace/paidf-anomalygen`.
Mount node-local working data somewhere else (for example `/runwork`); mounting
scratch over `/workspace` hides the native `anomalygen` package and is invalid.
When node-local staging gives AMP a short `--prepared-root`, pass the durable
copy-back location as `--published-root`. Native resize filenames are derived
from the runtime path and can exceed the filesystem component limit if AMP is
executed directly under a long durable path; emitted metadata is rewritten to
the published root only after native AMP succeeds.

Allocate exactly one NVIDIA GPU for `run_amp`; Qwen and SAM2 placement are not
CPU actions. The preparation and finalization actions remain CPU-compatible.
The top-level `gpu_spec_key: null` exists because this mixed-action skill has no
single config field from which a platform can infer that runtime distinction.

```bash
scripts/finalize_anomalygennext_inputs.py --prepared-root /existing/result/root
```

The finalizer writes pair status, selected pairs, copied aligned masks,
per-dataset testcase and provenance JSONL, and a generation plan plus integrity
manifest. A clean image is unique only within one FN; separate FNs may reuse it.
Platform skills own staging and cache placement.

## Gates

- Preserve each box-level FN as a distinct `fn_id`.
- Produce exactly two masks per FN: isolated FN and deterministic same-type.
- Convert both AMP-facing branches from label masks to strict binary `0`/`255`
  with `foreground = source > 0`; preserve the source canvas and record the
  validated tight foreground extent.
- Embed a repeated source image once while preserving every FN query row.
- Require successful, nonempty, non-full-image aligned masks with a non-full
  tight extent for both branches.
- Never infer an anomaly type or placement prompt.
- Refuse output reuse and never mutate a training pool.
- Platform skills own transient staging, caches, and copy-back.

Read `references/input-contract.md` when authoring the normalized parquet or
dataset mapping.
