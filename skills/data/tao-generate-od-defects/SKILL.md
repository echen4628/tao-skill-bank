---
name: tao-generate-od-defects
description: >-
  Generate synthetic object-detection defects with AnomalyGenNext from a completed, hash-validated
  testcase input directory. Run generation, evaluation, zero-round selection, pseudo-labeling, and
  COCO reconciliation while preserving fine-grained defect types. Use when asked to generate OD
  defects, run AnomalyGenNext from frozen inputs, create synthetic defect images with COCO labels,
  or validate an AnomalyGenNext object-detection dataset.
license: Apache-2.0
compatibility: Requires an AnomalyGenNext Python environment, uv, one or more CUDA GPUs, and completed inputs from tao-prepare-anomalygen-inputs.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- data
- anomalygen-next
- object-detection
- synthetic-data
- coco
---

# Generate AnomalyGenNext OD Defects

Consume only frozen inputs from `tao-prepare-anomalygen-inputs` and produce a
validated synthetic defect dataset with native fine-grained COCO labels.

## Boundary

This skill does not read an OD gap parquet, select false negatives, compute
embeddings, retrieve clean images, or run source-mask filtering. Those decisions
are already frozen in the input manifest and testcase JSONLs.

## Inputs

- Completed input root containing `phase1/phase1_manifest.json`.
- AnomalyGenNext checkout or shared installation.
- Cosmos3-Nano base checkpoint.
- One or more visible GPUs.
- Optional comma-separated dataset subset from the frozen plan.

The manifest must report `COMPLETE` and `phase2_ready=true`. Every artifact hash
is recomputed before GPU work starts.

## Quick Start

Run inside the selected platform allocation after its preflight and launch
review have completed:

```bash
GEN_SKILL=skills/data/tao-generate-od-defects

bash "$GEN_SKILL/scripts/generate_od_defects.sh" \
  --inputs-dir /path/to/completed-inputs \
  --output-dir /path/to/generated-defects \
  --anomalygen-repo /path/to/cosmos3-anomalygen \
  --base-checkpoint /path/to/Cosmos3-Nano/model \
  --num-gpus 1
```

Use `--datasets dagm,mpdd` to generate a self-contained subset without changing
the frozen input directory. `--num-gpus` must match the platform allocation.

Read `references/execution-contract.md` for the exact upstream commands and
output accounting.

## Workflow

For each selected dataset group:

1. Recompute and validate every frozen-input SHA-256.
2. Run AnomalyGenNext distributed generation.
3. Evaluate raw generated images.
4. Run zero-round selection/copying; do not launch iterative refinement without
   a separately approved policy.
5. Evaluate the selected output.
6. Generate native fine-grained pseudo-labels.
7. Reconcile requested, generated, blocked, pseudo-labeled, and annotated rows.
8. Merge per-group COCO files and write the optional binary `defect` projection.

## Output Contract

```text
generation/DATASET/raw/
generation/DATASET/searched/reconstructed_image/
generation/DATASET/searched/pseudo_labels/coco_annotations.json
pseudo_labels/coco_annotations.json
pseudo_labels/coco_annotations_od_defect.json
validation_summary.json
report/index.html
status.json
```

`pseudo_labels/coco_annotations.json` is authoritative and retains the
fine-grained `TEXTURE+TYPE` categories. The `_od_defect` file is only a binary
projection for compatible detector training.

## Completion Gates

- `generated + guardrail_blocked == requested` for every group.
- Pseudo-labeled image count equals generated image count.
- Every COCO image resolves to a generated file and has at least one annotation.
- Every annotation references an existing image and category.
- Every bbox has positive dimensions and lies inside the image.
- Every native category belongs to the frozen group's anomaly types.
- `validation_summary.json` reports `status=COMPLETE` and
  `training_pool_mutated=false`.

## Quarantine

Treat `source_tag` as opaque provenance and propagate it from the frozen input
manifest. Also propagate the frozen boolean `training_eligible` decision
without interpreting or changing it. This skill never appends outputs to a
detector training pool; a calling application must require explicit eligibility
before staging data for training.

## Gallery

After completion, build a self-contained provenance gallery:

```bash
python "$GEN_SKILL/scripts/build_od_defect_gallery.py" \
  --phase1-root /path/to/completed-inputs \
  --phase2-root /path/to/generated-defects \
  --output-dir /path/to/generated-defects/synthetic_gallery
```

The gallery shows FN image, source mask, clean neighbor, aligned mask, generated
defect, and pseudo-label overlay for each generated row.
