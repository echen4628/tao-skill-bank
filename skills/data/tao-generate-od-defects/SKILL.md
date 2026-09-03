---
name: tao-generate-od-defects
description: >-
  Run AnomalyGenNext inference from completed, hash-validated object-detection testcases. Generate,
  evaluate, select, pseudo-label, and reconcile synthetic defects as COCO while preserving
  fine-grained defect types. This version supports inference only and requires an existing
  task-fine-tuned AnomalyGenNext checkpoint. Use when asked to run AnomalyGenNext inference,
  generate OD defects from frozen inputs, create synthetic defect images with COCO labels, or
  validate an AnomalyGenNext object-detection dataset.
license: Apache-2.0
compatibility: Requires the pinned AnomalyGenNext 1.1 container, one or more CUDA GPUs, completed inputs from tao-prepare-anomalygennext-inputs, a Cosmos3-Nano base checkpoint, and an existing task-fine-tuned AnomalyGenNext checkpoint.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
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

Consume only frozen inputs from `tao-prepare-anomalygennext-inputs` and run
AnomalyGenNext in `inference_only` mode to produce a validated synthetic defect
dataset with native fine-grained COCO labels.

## Boundary

This skill does not read an OD gap parquet, select false negatives, compute
embeddings, retrieve clean images, or run source-mask filtering. Those decisions
are already frozen in the input manifest and testcase JSONLs.

This skill does not fine-tune AnomalyGenNext. Every generation-plan row must
point to an existing checkpoint already fine-tuned for that row's anomaly types
and to its matching recipe. Future full and fine-tune-only workflows belong in
this generic data-skill layer, not in a DEFT application overlay.

The structured execution metadata is in `references/skill_info.yaml`. This
action runs in the pinned AnomalyGenNext 1.1 container. The image contains the
release code and Python environment at `/workspace/paidf-anomalygen`; stage only
the bank wrapper, frozen inputs, task weights, base checkpoints, and outputs.
Resolve relative `command` and script-default paths against this skill directory
before staging.

Do not substitute the PAIDF AnomalyGen 1.0.1 image. It is the older
Cosmos-Predict2 release and is incompatible with Cosmos3-Nano task weights.
Read `references/container-runtime.md` when pulling the official 1.1 image or
constructing the Docker invocation.

## Inputs

- Completed input root containing
  `prepared_anomalygennext_inputs/prepared_inputs_manifest.json`.
- The container image resolved from `references/skill_info.yaml`.
- Cosmos3-Nano base checkpoint.
- A task-fine-tuned AnomalyGenNext checkpoint and matched recipe for each
  dataset route, frozen into the generation plan.
- One or more visible GPUs.
- Optional comma-separated dataset subset from the frozen plan.

The manifest must report `COMPLETE` and `generation_ready=true`. Every artifact
hash is recomputed before GPU work starts. `defect_spec` is required by the
preparation skill and is already represented in the frozen testcases; it is not
a second generation CLI argument.

## Docker generation

Resolve the exact image instead of guessing a tag:

```bash
AG_IMAGE=$(
  "$TAO_SKILL_BANK_PATH/.venv/deft/bin/python" \
    "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" \
    images.metropolis_sdg.anomalygen_next
)
```

Then run the wrapper inside that image. The image already supplies Python,
`uv`, and the AnomalyGenNext source tree:

```bash
GEN_SKILL=skills/data/tao-generate-od-defects

bash "$GEN_SKILL/scripts/generate_od_defects.sh" \
  --inputs-dir /path/to/completed-inputs \
  --output-dir /path/to/anomalygen_next_generation \
  --base-checkpoint /path/to/Cosmos3-Nano/model \
  --num-gpus 1
```

Use `--datasets plant_a,texture_set` to generate a self-contained subset without changing
the frozen input directory. `--num-gpus` must match the platform allocation.
Use `anomalygen_next_generation` as the semantic output-directory name.
`--output-dir` must not exist unless `--resume-existing-generation` is supplied;
ordinary runs never overwrite another generation.

Read `references/execution-contract.md` for the exact upstream commands and
output accounting. Read `references/container-runtime.md` for a complete
`docker run` example, required mounts, cache locations, private-registry access,
and the image-version check. Platform-specific staging remains owned by the
selected platform skill.

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
DATASET/raw/
DATASET/searched/reconstructed_image/
DATASET/searched/pseudo_labels/coco_annotations.json
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

## Training handoff

Treat `source_tag` as opaque provenance and propagate it from the frozen input
manifest. This skill never appends outputs to a detector training pool and
always reports `training_pool_mutated=false`. A calling application owns the
separate, validated training-admission boundary.

## Gallery

After completion, build a self-contained provenance gallery:

```bash
python "$GEN_SKILL/scripts/build_od_defect_gallery.py" \
  --prepared-inputs-root /path/to/completed-inputs \
  --generation-root /path/to/anomalygen_next_generation \
  --output-dir /path/to/anomalygen_next_generation/synthetic_gallery
```

The gallery shows FN image, source mask, clean neighbor, aligned mask, generated
defect, and pseudo-label overlay for each generated row.
