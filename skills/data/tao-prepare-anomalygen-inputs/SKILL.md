---
name: tao-prepare-anomalygen-inputs
description: >-
  Prepare frozen AnomalyGenNext object-detection inputs from box-level false negatives: resolve
  compatible defect types and source masks, emit FN and clean-image embedding inputs, preserve
  pairwise clean-neighbor provenance, run automatic mask placement, and write hash-validated
  testcase JSONL files. Use when asked to prepare AnomalyGenNext data, turn OD false negatives into
  generator inputs, build AnomalyGen testcase files, or stage FN-driven synthetic-defect inputs.
license: Apache-2.0
compatibility: Requires Python 3.10+ with pandas, pyarrow, NumPy, Pillow, and PyYAML. Finalization requires an AnomalyGenNext environment with a CUDA GPU.
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
- input-preparation
---

# Prepare AnomalyGenNext Inputs

Turn box-level OD false negatives into immutable AnomalyGenNext testcase and
provenance files. Do not generate synthetic images; invoke
`tao-generate-od-defects` after this skill completes.

## Boundary

This skill owns filtering, source-mask resolution, pair-preserving retrieval,
AMP, and the frozen-input manifest. It intentionally does not compute image
embeddings. Invoke `tao-generate-image-embeddings` for the two specs emitted by
the first command.

```text
box_gaps.parquet
  -> prepare sources
  -> tao-generate-image-embeddings (clean + FN)
  -> pair-preserving cosine retrieval + AMP
  -> frozen testcase.jsonl + provenance + manifest
```

The generic `tao-mine-nearest-neighbors` output is not currently suitable for
this handoff because it collapses to unique source filepaths. AnomalyGenNext
requires every FN-to-clean pair to remain explicit through AMP and provenance.

## Inputs

Pass one nested YAML config. Start from `assets/default_filtering.yaml` and set:

- `gap_parquet`: box-level OD gaps with `image_id`, `filepath`, `gap_type`,
  `bbox`, and `class`.
- `split_root`: per-dataset `split_manifest.json` files keyed by source path.
- `pool_dataset_root`: AnomalyGen-compatible clean images and same-type masks.
- `defect_spec`: exact `TEXTURE+TYPE` placement definitions.
- `datasets`: path parsing plus matched checkpoint and recipe for each dataset.
- `selection`: either bounded `per_dataset` selection or `all_eligible`.
- `embedding`: one encoder identity reused for FN and clean images.
- `retrieval`: cosine candidate and retention settings.

Read `references/input-contract.md` before adapting a dataset layout.

## Quick Start

Use the project control-plane Python for deterministic host preparation:

```bash
PREP_SKILL=skills/data/tao-prepare-anomalygen-inputs
RUN_ROOT=/path/to/anomalygen-inputs
CONFIG=/path/to/filtering.yaml

bash "$PREP_SKILL/scripts/prepare_anomalygen_sources.sh" \
  --config "$CONFIG" --output-dir "$RUN_ROOT" \
  --python .venv/deft/bin/python
```

The command writes:

```text
specs/clean_embeddings.yaml
specs/fn_embeddings.yaml
manifests/fn_queries.parquet
manifests/selected_fn_queries.parquet
manifests/mask_selection.parquet
```

Invoke `tao-generate-image-embeddings` twice with those exact specs. Do not
change the encoder between calls.

After both output parquets exist, run finalization inside an AnomalyGenNext GPU
environment:

```bash
bash "$PREP_SKILL/scripts/finalize_anomalygen_inputs.sh" \
  --config "$CONFIG" --output-dir "$RUN_ROOT" \
  --anomalygen-repo /path/to/cosmos3-anomalygen
```

This runs pair-preserving cosine retrieval, AMP for both mask branches, freezes
the generator JSONLs and provenance, and validates every recorded SHA-256.

## Invariants

- Preserve every eligible box-level FN even when several boxes share one image.
  Embed the image once, then join that embedding back to every FN query.
- Require exactly two mask branches: the bbox-isolated FN mask and one
  deterministic same-type sampled mask selected independently for that FN.
  Same-image FNs must never inherit a representative FN's mask pair; use
  distinct sampled masks between them when the same-type pool permits.
- Retain a clean neighbor only when AMP succeeds for both branches.
- Treat `fn_id` as the generation unit. Different FNs may reuse the same clean
  image, but each `(fn_id, clean_image)` pair must run its own two AMP branches
  and produce separately keyed testcase, provenance, and generated-output rows.
- Prevent duplicate clean neighbors only within one FN's retained list; do not
  enforce global clean-image uniqueness across different FN ids.
- Match every anomaly type against both the recipe and `defect_spec.jsonl`.
- Never edit a completed frozen-input directory. Start a new directory when
  configuration, source gaps, checkpoint, or recipe changes.
- Treat `source_tag` as an optional opaque provenance label, defaulting to
  `user_provided`. Preserve it and the explicit boolean `training_eligible`
  decision in all handoff manifests; never infer eligibility from the label or
  a path.
- Keep `training_pool_mutated: false` in all produced manifests. Downstream
  application staging is the only boundary that may add generated data to a
  training pool.

## Outputs

The completion gate is `phase1/phase1_manifest.json` with `status=COMPLETE` and
`phase2_ready=true`. Its `artifacts` array hashes the configuration snapshot,
input contract, routing plan, generator JSONLs, and provenance files.

```text
phase1/
  filtering_config.yaml
  input_contract.json
  phase1_manifest.json
  phase2_plan.json
  anomalygen_inputs.jsonl
  anomalygen_inputs/DATASET/testcase.jsonl
  anomalygen_inputs/DATASET/provenance.jsonl
```

Run the gate independently with:

```bash
.venv/deft/bin/python \
  "$PREP_SKILL/scripts/prepare_anomalygen_inputs.py" validate-phase1 \
  --phase1-root "$RUN_ROOT"
```

## Failure Handling

- Record ineligible rows with explicit skip reasons; never silently drop them.
- Stop when an embedding is missing, non-finite, zero-norm, or produced by a
  mismatched encoder.
- Stop if no generator rows survive AMP.
- Refuse reuse when the supplied config differs from the frozen snapshot.
- Do not regenerate or repair frozen files in place after the manifest is
  complete.
