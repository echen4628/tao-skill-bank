# Optional AnomalyGenNext producer in the DEFT OD loop

Read this during preflight when synthetic augmentation is requested, and again
during each iteration's `stage` boundary. This is a reusable loop contract; it
contains no user, cluster, dataset, checkpoint, or run-specific paths.

The data skills own the reusable producer work:

- `tao-prepare-anomalygen-inputs` owns FN/type eligibility, masks,
  pair-preserving retrieval, AMP, frozen testcases, and provenance.
- `tao-generate-image-embeddings` owns clean and FN embeddings.
- `tao-generate-od-defects` owns generation, pseudo-labeling, COCO merging, and
  generation gates.

This application owns the final admission boundary: only validated output with
the frozen `training_eligible: true` decision is staged, converted to ODVG,
committed with the iteration, and appended to the cumulative Grounding DINO
training sources.

## Preflight — all values are required when enabled

Resolve and show these in the single launch review:

- AnomalyGenNext platform, installation/repository, Cosmos3-Nano base
  checkpoint, visible GPU count, and expected runtime.
- A nested-YAML preparation template containing real `split_root`,
  `pool_dataset_root`, `defect_spec`, dataset routing, matched checkpoint/recipe
  pairs, selection policy, encoder, and retrieval settings. The loop replaces
  only `gap_parquet` per iteration.
- An optional non-empty `source_tag` as an opaque provenance label; it defaults
  to `user_provided`. The loop does not attach policy meaning to the label, so
  a fresh user does not need to classify a path with a special name.
- `training_eligible: true`. This explicit preflight decision is required for
  loop integration. Use `false` for any input that may be used for generation
  or evaluation but must not be added to detector training. Do not infer this
  decision from the path or `source_tag`.
- `anomalygen_target_class`, which must be one of the detector's target classes.
  The current training admission deliberately projects all generated native
  anomaly types onto this one approved detector class. Do not infer it.
- The clean and FN embedding encoder must match the preparation template.

Run the selected platform's normal access, GPU, path, credential, and image
checks. Also validate every stable path in the preparation template before the
user gate. `init_deft_state.py` freezes the approved values so resume cannot
change producer policy.

## Per-iteration sequence

Use the current iteration's `${RESULTS_DIR}/iter${N}/gaps/box_gaps.parquet`.
`weak_images.parquet` is for mining and is not a substitute: AnomalyGenNext
needs each FN box and mask identity.

1. Materialize a fresh nested config without editing the template:

   ```bash
   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
     --spec "<config.anomalygen_config_template>" \
     --out "${RESULTS_DIR}/iter${N}/synthetic/filtering.yaml" \
     --set "gap_parquet=${RESULTS_DIR}/iter${N}/gaps/box_gaps.parquet"
   ```

2. Invoke `tao-prepare-anomalygen-inputs` source preparation, embed its clean
   and FN specs with the exact frozen encoder, then finalize the inputs.
   Inputs live under `${RESULTS_DIR}/iter${N}/synthetic/inputs/`.
3. Invoke `tao-generate-od-defects` into
   `${RESULTS_DIR}/iter${N}/synthetic/generation/`. Require
   `validation_summary.json::status=COMPLETE`, exact generation accounting,
   and `training_pool_mutated=false`.
4. Cross the explicit training-admission boundary:

   ```bash
   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/stage_anomalygen_coco.py \
     --validation-summary "${RESULTS_DIR}/iter${N}/synthetic/generation/validation_summary.json" \
     --source-coco "${RESULTS_DIR}/iter${N}/synthetic/generation/pseudo_labels/coco_annotations_od_defect.json" \
     --output-images-dir "${RESULTS_DIR}/iter${N}/synthetic/training/images" \
     --output-coco "${RESULTS_DIR}/iter${N}/synthetic/training/synthetic_train.json" \
     --target-class "<config.anomalygen_target_class>" \
     --report-json "${RESULTS_DIR}/iter${N}/synthetic/training/staging_report.json"
   ```

   The script requires the generated summary to preserve
   `training_eligible=true`, collision-proofs basenames, and records
   `training_pool_mutated=true` only in its new staging report. It never alters
   the generator's immutable validation summary.
5. Convert the staged COCO to ODVG with the standard
   `assets/overlays/coco_to_odvg.yaml` flow used by source-pool preparation.
   Write both files under
   `${RESULTS_DIR}/iter${N}/synthetic/training/annotations/`, then run
   `validate_odvg_images.py` against the staged images.

   ```bash
   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/emit_default_spec.py \
     --stage coco_to_odvg \
     --out "${RESULTS_DIR}/iter${N}/synthetic/training/coco_to_odvg.yaml" \
     --ds-image "$TAO_DS_IMAGE"

   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
     --spec "${RESULTS_DIR}/iter${N}/synthetic/training/coco_to_odvg.yaml" \
     --overlay <skill_root>/assets/overlays/coco_to_odvg.yaml \
     --set "coco.ann_file=${RESULTS_DIR}/iter${N}/synthetic/training/synthetic_train.json" \
     --set "results_dir=${RESULTS_DIR}/iter${N}/synthetic/training/annotations" \
     --require-no-mandatory

   docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
     -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" "$TAO_DS_IMAGE" \
     annotations convert -e "${RESULTS_DIR}/iter${N}/synthetic/training/coco_to_odvg.yaml"

   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/validate_odvg_images.py \
     --image-dir "${RESULTS_DIR}/iter${N}/synthetic/training/images" \
     --odvg "${RESULTS_DIR}/iter${N}/synthetic/training/annotations/synthetic_train_odvg.jsonl" \
     --key-field file_name
   ```
6. Commit the normal `stage` artifacts plus:

   ```text
   --synthetic-validation-summary <generation/validation_summary.json>
   --synthetic-coco <training/synthetic_train.json>
   --synthetic-odvg <training/annotations/synthetic_train_odvg.jsonl>
   --synthetic-label-map <training/annotations/synthetic_train_odvg_labelmap.json>
   --synthetic-images-dir <training/images>
   --synthetic-staging-report <training/staging_report.json>
   ```

7. During `train`, call `update_train_spec.py` with both the normal `--tmm-*`
   triplet and the optional `--synthetic-*` triplet. The output spec therefore
   appends two sources for iteration N. Since it copies iteration N-1's spec,
   all prior mined and synthetic sources remain in the list.

## Iteration-2 invariant

After iteration 1, inference writes the labels consumed by iteration 2 gap
analysis. Iteration 2 then produces a new mined source and, when enabled, a new
synthetic source. Its train spec contains:

```text
seed sources
+ iter1 mined ODVG
+ iter1 synthetic ODVG
+ iter2 mined ODVG
+ iter2 synthetic ODVG
```

That accumulated list — not the generator output directory by itself — is the
proof that synthetic data participates in the next training cycle.

## Hard stops

- zero eligible/generated synthetic rows when the producer is enabled;
- any incomplete or inconsistent generation summary;
- `training_eligible` absent, false, or changed across the producer handoff;
- a target class absent from the detector target set;
- COCO image/annotation mismatch or failed COCO→ODVG validation;
- a train spec missing either enabled producer's current-iteration source.

For a generation-only experiment, run the two leaf skills outside the active
DEFT result tree and do not call `stage_anomalygen_coco.py` or `commit_stage.py`.
