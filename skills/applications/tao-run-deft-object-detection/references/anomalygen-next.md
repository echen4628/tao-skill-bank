# Optional AnomalyGenNext producer in the DEFT OD loop

Read this during preflight when synthetic augmentation is requested, and again
during each iteration's `stage` boundary. This is a reusable loop contract; it
contains no user, cluster, dataset, checkpoint, or run-specific paths.

The data skills own the reusable producer work:

- `tao-prepare-anomalygennext-inputs` owns FN/type eligibility, prompts, masks,
  pair-preserving retrieval, AMP, frozen testcases, and provenance.
- `tao-generate-image-embeddings` owns clean and FN embeddings.
- `tao-generate-od-defects` owns generation, pseudo-labeling, COCO merging, and
  generation gates.

The data skills currently support AnomalyGenNext inference only. DEFT needs
only inference: every dataset route supplies a checkpoint already fine-tuned
for its requested anomaly types plus the matching recipe. This application
owns the final admission boundary: validated output is staged, converted to
ODVG for Grounding DINO or kept as COCO for RT-DETR, committed with the
iteration, and appended to the cumulative detector-native training sources.

## Preflight — all values are required when enabled

Resolve and show these in the single launch review:

- AnomalyGenNext platform, installation/repository, Cosmos3-Nano base
  checkpoint, visible GPU count, and expected runtime.
- A nested-YAML preparation template containing real `split_root`,
  `pool_dataset_root`, `defect_spec`, dataset routing, matched checkpoint/recipe
  pairs, selection policy, encoder, and retrieval settings. The loop replaces
  only `gap_parquet` per iteration.
- For every dataset route, require an AnomalyGenNext checkpoint already
  fine-tuned for the requested `TEXTURE+TYPE` values. Confirm that its recipe
  declares those same types. DEFT never fine-tunes AnomalyGenNext.
- `defect_spec` is required. Every selected type must have one entry; a
  `spatial_dependency: text` entry must include a non-empty
  `roi_prompt_defect_location`. Treat that as the authoritative placement
  prompt. Do not infer or synthesize it from the OD false negative.
- An optional non-empty `source_tag` as an opaque provenance label; it defaults
  to `user_provided`. The loop does not attach policy meaning to the label, so
  a fresh user does not need to classify a path with a special name.
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

2. Invoke `tao-prepare-anomalygennext-inputs` source preparation, embed its clean
   and FN specs with the exact frozen encoder, then finalize the inputs.
   Inputs live under `${RESULTS_DIR}/iter${N}/synthetic/inputs/`.
3. Invoke `tao-generate-od-defects` into
   `${RESULTS_DIR}/iter${N}/synthetic/anomalygen_next_generation/`. Require
   `validation_summary.json::status=COMPLETE`, exact generation accounting,
   and `training_pool_mutated=false`.
4. Cross the explicit training-admission boundary:

   ```bash
   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/stage_anomalygen_coco.py \
     --validation-summary "${RESULTS_DIR}/iter${N}/synthetic/anomalygen_next_generation/validation_summary.json" \
     --source-coco "${RESULTS_DIR}/iter${N}/synthetic/anomalygen_next_generation/pseudo_labels/coco_annotations_od_defect.json" \
     --output-images-dir "${RESULTS_DIR}/iter${N}/synthetic/training/images" \
     --output-coco "${RESULTS_DIR}/iter${N}/synthetic/training/synthetic_train.json" \
     --target-class "<config.anomalygen_target_class>" \
     [--category-contract-coco "<config.source_detection_file>"] \
     --report-json "${RESULTS_DIR}/iter${N}/synthetic/training/staging_report.json"
   ```

   The script requires a complete immutable generation summary,
   collision-proofs basenames, and records `training_pool_mutated=true` only in
   its new staging report. It never alters the generator's validation summary.
   Pass `--category-contract-coco` for RT-DETR. It preserves the prepared pool's
   complete category list and assigns the synthetic target its frozen id.
5. For Grounding DINO, convert the staged COCO to ODVG with the standard
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
   For RT-DETR, skip this conversion and validate/use `synthetic_train.json`
   directly.
6. Commit the normal `stage` artifacts plus the common fields below. Add
   `--synthetic-odvg` and `--synthetic-label-map` only for Grounding DINO:

   ```text
   --synthetic-validation-summary <anomalygen_next_generation/validation_summary.json>
   --synthetic-coco <training/synthetic_train.json>
   --synthetic-images-dir <training/images>
   --synthetic-staging-report <training/staging_report.json>
   # Grounding DINO only:
   --synthetic-odvg <training/annotations/synthetic_train_odvg.jsonl>
   --synthetic-label-map <training/annotations/synthetic_train_odvg_labelmap.json>
   ```

7. During `train`, call `update_train_spec.py` with both the normal `--tmm-*`
   source and the detector-native optional `--synthetic-*` source. The output spec therefore
   appends two sources for iteration N. Since it copies iteration N-1's spec,
   all prior mined and synthetic sources remain in the list.

## Iteration-2 invariant

After iteration 1, inference writes the labels consumed by iteration 2 gap
analysis. Iteration 2 then produces a new mined source and, when enabled, a new
synthetic source. Its train spec contains:

```text
seed sources
+ iter1 mined detector-native source
+ iter1 synthetic detector-native source
+ iter2 mined detector-native source
+ iter2 synthetic detector-native source
```

That accumulated list — not the generator output directory by itself — is the
proof that synthetic data participates in the next training cycle.

## Hard stops

- zero eligible/generated synthetic rows when the producer is enabled;
- any incomplete or inconsistent generation summary;
- a target class absent from the detector target set;
- COCO image/annotation/category mismatch, or failed Grounding DINO COCO→ODVG validation;
- a train spec missing either enabled producer's current-iteration source.

For a generation-only experiment, run the two leaf skills outside the active
DEFT result tree and do not call `stage_anomalygen_coco.py` or `commit_stage.py`.
