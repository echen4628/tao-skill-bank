# DEFT OD AOI pipeline and artifacts

Iteration 0 is warehouse-checkpoint inference plus loose and strict KPI gaps.
Training iteration N consumes gaps from iteration N-1, then produces the
checkpoint, metrics, and gaps labeled N. A stage's output becomes input only
after the platform reports `COMPLETE` and its artifact gate passes. Stage to
leaf-skill mapping is in `references/scripts-and-agents.md`.

## Contents

- 1. Inference
- 2. Dual gap analysis
- 3. Route and admit real data
- 4. Generate and admit synthetic data
- 5. Assemble cumulative COCO
- 6. Probe, train, and select
- 7. Measure, gap, and advance
- Suggested result layout

## 1. Inference

Invoke `tao-train-rtdetr` action `inference` on KPI and test with the selected
checkpoint. Iteration 0 uses the frozen warehouse checkpoint; later inference
uses that iteration's KPI-selected checkpoint. Emit KITTI predictions with a
threshold low enough to preserve candidates for both gap passes; the reference
used 0.001. Preserve a one-line `defect` class map.

## 2. Dual gap analysis

Write both specs:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/write_gap_specs.py \
  --policy "${RESULTS_DIR}/deft_od_aoi_policy.json" \
  --ground-truth-ann-path "${KPI_KITTI_LABELS}" \
  --inference-ann-path "${ITER_DIR}/inference/kpi/labels" \
  --images-dir "${KPI_IMAGES}" \
  --kpi "iter${ITERATION}" \
  --output-dir "${ITER_DIR}/gaps"
```

Invoke `tao-analyze-gaps-od-map` once with each emitted
`od_gap_spec.yaml`. Gate on both `box_gaps.parquet` files.

## 3. Route and admit real data

Run `scripts/route_deft_od_aoi.py` with iteration N-1 loose and strict gap parquets,
frozen policy, four pool paths, and the prior committed ledgers/index. For
iteration 2 and later, pass strict gap N-2 as conversion-old and strict gap
N-1 as conversion-new when both exist. When synthesis is enabled, also pass a
frozen JSON array through `--valid-generator-types`; unsupported pockets remain
real-mining routes and are reported rather than sent to a nonexistent model.

Outputs:

```text
routing/mined_manifest.json
routing/synthetic_plan.json
routing/routing_report.json
routing/defect_ledger.json
routing/clean_ledger.json
routing/admission_index.npy
```

The manifest contains newly admitted real and clean images. Ledgers and the
admission index are cumulative and must be advanced atomically only after the
route stage is committed.

## 4. Generate and admit synthetic data

When `synthetic_plan.json` is non-empty, invoke these leaf skills in order:

1. `tao-prepare-anomalygennext-inputs` for box-level strict-FN queries,
   pair-preserving clean retrieval, masks, and frozen testcases;
2. `tao-generate-image-embeddings` for both preparation specs;
3. `tao-generate-od-defects` for generation, pseudo-labeling, validation, and
   binary COCO output.

Use only the per-pocket counts from the frozen synthetic plan. Generation may
return fewer admitted images after validation; never fill the difference with
unvalidated output. If synthesis is enabled and requested but produces no
valid images, stop the loop.

## 5. Assemble cumulative COCO

Pass every committed route manifest through the current iteration and every
admitted synthetic source:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/assemble_deft_od_aoi_coco.py \
  --route-manifest "${ITER1_MANIFEST}" \
  --route-manifest "${CURRENT_MANIFEST}" \
  --synthetic-source "${SYNTH_COCO}::${SYNTH_IMAGES}" \
  --output-coco "${ITER_DIR}/train/annotations.json" \
  --output-images-dir "${ITER_DIR}/train/images"
```

Repeat flags for all committed sources; do not literally pass only iteration
1 and the current iteration when intermediate iterations exist. Gate on the
assembly report, one `defect` category, and an image count equal to the sum of
unique admitted sources.

## 6. Probe, train, and select

Use `scripts/write_rtdetr_specs.py` to emit nested main and inference specs.
When `training.probes_enabled` is true, also emit three probe specs from
iteration 3 onward. Each probe and main train is a separate job-record and
starts from the frozen warehouse checkpoint. After probes,
`scripts/select_deft_od_aoi_probe.py` applies the KPI-best deltas and frozen epoch
budget to the main spec. If probes are disabled, skip that script; the main
spec already has frozen learning rates and the size-based epoch budget.

After the main job, run `scripts/select_deft_od_aoi_checkpoint.py`. When it emits
`action: extend`, set `train.num_epochs` to `extended_num_epochs` and
`train.resume_training_checkpoint_path` to `resume_checkpoint`, then submit one
extension of that same iteration. Rerun selection with `--extension-applied`.
Never extend twice. Gate on the selected `model_epoch_*.pth` and replace the
literal `DEFT_OD_AOI_SELECTED_CHECKPOINT` in both inference specs before launch.

## 7. Measure, gap, and advance

Run inference/evaluation on KPI and test using the fixed selected checkpoint.
Record both metrics, then create loose and strict KPI gaps labeled N. Route the
next iteration only from those KPI gaps. Continue through `max_iterations`; a
target metric is report-only unless the user froze a different stopping
contract before launch.

## Suggested result layout

```text
results_dir/
  deft_od_aoi_policy.json
  input_validation.json
  iterations/iterN/
    inference/{kpi,test}/
    gaps/{loose,strict}/
    routing/
    synthetic/
    train/{images,annotations.json,assembly_report.json}/
    probes/
    main_train/
    selection.json
    metrics.json
```
