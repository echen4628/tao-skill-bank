# DEFT OD AOI pipeline and artifacts

Iteration 0 is base-checkpoint inference plus loose and strict KPI gaps.
Training iteration N consumes gaps from iteration N-1, then produces the
checkpoint, metrics, and gaps labeled N. A stage's output becomes input only
after the platform reports `COMPLETE` and its artifact gate passes. Stage to
leaf-skill mapping is in `references/scripts-and-agents.md`.

## Contents

- 0. Normalize and validate inputs
- 1. Inference
- 2. Dual gap analysis
- 3. Route and admit real data
- 4. Generate and admit synthetic data
- 5. Assemble cumulative COCO
- 6. Probe, train, and select
- 7. Measure, gap, and advance
- Suggested result layout

## 0. Inspect, prepare, and validate inputs

For one or many dataset paths, run `inspect_deft_od_aoi_sources.py`, freeze a
strict `dataset_sources.json`, and run `prepare_deft_od_aoi_sources.py` as
specified in `references/preflight.md`. Gate on
`source_preparation_report.json`, zero fallbacks, the frozen source manifest,
and the four canonical COCO files, then run
`validate_deft_od_aoi_inputs.py`. Downstream stages consume only this frozen
view. Do not copy KPI or test images into training, infer clean status from an
unapproved boxless source, or reinterpret metadata after iteration 0 begins.

## 1. Inference

Invoke `tao-train-rtdetr` action `inference` on KPI and test with the selected
checkpoint. Iteration 0 uses the frozen base checkpoint; later inference
uses that iteration's KPI-selected checkpoint. Emit KITTI predictions with a
threshold of 0.001 to preserve candidates for both gap passes. Preserve a
one-line `defect` class map.

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

## 3. Embed, route, and admit real data

Once per frozen source view, run
`scripts/prepare_deft_od_aoi_siglip_candidates.py`. This emits contextual GT
crops for every source annotation and a 1×1 plus 2×2 patch set for every
verified-clean image. Embed its parquet with `tao-generate-image-embeddings`
using the policy's SigLIP model and model path.

Each iteration, run `scripts/prepare_deft_od_aoi_siglip_queries.py` on the N-1
loose and strict gap parquets, then embed that parquet with the exact same
encoder. Run `scripts/route_deft_od_aoi_siglip.py` with both embedding
parquets, frozen policy, four pool paths, and prior committed ledgers/index.
For iteration 2 and later, pass strict gap N-2 as conversion-old and strict gap
N-1 as conversion-new when both exist. When synthesis is enabled, also pass a
frozen JSON array through `--valid-generator-types`; unsupported pockets remain
real-mining routes and are not sent to a nonexistent model.

This is a global search within two hard roles: FN/near-miss queries can select
any defect source, and background-FP queries can select any verified-clean
source. There is no uniform bootstrap and no benchmark/texture eligibility
filter.

Outputs:

```text
routing/mined_manifest.json
routing/synthetic_plan.json
routing/routing_report.json
routing/defect_ledger.json
routing/clean_ledger.json
routing/admission_index.npy
routing/retrieval_audit.parquet
```

The manifest contains newly admitted real and clean images. Ledgers and the
admission index are cumulative and must be advanced atomically only after the
route stage is committed.

## 4. Generate and admit synthetic data

When `synthetic_plan.json` is non-empty, invoke these leaf skills in order
using the AnomalyGenNext assets frozen at launch review (`defect_spec`,
per-route fine-tuned checkpoint and recipe, Cosmos3-Nano base checkpoint, and
AnomalyGenNext checkout). Do not invent types or checkpoints mid-loop.

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
starts from the frozen base checkpoint. After probes,
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
  normalized_inputs/
    dataset_sources.json
    source_preparation_report.json
    {kpi,test,source,clean}.json
    {kpi_images,test_images}/
    anomalygen_clean/<benchmark>_<texture>/clean_image/
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
