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
- 8. Consolidate with model soup
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
  --ground-truth-coco "${KPI_COCO}" \
  --inference-ann-path "${ITER_DIR}/inference/kpi/labels" \
  --images-dir "${KPI_IMAGES}" \
  --kpi "iter${ITERATION}" \
  --output-dir "${ITER_DIR}/gaps" \
  --analyze-binary
```

For this binary AOI workflow, `--analyze-binary` performs deterministic
confidence-ranked IoU matching and writes both complete artifact sets in the
same command. The two emitted `od_gap_spec.yaml` files remain available for an
equivalent `tao-analyze-gaps-od-map` run when its action-capable image is
installed. Gate on both `box_gaps.parquet` files.
When `--ground-truth-coco` is supplied, the same command writes the exact
binary KITTI projection (including empty label files for clean KPI images), so
no separate converter is required.

## 3. Embed, route, and admit real data

Once per frozen source view, run
`scripts/prepare_deft_od_aoi_siglip_candidates.py`. This emits contextual GT
crops for every source annotation and a 1×1 plus 2×2 patch set for every
verified-clean image. Embed its parquet with `tao-generate-image-embeddings`
using the policy's SigLIP model and model path.

Each iteration, use the single public driver
`scripts/route_deft_od_aoi_iteration.py`. Its `prepare` command consumes the
N-1 loose and strict gaps and writes query crops plus
`specs/query_embeddings.yaml`. Submit that spec to
`tao-generate-image-embeddings`, run `finalize-embeddings` to strip ephemeral
node-local crop paths, then use the driver's `commit` command with
the reusable candidate embeddings, frozen pools, and prior committed
ledgers/index. `commit` performs the route artifact gate before marking
`routing_state.json` complete.
On SLURM, use the same driver's `stage-coco` command to materialize source and
clean images from their frozen `source_path` entries onto node-local storage;
no separate staging utility is required.
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

Use the one `prepare_anomalygennext_inputs.py` entry point for host and SLURM
handoffs: `materialize-aoi-plan`, `prepare-inputs`, `stage-embedding`,
`restore-embedding`, `build-knn-and-amp`, `stage-amp`, `restore-amp`,
`finalize-inputs`, and `validate-prepared-inputs`. This replaces run-local path
remapping scripts. GPU embedding, AMP, and generation remain separate tracked
jobs.

Use only the per-pocket counts from the frozen synthetic plan. Generation may
return fewer admitted images after validation; never fill the difference with
unvalidated output. If synthesis is enabled and requested but produces no
valid images, stop the loop.

## 5. Assemble cumulative COCO

Use the immediately previous assembled COCO as the cumulative state, then add
only the current committed route manifest and current admitted synthetic
source. Omit `--previous-assembled-coco` only for iteration 1:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/assemble_deft_od_aoi_coco.py \
  --route-manifest "${CURRENT_MANIFEST}" \
  --current-routing-report "${CURRENT_ROUTING_REPORT}" \
  --previous-assembled-coco "${PREVIOUS_TRAIN_COCO}" \
  --synthetic-source "${CURRENT_SYNTH_COCO}::${CURRENT_SYNTH_IMAGES}" \
  --output-coco "${ITER_DIR}/train/annotations.json" \
  --output-images-dir "${ITER_DIR}/train/images"
```

Do not pass prior synthetic roots separately: they are already present in the
previous assembled COCO. The script rejects publication unless the assembled
real and clean counts exactly match `cumulative_real_defectives` and
`cumulative_clean_negatives` in the current routing report. Gate additionally
on one `defect` category, `retained_previous_images` equal to the previous
COCO's image count, and nondecreasing per-kind counts. Rebuilding from every
committed route/synthetic source remains an equivalent audit path, but the
single previous-state handoff is the normal interface.

## 6. Probe, train, and select

Use `scripts/write_rtdetr_specs.py` to emit nested main and inference specs. On
a staged platform, pass the node-local directory as `--output-dir` and the
durable spec identity as `--published-output-dir`; the inference classmap path
must refer to the durable copied bundle and contain no node-local prefix.
On clusters with known DataLoader shared-memory instability, pass
`--training-workers 0`; the emitted manifest records the effective training and
inference worker counts. See the RT-DETR
recovery guidance in `references/training-policy.md` before resuming a failure.
When `training.probes_enabled` is true, also emit three probe specs from
iteration 3 onward. Each probe and main train is a separate job-record and
starts from the frozen base checkpoint. After probes,
`scripts/select_deft_od_aoi_probe.py` applies the KPI-best deltas and frozen epoch
budget to the main spec. If probes are disabled, skip that script; the main
spec already has frozen learning rates and the size-based epoch budget.

On SLURM, render the packaged four-hour platform default unless the reviewed
site policy says otherwise, and override a reused wrapper's static job name at
submit time so it includes the actual iteration. After epoch 0, compare the
structured ETA with the backend's real time limit plus copy-back margin. Follow
the training-policy recovery procedure if the measured ETA no longer
fits; do not infer the iteration from the wrapper filename or static SBATCH
label.

After the main job, run `scripts/select_deft_od_aoi_checkpoint.py`. When it emits
`action: extend`, set `train.num_epochs` to `extended_num_epochs` and
`train.resume_training_checkpoint_path` to `resume_checkpoint`, then submit one
extension of that same iteration. Rerun selection with `--extension-applied`
and repeat `--status` for the initial phase plus every resume/extension phase;
selection must cover the complete KPI history, not only the last allocation.
Never extend twice. Gate on the selected `model_epoch_*.pth` and replace the
literal `DEFT_OD_AOI_SELECTED_CHECKPOINT` before launch by running packaged
`scripts/prepare_rtdetr_measurement_specs.py`. On a staged platform, give it a
node-local `--output-dir` and the durable spec directory as
`--published-output-dir`; gate the four generated KPI/test inference/evaluation
YAMLs and the scratch-free `measurement_manifest.json` before measurement.

## 7. Measure, gap, and advance

Run inference/evaluation on KPI and test using the four frozen specs and fixed
selected checkpoint.
Record both metrics, then create loose and strict KPI gaps labeled N. Route the
next iteration only from those KPI gaps. Continue through `max_iterations`; a
target metric is report-only unless the user froze a different stopping
contract before launch.

## 8. Consolidate with model soup

After `max_iterations`, and only when `model_soup.enabled` is frozen true,
invoke the separate `tao-model-soup` application. Supply every iteration's
KPI-selected `model_epoch_*.pth` whose train job is `COMPLETE` and whose
`selection.json` passed its artifact gate. Do not include the iteration-0 base
checkpoint, failed or partial jobs, probe checkpoints, non-selected epoch
checkpoints, or checkpoints with a different architecture or binary head.

Use the frozen greedy method, KPI AP50, maximize direction, and strict
zero-minimum improvement gate. The model-soup CLI reevaluates each individual
checkpoint and every proposed equal-weight soup on KPI in one platform job;
do not substitute already observed test metrics. Gate on the final
`model_soup.pth`, its matching SHA-256 in `soup_manifest.json`, and its KPI
score. If no proposal improves the best individual, the valid greedy result is
that single best checkpoint represented in the final output.

Once method and ingredients are frozen, evaluate the final soup once on test
for reporting. Never compare the soup and a non-soup candidate on test and use
that comparison to choose the deployed model. If fewer than two compatible
iteration-selected checkpoints exist, record the stage as skipped instead of
calling the soup CLI.

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
    main/train/
    selection.json
    metrics.json
  model_soup/
    model_soup.pth
    soup_manifest.json
    evaluations/
```
