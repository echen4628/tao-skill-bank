# DEFT OD AOI pipeline

Read this for stage order and completion gates. Read only the stage-specific
reference and leaf skill needed for the current step.

## 0. Freeze the contract

Copy and complete `assets/default_policy.yaml`, then run
`init_deft_od_aoi.py` into a new result directory. Gate on
`deft_od_aoi_policy.yaml`, `input_contract.json`, and `deft_state.json`.

When a synthesis route lacks task weights, run
`resolve_deft_od_aoi_synthesis.py` and complete every emitted fine-tuning
request before building the candidate cache.

## 1. Build candidate embeddings

Run `prepare_deft_od_aoi_retrieval.py candidates`. Submit the emitted real
and clean embedding specs through `tao-generate-image-embeddings`. The
completed outputs are immutable and reused by every iteration. Commit the
`candidate_cache` stage.

## 2. Baseline measurement and gaps

Run `prepare_deft_od_aoi_measurement.py` with the frozen base checkpoint.
For RT-DETR, submit KPI and test inference via `tao-train-rtdetr`. For YOLO,
use iteration-0 `write_yolo_specs.py` output and submit separate `evaluate`
actions through `tao-train-yolo`. KPI controls selection; test is report-only.
Submit the loose and strict gap specs via
`tao-analyze-gaps-od-map`. Commit baseline measurement and gap artifacts.

## 3. Per-iteration retrieval

Run `prepare_deft_od_aoi_retrieval.py queries` with the previous strict and
loose gap parquets. Submit each enabled query-embedding action, then each
enabled `tao-mine-od-images` action. Empty roles require no job. Commit
`iteration_retrieval`.

## 4. Admit real and clean data

Run `admit_deft_od_aoi_coco.py`. For iteration 2 and later, pass the prior
iteration's cumulative `train.json`. Gate on the new binary COCO,
`admitted_sources.parquet`, and `admission_report.json`. Commit
`iteration_admission`.

## 5. Optional synthesis

Run `prepare_deft_od_aoi_synthesis.py` against strict FN gaps. Pass the
filtering YAML through `tao-prepare-anomalygennext-inputs`, complete its
embedding and AMP actions, then invoke `tao-generate-od-defects`.

Re-run admission with the generated binary COCO and image root. Commit
`iteration_synthesis`. Never synthesize from a box without its exact mask.

## 6. Train and select

For RT-DETR, run `prepare_deft_od_aoi_training.py`. Iterations 1–2 emit direct
`train.yaml`. Later iterations may emit three probe specs. Submit probes,
then run `select_deft_od_aoi_training.py probes` to materialize the selected
main spec. Submit main training from the frozen base checkpoint.

Run `select_deft_od_aoi_training.py checkpoint` over all status phases. If it
emits `extension.yaml`, submit that same-iteration resume once and reselect
with `--extension-applied`. Commit `iteration_training`.

For YOLO, run `write_yolo_specs.py` with the cumulative COCO and image root,
then submit its fresh-base train spec followed by separate KPI and report-only
test evaluations through `tao-train-yolo`. YOLO skips RT-DETR probes,
extension, and model soup.

## 7. Measure, gap, and advance

Prepare measurement with the KPI-best selected checkpoint, then repeat KPI/test
inference and dual gap analysis. Commit measurement and gaps. Start the next
iteration only from the committed state.

After every application-owned stage, use `commit_deft_od_aoi_stage.py` with
at least one existing completion artifact. The state is durable history; native
platform status remains authoritative while work is live.
