---
name: tao-run-deft-od-aoi
description: >-
  Run a binary industrial-inspection DEFT loop with RT-DETR or YOLO: measure on
  frozen KPI/test roles, mine gap-similar real and clean images with SigLIP,
  accumulate admitted COCO data, retrain from one base checkpoint, and select
  by KPI AP50. Use for iterative AOI defect detection, not generic multiclass OD.
license: Apache-2.0
compatibility: Requires the selected RT-DETR or Ultralytics YOLO detector image, Data Services images, CUDA GPUs, and normalized COCO roles.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash Write
tags: [application, workflow, deft, object-detection, aoi, rtdetr, yolo]
---

# TAO DEFT OD AOI

This application is a disk-backed detector loop for one foreground class,
`defect`. Its core is real-data-only; AnomalyGenNext synthesis is an optional
route with separate preparation, generation, and admission gates.

## References

Read only the references needed for the current stage:

- intake and launch: `references/defaults.md`, `references/data-contract.md`,
  `references/source-manifest.md`, and `references/preflight.md`;
- orchestration: `references/pipeline.md` and
  `references/scripts-and-agents.md`;
- gaps and retrieval: `references/gap-routing.md` and
  `references/tao-analyze-gaps-od-map.md`;
- training and selection: `references/training-policy.md`;
- YOLO backend setup and run sequence: `references/yolo-backend.md`;
- optional synthesis: `references/anomalygen-pool.md`.

## Start

Select an installed platform, read its skill, then invoke
`tao-launch-workflow`. The single launch review must include the four normalized
COCO roles, trainable detector base checkpoint, detector backend, maximum iterations, image and
Data Services containers, GPU shape, and expected runtime. After approval,
copy `assets/default_policy.yaml`, fill its required values, and initialize once:

```bash
scripts/init_deft_od_aoi.py \
  --config /workspace/deft_policy.yaml \
  --output-dir /new/results/deft_contract
```

Never reinitialize an existing result. The validator requires disjoint KPI,
test, defective-real, and verified-clean roles; every COCO must declare only
`defect`. KPI and test may mix boxed and boxless images because they never enter
training. Every defective-real image needs at least one box, while clean images
remain explicit zero-annotation COCO entries.

When staging initialization under node-local scratch, pass the final durable
contract directory with `--published-output-dir`; state must never record the
temporary scratch path.

## Loop boundary

The loop composes existing bank actions:

1. Detector-specific inference/evaluation on KPI and test: `tao-train-rtdetr`
   for RT-DETR or `tao-train-yolo` for YOLO.
2. Two `tao-analyze-gaps-od-map` actions: loose confidence for FP routing and
   strict confidence for FN routing.
3. `tao-generate-image-embeddings` with one frozen SigLIP encoder, followed by
   application-owned role-aware routing, deduplication, and admission preview.
4. Application-owned admission and cumulative binary COCO assembly.
5. Direct detector training from the same frozen base checkpoint, then KPI-only
   checkpoint selection. Test remains report-only. YOLO intentionally skips the
   RT-DETR probe, late-extension, and model-soup stages.

All specs are nested YAML dictionaries. Every GPU/Data Services action uses the
selected platform's `submit/status/logs/cancel` contract and a job record.
Stop on missing artifacts, role overlap, empty enabled mining, class drift, or
failed training. Never infer live state from the JSON record alone.

## Retrieval preparation

Build the reusable candidate cache once, then prepare queries after each pair
of loose/strict gap jobs:

```bash
scripts/prepare_deft_od_aoi_retrieval.py candidates \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --published-output-dir "$RESULTS/candidates" \
  --output-dir "$RESULTS/candidates"

scripts/prepare_deft_od_aoi_retrieval.py queries \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --strict-gaps "$ITER/strict/box_gaps.parquet" \
  --loose-gaps "$ITER/loose/box_gaps.parquet" \
  --iteration 1 --candidate-root "$RESULTS/candidates" \
  --published-output-dir "$ITER/retrieval" \
  --output-dir "$ITER/retrieval"
```

Run the emitted embedding spec through `tao-generate-image-embeddings`, then run
`route_deft_od_aoi_siglip.py` with the candidate and query embeddings. Defective
candidates and gap queries use contextual crops; clean candidates use the
frozen grid. Strict FNs and loose near-miss FPs route to real data.
Background-like loose FPs route only to the verified-clean role. The router
writes the routing report, ledgers, cumulative manifest, and a hash-bound
admission preview; validate all of them before committing retrieval.

## Admission and cumulative COCO

After routing completes, publish the previewed cumulative dataset:

```bash
scripts/assemble_deft_od_aoi_coco.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --route-manifest "$ITER/retrieval/route/mined_manifest.json" \
  --current-routing-report "$ITER/retrieval/route/routing_report.json" \
  --previous-assembled-coco "$PREVIOUS/train.json" \
  --synthetic-source "$ITER/synthesis/pseudo_labels/coco_annotations_od_defect.json::$ITER/synthesis/raw/reconstructed_image" \
  --output-coco "$ITER/training_data/train.json" \
  --output-images-dir "$ITER/training_data/images" \
  --link-mode symlink
```

Omit `--previous-assembled-coco` only for iteration 1 and omit
`--synthetic-source` until generation has completed. Routing has already applied the
strict/near/clean controller, source deduplication, valid-box checks, prior
ledgers, and cumulative clean bound. Assembly verifies the frozen manifest,
excludes prior sources, deterministically limits newly generated images to the
frozen cumulative synthetic fraction, and publishes the previewed records.
It retains every prior image and box and emits one binary COCO with explicit
zero-annotation clean images. Use `--link-mode symlink` only when durable source
paths remain available; portable staging should use `copy`.

## Measurement specs

In true-fresh RT-DETR mode, baseline routing uses
`cold_start_all_kpi_gt`: it skips inference with the incompatible warehouse
head and passes empty predictions to packaged gap analysis, making every KPI
ground-truth box an FN. A binary-compatible seed may use
`checkpoint_inference` in `warm_seeded_historical` mode. After
iteration *n*, its selected checkpoint routes iteration *n+1*, while every
probe and main training job still initializes from the public base. Prepare
both inference specs and the two gap-analysis specs together:

```bash
scripts/prepare_deft_od_aoi_measurement.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-role "$CHECKPOINT_ROLE" \
  --kpi-predictions "$MEASURE/kpi/inference/labels" \
  --results-root "$MEASURE" --output-dir "$MEASURE/specs"
```

When the specs are first written under node-local scratch, also pass their
final durable directory with `--published-output-dir`.

For RT-DETR, submit KPI/test inference through `tao-train-rtdetr`. For YOLO,
generate the baseline or post-training evaluate specs with
`write_yolo_specs.py` and submit them through `tao-train-yolo`. Once KPI labels exist,
submit `gap_loose.yaml` and `gap_strict.yaml` through
`tao-analyze-gaps-od-map`. The helper projects the frozen KPI COCO to KITTI,
keeps the two-line inference class map durable, and emits only nested specs.
Test inference is report-only and its output cannot influence routing or
checkpoint selection.

## RT-DETR training specs

Build nested specs from the current cumulative COCO:

```bash
scripts/prepare_deft_od_aoi_training.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" --iteration 1 \
  --train-coco "$ITER/training_data/train.json" \
  --train-images "$ITER/training_data/images" \
  --results-root "$ITER/training" --output-dir "$ITER/train_specs"
```

Invoke the RT-DETR leaf as direct training with `automl_policy: off`; the AOI
application already owns its frozen probe policy. Iterations 1–2 use 36 epochs.
Probe eligibility starts at frozen `training.probe_start_iteration` (default 3),
which may be lowered to 1 or 2. Later iterations use the clipped size-adaptive
budget. When probes are enabled,
the helper emits independent incumbent, data-growth-scaled, and deterministic
jitter specs, all initialized from the same frozen base checkpoint. Run all
three before selecting and materializing `train.yaml`; never initialize the
next iteration from a previous iteration checkpoint.

Select all three probes before main training:

```bash
scripts/select_deft_od_aoi_training.py probes \
  --manifest "$SPECS/training_manifest.json" \
  --main-template "$SPECS/main_template.yaml" \
  --status "$P0/status.json" --status "$P1/status.json" --status "$P2/status.json" \
  --output-dir "$SPECS/selected"
```

After main training, select the KPI-best checkpoint from structured status
rows. Repeat `--status` for resumed phases. If the best epoch is within the
frozen late window, the first call emits one same-iteration `extension.yaml`
from the terminal checkpoint; submit it, then reselect with
`--extension-applied`. The checkpoint carried to measurement is always the
maximum KPI `val_mAP50`, never the latest checkpoint or test result.

## YOLO training and measurement specs

Set `model.backend: yolo` and a YOLO architecture such as `yolo26x` in the
policy before initialization. `write_yolo_specs.py` reads the frozen KPI/test
roles and initializer directly from that policy. Iteration 0 emits only KPI and
report-only test evaluation specs. Later iterations use `--phase train` with
the cumulative `--train-coco` and `--train-images`, then `--phase measure` with
the completed train job's `selected.pt`. Dispatch each YAML through
`tao-train-yolo`; the KPI action's scored KITTI labels feed shared gap analysis.
Every iteration starts from the same initializer, and every action's job record
binds its own `results_dir`.

After each stage succeeds, commit at least one completion artifact with
`commit_deft_od_aoi_stage.py`. It accepts only the next frozen stage, verifies
and hashes every named file, atomically updates `deft_state.json`, and appends
`loop_log.jsonl`. The final iteration becomes `COMPLETE` only after its gap
artifacts are committed. Poll native backends for live state; this record is
durable workflow history, not a scheduler substitute.

## Optional synthesis with existing task weights

Enable `synthesis` only when KPI annotations carry `dataset_id`, `texture_id`,
`defect_class`, and a pixel `fn_mask_source`. Each configured dataset route
must provide an existing AnomalyGenNext checkpoint and matching recipe. After
strict gap analysis, normalize exact FN/annotation matches:

```bash
scripts/prepare_deft_od_aoi_synthesis.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --strict-gaps "$MEASURE/gap_strict/box_gaps.parquet" \
  --synthetic-plan "$ITER/retrieval/synthetic_plan.json" \
  --synthetic-plan-sha256 "$SYNTHETIC_PLAN_SHA256" \
  --output-dir "$ITER/synthesis_request"
```

When preparing under node-local scratch, also pass the final durable directory
with `--published-output-dir` so the emitted filtering config never records a
scratch path.

The SHA must be the hash recorded when `iteration_retrieval` committed that
plan. Preparation deterministically freezes at most `requested_images // 2`
false-negative boxes per anomaly type, tries three clean-image candidates per
box, and retains one successful pair (two mask branches). Pass the emitted
filtering YAML through `tao-prepare-anomalygennext-inputs`, then its finalized
generation plan through `tao-generate-od-defects`. Commit
`iteration_synthesis` only with the plan reconciliation artifact. Re-run
admission with the generated native COCO and image root; synthetic categories
are folded to `defect`, and the frozen cumulative fraction cap is applied
against admitted real defects. Boxes alone never substitute for the required
pixel mask.

## Missing AnomalyGenNext task weights

A synthesis route may replace `checkpoint` and `recipe` with a `finetune` block
containing `dataset_root`, frozen `validation_testcase`, Cosmos3-Nano
`base_checkpoint`, `vae_path`, `nn_backbone`, future `result_handoff`, and
optional user `recipe_template`/`defect_spec`. These are AnomalyGenNext inputs;
the application policy is not an upstream training recipe.

The resolver retains `base_checkpoint` and `vae_path` after task-weight
resolution and writes them into the prepared generation plan. The base is the
distributed DCP generation checkpoint, distinct from the Hugging Face Cosmos
payload selected by `amp_model_id`; the platform must expose the Wan2.2 VAE at
the generator's canonical container path.

Run `resolve_deft_od_aoi_synthesis.py` before the candidate cache. If a handoff
is absent, it emits `finetune_requests.json`; execute each request through
`tao-finetune-anomalygennext` once. Resolve again into a new directory, verify
the hash-bound handoff, commit `synthesis_bootstrap`, and use the emitted
`resolved_synthesis_policy.yaml` for all later synthesis preparation. Never
start or resume AnomalyGenNext training inside a DEFT iteration.
