---
name: tao-run-deft-od-aoi
description: >
  Run the DEFT OD AOI iterative data-selection workflow for binary
  NVIDIA TAO RT-DETR object detection: generic mixed or sharded COCO intake,
  a frozen base checkpoint, loose and strict box-gap analysis, global
  role-separated SigLIP retrieval for FN/FP-driven mining,
  optional AnomalyGenNext synthesis, admission control, cumulative COCO
  assembly, adaptive training, and KPI-only checkpoint selection. Use for
  "run DEFT OD AOI", "dual-threshold DEFT OD", or a
  binary RT-DETR loop that explicitly routes false positives and false
  negatives. Do not use for the canonical Grounding-DINO whole-image DEFT
  workflow, generic object detection, or one-off training.
license: Apache-2.0
compatibility: Requires the TAO skill bank, Python with pandas, pyarrow, numpy, Pillow, and PyYAML, an RT-DETR TAO image, a selected execution platform, labeled binary COCO pools, and an optional configured AnomalyGenNext producer.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- application
- workflow
- deft
- deft-od-aoi
- object-detection
- rtdetr
---

# TAO DEFT OD AOI

Run DEFT OD AOI as a separate application contract. Never reinterpret the
canonical `tao-run-deft-object-detection` defaults as DEFT OD AOI.

## Execution contract

1. Ask which installed platform to use. Do not default among Docker, SLURM,
   Kubernetes, Brev, virtualenv, or an external platform.
2. Read the selected platform skill and run its Preflight. Read
   `references/defaults.md` and `references/preflight.md`, resolve required
   inputs without re-asking for documented defaults, inspect and prepare
   generic or mixed dataset sources when necessary, validate every DEFT OD AOI input, and show
   one launch review through `tao-launch-workflow` before any submission.
3. Freeze policy once with `scripts/init_deft_od_aoi_policy.py`. Store the emitted
   JSON in the result directory and never edit it during a run.
4. Submit GPU work through the selected platform's `submit` / `status` /
   `logs` / `cancel` verbs. Open the job-record before launch and poll the
   backend rather than treating the record as live state.
5. Keep KPI and test roles separate. KPI selects probe configurations, epochs,
   and checkpoints. Test is report-only and must not affect routing or model
   selection.
6. Stop on a missing artifact, policy mismatch, class-contract drift, empty
   enabled producer, or failed training. Never fabricate an artifact or reuse
   a checkpoint from a failed stage.

Never place secrets in specs or commands. Check only whether required
environment variables are set; do not read credential values or files.

Read `references/defaults.md` before asking for inputs or constructing a launch
review. Read `references/gap-routing.md` before changing any confidence, IoU,
dose, or cap.

Accept either four already-normalized COCO pools (kpi.json, test.json, source.json, clean.json) or one or many dataset paths.
Do not ask the user to manually merge shards, separate defective and boxless
images, or add routing fields image by image. For path-based intake, run
`scripts/inspect_deft_od_aoi_sources.py`, author the common strict source
manifest in `references/source-manifest.md`, then run
`scripts/prepare_deft_od_aoi_sources.py`.
When the user wants AnomalyGenNext, ask whether the mining-pool defects have
per-pixel masks. That is a prerequisite; boxes are not enough. If masks exist,
also resolve `defect_spec` and the matching fine-tuned checkpoint before the
launch review. See `references/defaults.md`.

## Quick Start

After the user approves the launch review, freeze the policy:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/init_deft_od_aoi_policy.py \
  --max-iterations 10 \
  --synthetic-enabled true \
  --output "${RESULTS_DIR}/deft_od_aoi_policy.json"
```

Pass `--probes-enabled false` to skip the iteration-3+ LR bake-off. Omit the
flag to keep probes on.

## Workflow

Read `references/pipeline.md` for commands, artifacts, and stage gates.
Read `references/scripts-and-agents.md` before invoking a leaf skill: each
GPU or data-services stage reuses an existing bank skill; this application
only adds DEFT OD AOI overlays and bundled glue.

1. Inspect, prepare, and validate the four disjoint pool roles without copying
   KPI or test pixels into training. Build one reusable SigLIP candidate cache:
   contextual GT crops for the defect role and multiscale patches for the
   explicitly clean role.
2. Establish iteration 0 by running base-checkpoint inference on KPI and
   test, then loose and strict KPI gaps.
3. For training iteration N, crop and embed strict FNs and loose FPs from
   iteration N-1, then route them with
   `scripts/route_deft_od_aoi_siglip.py`. Candidate eligibility is global
   within the requested role; dataset and texture labels do not gate search.
4. Generate and validate the requested synthetic dose when enabled.
5. Assemble one cumulative binary COCO containing admitted real positives,
   clean negatives, and admitted synthetic positives.
6. Train every iteration from the same frozen base checkpoint. Use the frozen
   training policy and select the checkpoint using KPI validation AP50 only.
7. Infer the selected checkpoint on KPI and test, report both, create loose and
   strict KPI gaps for iteration N, then repeat. Do not stop early on a metric
   target.

## Fixed semantic boundaries

- One class: `defect`. Background is implicit, not a trainable class.
- Matching is same-class greedy one-to-one IoU matching with predictions in
  descending confidence order.
- The loose pass supplies FP routes. The strict pass supplies FN routes.
- KPI crops are retrieval queries only; KPI pixels never enter training.
- Uniform or random bootstrap mining is disabled. Every selected source image
  must be reached by a gap query through SigLIP similarity.
- A clean negative must be explicitly admitted as a COCO image with zero
  annotations. A copied image absent from annotations is an orphan, not a
  clean training example.
- Every iteration starts from the frozen base checkpoint; accumulated data,
  not prior iteration weights, carries learning forward.

## References

- `references/preflight.md` — required inputs and the single launch review.
- `references/defaults.md` — required decisions, packaged defaults, and sources.
- `references/data-contract.md` — binary COCO metadata, pools, and exclusions.
- `references/source-manifest.md` — one/many-path intake and exact metadata rules.
- `references/gap-routing.md` — exact FP/FN matching, routing, dosing, and caps.
- `references/pipeline.md` — stage commands, output layout, and hard gates.
- `references/training-policy.md` — probes, epoch budgets, checkpoint choice.
- `references/scripts-and-agents.md` — bundled scripts and stage→skill map.

For the original Grounding-DINO, whole-image SigLIP, mining-only workflow,
invoke `tao-run-deft-object-detection` instead.
