---
name: tao-train-yolo
description: Train, resume, and evaluate Ultralytics YOLO object detectors from binary COCO data with KPI-based checkpoint selection and KITTI prediction export. Use when a user explicitly asks for YOLO detector training/evaluation, or when tao-run-deft-od-aoi selects the YOLO backend. Do not use for RT-DETR, generic TAO model training, or YOLO redistribution/licensing advice.
license: Apache-2.0
compatibility: Requires Python with PyYAML, an Ultralytics 8.4.131 container, a selected execution platform, a user-supplied YOLO checkpoint, binary COCO data, and a reviewed Ultralytics/YOLO license posture. Edge-AI training also requires OneLogger callbacks and login material.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
requires_external: true
tags:
- model
- object-detection
- yolo
- deft
---

# Train YOLO

Run YOLO object-detection training and evaluation through a nested YAML contract.
This skill is local-only while the project resolves Ultralytics/YOLO licensing;
do not publish the branch, container, weights, or derived redistribution bundle.

## Required reads

1. Read [spec-contract.md](references/spec-contract.md).
2. Read [checkpoint-selection.md](references/checkpoint-selection.md) for train
   or resume.
3. Read [runtime.md](references/runtime.md) before rendering a platform job.
4. Read [skill_info.yaml](references/skill_info.yaml) and use the declared
   action without changing its execution contract.

## Workflow

1. Resolve `model.checkpoint`, each binary COCO input and its explicit image
   directory, and the immutable `results_dir`. Require exactly one COCO
   category: id `1`, name `defect`.
2. Keep the YAML nested. Never serialize dotted keys.
3. For `train`, decide explicitly between a fresh run and recovery resume.
   Each DEFT iteration is fresh from the same frozen initializer. A recovery
   resumes only the interrupted iteration from its terminal periodic
   checkpoint with optimizer state.
4. Run the selected platform's preflight and the shared `$tao-launch-workflow`
   launch review. On an edge-AI cluster, OneLogger callbacks and enablement are
   a pre-submit gate; do not infer a waiver from old experiment artifacts.
5. Ask for confirmation before pulling an image or submitting a job.
6. Dispatch `train` or `evaluate` through the platform four-verb contract and
   track it in a job-record.
7. Poll the backend. Treat the runner's `status.json` as completion evidence,
   not as a replacement for platform status.

## DEFT invariants

- Select training checkpoints on KPI validation AP50 only.
- Publish both `selected.pt` for inference and `terminal_resume.pt` for recovery.
- Never initialize the next DEFT iteration from the prior selected checkpoint.
- Evaluate KPI before the sealed test. Run test as a separate `evaluate` job
  with `evaluation.report_only: true` after the checkpoint is frozen.
- Export scored KITTI labels from KPI predictions so the existing DEFT gap
  matcher can consume them without detector-specific logic.
- Initial YOLO integration has no learning-rate probes, late-best extension,
  or model soup. Those features require independent YOLO validation.

## Outputs

Training produces `selected.pt`, `terminal_resume.pt`, `last.pt`,
`results.csv`, `selection.json`, `training_manifest.json`, `timings.json`, and
`status.json`. Evaluation produces `predictions.json`,
`ultralytics_metrics.json`, `common_coco_metrics.json`, scored `kitti_labels/`,
`measurement_manifest.json`, `timings.json`, and `status.json`.

Do not copy the node-local staged dataset, caches, plots, or temporary trainer
directory back to durable storage.
