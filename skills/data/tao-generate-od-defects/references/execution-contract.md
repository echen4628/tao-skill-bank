# AnomalyGenNext OD generation contract

Read this before adapting the generator wrapper to a new AnomalyGenNext release
or execution platform.

## Native commands

For each frozen dataset group the wrapper invokes, in order:

1. `anomalygen/scripts/texture/generate.py`
2. `anomalygen/scripts/texture/evaluate.py`
3. `anomalygen/scripts/texture/quality_refine.py select`
4. `anomalygen/scripts/texture/evaluate.py` on the selected output
5. `anomalygen/scripts/texture/pseudo_label.py`

Generation uses `torchrun --nproc_per_node=N`. The frozen testcase ordering is
shared by all ranks; AnomalyGenNext owns distributed row partitioning and rank-0
metadata merging.

## Inputs and routing

`phase1/phase2_plan.json` is the routing authority. Each row supplies the
dataset id, testcase and provenance paths, checkpoint, matched recipe, real-data
root, anomaly types, and requested row count. Dataset subsets filter these rows
without editing the plan.

## Accounting

The raw generation CSV counts successful images. `guardrail_blocked.csv` counts
blocked rows. Their sum must equal the frozen requested count. The selected
pseudo-label COCO must contain exactly one image record per successful generated
image, and every image must have at least one valid annotation.

The merged native COCO retains the exact fine-grained category names. The
binary companion rewrites every annotation to category id 1 named `defect`.
