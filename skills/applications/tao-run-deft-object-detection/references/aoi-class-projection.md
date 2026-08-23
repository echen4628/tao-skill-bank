# AOI Class Projection Reference

AOI is a data-policy choice, not a second DEFT workflow. Training still follows
the normal selected detector path. When the operational decision is binary
`clean` versus `defect`, optionally project the detector taxonomy to one class
before initializing the run.

## What “collapse to defect” means

For every annotated box:

- replace its training `category_id` with one target category (default id `1`,
  name `defect`);
- leave box geometry, image records, and unrelated annotation fields unchanged;
- preserve original category id/name and a normalized `defect_type` under the
  annotation's custom `deft` object;
- write the same provenance to a JSONL sidecar, because ordinary COCO consumers
  may ignore or later drop custom fields.

The detector only sees the projected label. It does not care that the input was
formerly multiclass. The projection does matter to checkpoint head compatibility,
inference class order, KPI mapping, and later per-defect diagnostics, so it must be
performed before the class contract is frozen.

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/project_coco_classes.py \
  --input-coco /abs/raw_multiclass.json \
  --output-coco /abs/aoi_binary.json \
  --metadata-jsonl /abs/aoi_annotation_metadata.jsonl \
  --classmap-out /abs/aoi_classmap.txt \
  --kpi-mapping-out /abs/aoi_kpi_mapping.yaml \
  --source-tag '<stable dataset/version id>'
```

Then start the ordinary RT-DETR route with `target_classes=defect`, the projected
COCO as `source_detection_file`, the emitted classmap/KPI mapping, and a
one-class-compatible checkpoint/spec.

The emitted KPI mapping groups `defect` predictions with every original COCO
category name. Therefore KITTI ground truth may retain labels such as `scratch`
and `dent`; the existing KPI evaluator folds them to the one scored class without
discarding their source identity.

## Provenance semantics

`defect_type` is resolved from existing per-annotation metadata first, then from
the original COCO category name. It is diagnostic metadata, not a training label.
The stable `annotation_uid` lets a future KPI extension join matched GT boxes back
to this sidecar. FNs and TPs can then be attributed to a defect type; background
FPs have no ground-truth defect type and must not be invented.

Current KPI output remains binary `defect` mAP/AP50. Per-type KPI attribution and
AOI-specific two-pass FP/FN routing are deliberately deferred; this change does
not alter Arihant's gap, routing, or mining stages.
