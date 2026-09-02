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

Read `tao-skill-bank:tao-prepare-od-coco`, write its `project` config under
the planned run directory, and invoke the Data Services action after the user
approves Pre-Flight but before `init_deft_state.py`:

```yaml
data:
  annotation_file: /abs/raw_multiclass.json
projection:
  target_id: 1
  target_name: defect
  source_tag: <stable dataset/version id>
output:
  annotation_filename: aoi_binary.json
  metadata_filename: aoi_annotation_metadata.jsonl
  classmap_filename: aoi_classmap.txt
  kpi_mapping_filename: aoi_kpi_mapping.yaml
results_dir: <RESULTS_DIR>/intake
```

Run `annotations project -e <spec>` through the selected platform's four-verb
contract. Do not fall back to the former bundled host script; Data Services is
the owner of this transformation.

Then initialize the ordinary RT-DETR route with `target_classes=defect`,
`<RESULTS_DIR>/intake/aoi_binary.json` as `source_detection_file`, the emitted
classmap/KPI mapping, and a one-class-compatible checkpoint/spec.

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
