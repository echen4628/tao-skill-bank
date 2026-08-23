# DEFT OD — RT-DETR Train / Inference Overlay

This is the detector boundary for `config.detector=rtdetr`. Read
`tao-skill-bank:tao-train-rtdetr` first. Gap analysis, SigLIP embedding,
mining, exclusion, KPI evaluation, and the stage order remain the original DEFT
OD loop.

## Contract

| Boundary | RT-DETR value |
|---|---|
| Training annotations | COCO JSON + image directory |
| Category ids | dense `0..N-1` or `1..N`, frozen at init |
| `dataset.num_classes` | `max(category_id) + 1` |
| `dataset.eval_class_ids` | the actual category ids |
| Inference classmap | one class per line, in category-id order |
| Inference output | KITTI label files, directly consumed by existing gap/KPI stages |

The source-pool COCO is canonical for the class contract. `init_deft_state.py`
writes `${RESULTS_DIR}/rtdetr_classmap.txt` and the audit refuses class-order
drift. Do not re-sort names alphabetically after init.

## Checkpoint compatibility

The baseline checkpoint must already have an RT-DETR head and geometry compatible
with the selected spec: backbone, query count, feature levels, and class head.
This workflow does not transplant or resize a classification head. A multiclass
checkpoint cannot be silently reused after projecting the dataset to one `defect`
class; use a one-class-compatible base checkpoint/spec for that run.

Every iteration starts from this same base checkpoint. Dataset sources accumulate;
weights are not chained from the previous iteration. This preserves the original
DEFT comparison semantics.

## Baseline inference

Build an inference spec from the RT-DETR leaf template, then apply
`assets/overlays/rtdetr_inference.yaml` and the run values:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "${RESULTS_DIR}/<phase>/infer_rtdetr.yaml" \
  --apply-workflow-defaults <skill_root>/assets/overlays/rtdetr_inference.yaml \
  --set inference.checkpoint="<checkpoint>" \
  --set dataset.infer_data_sources.image_dir='["<config.kpi_images_dir>"]' \
  --set dataset.infer_data_sources.classmap="<config.inference_classmap>" \
  --set results_dir="${RESULTS_DIR}/<phase>" \
  --set inference.num_gpus="<config.num_gpus>"
```

Keep `inference.conf_threshold=0.0`: gap analysis and KPI evaluation need the
full confidence range. Run RT-DETR inference through the selected platform's
four-verb contract with `automl_policy: off`; do not write that field into YAML.

Commit `${RESULTS_DIR}/<phase>/inference/labels` with the existing
`--inference-labels-dir` flag. No prediction converter is needed.

## Stage the mined COCO source

The original ODVG stage still runs because the unchanged mining pipeline uses
that prepared pool. For RT-DETR, stage a COCO view in the same stage directory:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/stage_mined_coco.py \
  --mined-parquet "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
  --source-coco "<config.source_detection_file>" \
  --output-images-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --output-coco "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_coco.json" \
  --output-classmap "${RESULTS_DIR}/iter${N}/tmm/annotations/rtdetr_classmap.txt" \
  --report-json "${RESULTS_DIR}/iter${N}/tmm/rtdetr_staging_report.json" \
  --min-success-rate 0.9
```

Add these verified artifacts to the normal stage commit:

```bash
--staged-coco "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_coco.json" \
--inference-classmap "${RESULTS_DIR}/iter${N}/tmm/annotations/rtdetr_classmap.txt"
```

## Iteration training

Create a validation split once with category ids preserved:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_val_split_for_train.py \
  --coco "<config.source_detection_file>" \
  --out "${RESULTS_DIR}/val_rtdetr_coco.json" \
  --category-id-policy preserve
```

Then extend the previous RT-DETR spec:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_rtdetr_spec_for_train.py \
  --previous-spec "<template for iter1; previous iteration spec after that>" \
  --output-spec "${RESULTS_DIR}/iter${N}/train_rtdetr.yaml" \
  --tmm-image-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --tmm-coco-file "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_coco.json" \
  --val-image-dir "<source-pool image directory>" \
  --val-json-file "${RESULTS_DIR}/val_rtdetr_coco.json" \
  --pretrained-model-path "<config.zero_shot_checkpoint>" \
  --num-epochs "<config.num_epochs>" \
  --learning-rate "<config.learning_rate>" \
  --num-gpus "<config.num_gpus>"
```

Run `rtdetr train` through the selected platform. Resolve its output with:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/resolve_rtdetr_checkpoint.py \
  --train-dir "${RESULTS_DIR}/iter${N}/train"
```

Use `--ema` only when `train.enable_ema=true`. Commit the resolved checkpoint and
`train_rtdetr.yaml` through the normal `train` stage contract.
