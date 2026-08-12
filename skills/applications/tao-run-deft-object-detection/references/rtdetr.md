# DEFT OD — RT-DETR Train / Inference Overlay

Use this overlay when `deft_state.json::config.detector` is `rtdetr`. Layer it
on `tao-skill-bank:tao-train-rtdetr`; read that model skill before launching.

## Compatibility contract

The mining, KPI, and gap-analysis stages do not change: RT-DETR inference emits
the same per-image KITTI label files those stages consume. The detector boundary
does change:

| Boundary | Grounding DINO | RT-DETR |
|---|---|---|
| Train annotations | ODVG + label map | COCO |
| Inference classes | captions in the spec | newline classmap |
| Class shape | `dataset.max_labels` | `dataset.num_classes` + `eval_class_ids` |
| Checkpoint output | `gdino_model_latest.pth` | `model_epoch_<NNN>.pth` |
| Default train routing | direct | force `automl_policy: off` for the DEFT stage |

The base RT-DETR checkpoint must already be compatible with the target class
head. RT-DETR is not open-vocabulary: do not describe an arbitrary pretrained
checkpoint as zero-shot, and do not auto-fetch the Grounding DINO checkpoint.
The supplied train template must match the checkpoint's backbone,
`num_queries`, `num_select`, feature levels, and EMA policy.

## Frozen COCO class contract

The prepared source pool's `coco.json` is canonical. Its categories must:

- exactly match the run's target classes by name;
- use unique dense ids `0..N-1` or `1..N`;
- keep the same ids in every mined, synthetic, validation, train, and inference
  artifact.

`init_deft_state.py` freezes the sorted ids as `rtdetr_category_ids`, the
corresponding ordered names as `rtdetr_class_names`, sets
`rtdetr_num_classes=max(id)+1`, and writes `rtdetr_classmap.txt` in category-id
order. The audit rejects later classmap drift. For the common custom COCO ids `1..N`, this intentionally makes
`dataset.num_classes=N+1` and `dataset.eval_class_ids=[1,...,N]`; this avoids
the CUDA index failure described in the RT-DETR model skill.

Never run Grounding DINO's zero-based validation rewrite on this route. Use
`make_pool_val_split.py --category-id-policy preserve`.

## Baseline inference

Copy the approved train template so inference inherits the checkpoint-compatible
model geometry, then apply the stable overlay and frozen run values:

```bash
cp "<config.train_spec_template>" "${RESULTS_DIR}/baseline/infer_rtdetr.yaml"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "${RESULTS_DIR}/baseline/infer_rtdetr.yaml" \
  --overlay <skill_root>/assets/overlays/rtdetr_inference.yaml \
  --set inference.checkpoint="<config.zero_shot_checkpoint>" \
  --set inference.num_gpus="<config.num_gpus>" \
  --set 'inference.gpu_ids=<0..num_gpus-1 as a JSON list>' \
  --set results_dir="${RESULTS_DIR}/baseline" \
  --set dataset.infer_data_sources.image_dir='[<config.kpi_images_dir>]' \
  --set dataset.infer_data_sources.classmap="<config.inference_classmap>" \
  --set dataset.num_classes="<config.rtdetr_num_classes>" \
  --set 'dataset.eval_class_ids=<config.rtdetr_category_ids as a JSON list>' \
  --allow-new
```

Replace each angle-bracket token before running the command; for example, frozen
ids `[1, 2, 3]` become `--set 'dataset.eval_class_ids=[1,2,3]'`, and two GPUs
become `--set 'inference.gpu_ids=[0,1]'`. The `gpu_ids` length must equal
`num_gpus`. The checkpoint and copied template must describe the same RT-DETR
geometry.

Run through the selected platform. The direct-container fallback is:

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  rtdetr inference -e "${RESULTS_DIR}/baseline/infer_rtdetr.yaml"
```

`inference.conf_threshold` stays `0.0`; filtering at write time would remove
boxes that KPI and gap analysis need for their own threshold sweeps. Labels land
in `${RESULTS_DIR}/baseline/inference/labels/` in KITTI format.

Seed `${RESULTS_DIR}/train_rtdetr.yaml` from the approved train template.

## Iteration staging

Run the existing ODVG staging for the shared audit/backward-compatible artifact
contract, then stage detector-native COCO from the same mined parquet:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/stage_mined_coco.py \
  --mined-parquet "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
  --source-coco "<config.source_detection_file>" \
  --output-images-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --output-coco "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_coco.json" \
  --output-classmap "${RESULTS_DIR}/iter${N}/tmm/annotations/rtdetr_classmap.txt" \
  --report-json "${RESULTS_DIR}/iter${N}/tmm/coco_staging_report.json"
```

The emitted classmap must byte-match `config.inference_classmap`. A mismatch
means the pool class order drifted after Pre-Flight; hard-stop rather than
relabeling predictions silently.

When AnomalyGenNext is enabled, pass the source-pool COCO to
`stage_anomalygen_coco.py --category-contract-coco`. This preserves the full
category list and assigns the configured synthetic target its frozen id.
RT-DETR consumes `synthetic_train.json` directly; skip COCO-to-ODVG conversion
for this detector.

Commit the normal staging artifacts plus:

```bash
  --staged-coco "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_coco.json" \
  --inference-classmap "${RESULTS_DIR}/iter${N}/tmm/annotations/rtdetr_classmap.txt"
```

## Iteration train

Build an ID-preserving validation split once, then append this iteration's COCO
producer(s):

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/make_pool_val_split.py \
  --coco "<config.source_detection_file>" \
  --category-id-policy preserve \
  --out "${RESULTS_DIR}/val_coco_rtdetr.json"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/update_train_spec.py \
  --detector rtdetr \
  --previous-spec "<template or previous RT-DETR train spec>" \
  --output-spec "${RESULTS_DIR}/iter${N}/train_rtdetr.yaml" \
  --tmm-image-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --tmm-coco-file "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_coco.json" \
  [--synthetic-image-dir "${RESULTS_DIR}/iter${N}/synthetic/training/images" \
   --synthetic-coco-file "${RESULTS_DIR}/iter${N}/synthetic/training/synthetic_train.json"] \
  --val-image-dir "<prepared pool image directory>" \
  --val-json-file "${RESULTS_DIR}/val_coco_rtdetr.json" \
  --pretrained-model-path "<config.zero_shot_checkpoint>" \
  --num-epochs "<config.num_epochs>" \
  --learning-rate "<config.learning_rate>" \
  --num-gpus "<config.num_gpus>"
```

Every iteration resets `train.pretrained_model_path` to the frozen base
checkpoint. Only the accumulated COCO source list grows; do not resume from the
previous iteration.

Invoke `tao-train-rtdetr` with `automl_policy: off`. This override belongs to
the application call, never the YAML. The direct-container fallback is:

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  rtdetr train -e "${RESULTS_DIR}/iter${N}/train_rtdetr.yaml" \
  results_dir="${RESULTS_DIR}/iter${N}"
```

Resolve the exact checkpoint after a successful exit:

```bash
CHECKPOINT=$(<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/resolve_rtdetr_checkpoint.py \
  --train-dir "${RESULTS_DIR}/iter${N}/train" [--ema])
```

Use `--ema` iff `train.enable_ema=true`. A missing non-empty
`model_epoch_<NNN>.pth` (or `-EMA`) is a hard stop.

## Iteration inference and commit

Build the inference spec exactly as for baseline, but set
`inference.checkpoint=$CHECKPOINT` and `results_dir=${RESULTS_DIR}/iter${N}`.
Run `rtdetr inference`, then commit:

```bash
# train
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage train \
  --checkpoint "$CHECKPOINT" \
  --training-spec "${RESULTS_DIR}/iter${N}/train_rtdetr.yaml" \
  --summary "trained RT-DETR iter${N}: <epochs> epochs, <N> COCO sources"

# inference
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "<phase>" --stage inference \
  --inference-labels-dir "${RESULTS_DIR}/<phase>/inference/labels" \
  --summary "RT-DETR inference: <N> KITTI label files"
```
