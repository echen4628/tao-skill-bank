# Detector training and selection policy

Read this before preparing, submitting, resuming, or selecting training.

## Backend split

`model.backend: rtdetr` uses the existing adaptive training and selection
contract below. `model.backend: yolo` delegates to `tao-train-yolo` and
`write_yolo_specs.py`. The frozen YOLO comparison profile trains YOLO26X for
100 epochs at image size 640 with patience 20, batch/nbs 32, four devices, and
four data workers. It selects maximum KPI AP50, retains a separate terminal
periodic checkpoint for recovery, and evaluates KPI and sealed test in separate
actions. Test is report-only.

YOLO can optionally run exactly three independent LR probes from the frozen
initializer. The application selects their recipes by KPI AP50 only and starts
main training from the same initializer, never a probe checkpoint. Adaptive
epoch budgets, late-best extension, and model soup remain unsupported.
Every DEFT iteration is fresh from the same frozen initializer.
The runner stages the dataset and hot framework state on node-local storage and
publishes only the declared compact artifacts. YOLO remains local-only pending
license approval, and edge-AI training still requires enabled OneLogger
callbacks.

## Frozen model contract

- RT-DETR with ResNet-50.
- Binary COCO category id 1 named `defect`.
- `dataset.num_classes: 2`, `eval_class_ids: [1]`, and
  `remap_mscoco_category: false`.
- Four GPUs, batch size eight, base LR `1e-4`, backbone LR `1e-5`, and
  validation/checkpointing every epoch by default.
- Every iteration starts from the same frozen base checkpoint.
- KPI `val_mAP50` selects configurations and checkpoints; test never selects.
- `automl_policy: off`; this application owns its probe policy.

## Epoch policy

Iterations 1–2 skip probes and train for 36 epochs.

From iteration 3 onward, when probes are enabled, run three independent
ten-epoch probes from the frozen base:

1. incumbent learning rates;
2. a data-growth-scaled candidate;
3. deterministic jitter using seed `4000 + iteration`.

All three must complete before selection. The main budget is:

```text
round(36 * sqrt(10000 / training_images))
```

clipped to 24–48 epochs. When probes are disabled, skip the bake-off but retain
the size-based main budget.

`prepare_deft_od_aoi_training.py` emits the immutable specs and manifest.
`select_deft_od_aoi_training.py probes` accepts exactly three structured
status files and writes `train.yaml`, the winner, history, and incumbent
optimizer settings.

## Checkpoint selection and extension

Select the maximum finite KPI `val_mAP50`; ties favor the earlier epoch. The
matching `model_epoch_NNN.pth` must exist.

If the best epoch is within the final three epochs, the first selection emits a
single 12-epoch `extension.yaml` that resumes from the terminal checkpoint,
not necessarily the KPI-best checkpoint. Submit it once, then reselect with
`--extension-applied`. Pass every status phase so selection covers the full
history.

An operational resume preserves data, optimizer settings, target epoch, and
output identity. It is not the policy extension and never initializes the next
iteration. Platform-specific recovery belongs to the selected platform skill.
