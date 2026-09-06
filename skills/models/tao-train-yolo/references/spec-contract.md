# Nested YAML contract

## Train

```yaml
model:
  architecture: yolo26x
  checkpoint: /durable/weights/yolo26x.pt
dataset:
  train_coco: /durable/data/train.json
  train_images: /durable/data/train_images
  eval_coco: /durable/data/kpi.json
  eval_images: /durable/data/kpi_images
train:
  num_gpus: 4
  epochs: 100
  patience: 20
  imgsz: 640
  batch: 32
  nbs: 32
  devices: "0,1,2,3"
  workers: 4
  stage_workers: 16
  optimizer: auto
  lr0: 0.01
  lrf: 0.01
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3.0
  amp: true
  deterministic: true
  seed: 0
  close_mosaic: 10
  resume: false
runtime:
  scratch_root: /raid/scratch/d123/yolo
  allocation_start: 1788600000
logging:
  onelogger_enabled: true
  callback_module: site_onelogger_adapter
results_dir: /durable/runs/iteration_1/train
```

The values above document the first validated YOLO26X customer-comparison
recipe; they are not universal defaults. Freeze reviewed values in the DEFT
policy. For recovery set `train.resume: true`, set `model.checkpoint` to the
same run's `terminal_resume.pt`, and provide `train.prior_results_csv` so
selection covers every completed epoch. Also set `train.prior_best_checkpoint`
to the prior segment's `selected.pt`, so a best epoch before the interruption
can still be republished. Do not use that stripped KPI-best checkpoint for
recovery.

## Evaluate

```yaml
model:
  architecture: yolo26x
  checkpoint: /durable/runs/iteration_1/train/selected.pt
dataset:
  eval_coco: /durable/data/kpi.json
  eval_images: /durable/data/kpi_images
evaluation:
  split_name: kpi
  imgsz: 640
  batch: 64
  device: "0"
  workers: 8
  stage_workers: 16
  confidence: 0.001
  iou: 0.7
  max_det: 300
  report_only: false
runtime:
  scratch_root: /raid/scratch/d124/yolo
results_dir: /durable/runs/iteration_1/kpi
```

Use a different immutable `results_dir` for sealed test, set
`split_name: test`, and require `report_only: true`. Never put KPI and test in
the same action because the test must not affect selection or tuning.

COCO `file_name` values are resolved beneath the corresponding explicit image
directory; an absolute `source_path` on an image row takes precedence. All
paths are strings. `train.devices` may be a device string or a list of
integer device ids. Unknown top-level keys and flat dotted keys are rejected.
