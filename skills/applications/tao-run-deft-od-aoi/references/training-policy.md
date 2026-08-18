# RT-DETR training policy

## Frozen model contract

- RT-DETR with ResNet-50.
- Four GPUs, batch size 8, base learning rate `1e-4`, backbone learning rate
  `1e-5`, validation every epoch.
- Binary COCO category id 1 named `defect`; use `dataset.num_classes: 2`,
  `dataset.eval_class_ids: [1]`, and `remap_mscoco_category: false`.
- Every iteration starts from the same frozen warehouse checkpoint. Never use
  the prior iteration checkpoint as a training initializer.
- Training data is cumulative admitted real positives, explicit clean
  negatives, and admitted synthetic positives.
- Preserve inference scores below both gap gates; the reference inference
  threshold was 0.001 and gap analysis applied 0.3 or 0.8 afterward.

Read `tao-train-rtdetr/SKILL.md` and its `references/skill_info.yaml`. Build a
nested YAML spec. Do not write dotted keys into the YAML; dotted paths are only
valid in the leaf skill's override map before it expands them.

## Iterations 1 and 2

Skip probes and train for 36 epochs. Select the epoch checkpoint with maximum
KPI validation AP50.

## Iterations 3 and later

When `training.probes_enabled` is true (the default), run three independent
ten-epoch probes, each from the frozen base checkpoint:

1. incumbent configuration;
2. data-growth-scaled configuration;
3. deterministic random jitter.

The scaled probe raises LR by 1.15 and extends the LR step by two (maximum 33)
when data growth exceeds 1.25; it lowers LR to 0.85 when growth is below 0.9.
The jitter probe uses seed `4000 + iteration` and the frozen factor ranges in
`deft_od_aoi_policy.json`. Do not replace it with nondeterministic sampling.

Select the probe configuration by KPI validation AP50, then apply those
optimizer deltas to the main spec.

When probes are disabled (`--probes-enabled false`), skip the bake-off and
`scripts/select_deft_od_aoi_probe.py`. Train once from the warehouse checkpoint with
the frozen learning rates. The size-based epoch budget and late-best extension
still apply.

The main epoch budget is:

```text
round(36 * sqrt(10000 / training_images))
```

Clip it to 24–48 epochs. If the selected best epoch is within the last three
epochs, resume that same run for 12 additional epochs and reselect on KPI.
Resume is an extension of the current iteration, not the initialization of the
next iteration.

## Selection and reporting

KPI validation AP50 selects probe configuration when probes ran, plus epoch,
extension, and the checkpoint carried into the next inference stage. Test
metrics are computed only after the checkpoint is fixed and cannot influence
any choice.

The reference experiment's later weight-space search is outside this loop.
Do not describe a post-training soup as an ordinary DEFT OD AOI iteration
checkpoint or use its test metric to select weights.
