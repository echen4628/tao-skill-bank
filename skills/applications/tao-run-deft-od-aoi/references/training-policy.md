# RT-DETR training policy

## Contents

- [Frozen model contract](#frozen-model-contract)
- [Iterations 1 and 2](#iterations-1-and-2)
- [Iterations 3 and later](#iterations-3-and-later)
- [Training recovery and common runtime issues](#training-recovery-and-common-runtime-issues)
- [Selection and reporting](#selection-and-reporting)
- [Final model soup](#final-model-soup)

## Frozen model contract

- RT-DETR with ResNet-50.
- Four GPUs, batch size 8, base learning rate `1e-4`, backbone learning rate
  `1e-5`, validation every epoch.
- Binary COCO category id 1 named `defect`; use `dataset.num_classes: 2`,
  `dataset.eval_class_ids: [1]`, and `remap_mscoco_category: false`.
- Binary inference class maps are index-aligned and contain exactly two lines:
  `background` at index 0 and `defect` at index 1. A one-line `defect` map
  silently drops category-1 detections from TAO's KITTI label export.
- Every iteration starts from the same frozen base checkpoint. Never use
  the prior iteration checkpoint as a training initializer.
- Training data is cumulative admitted real positives, explicit clean
  negatives, and admitted synthetic positives.
- Preserve inference scores below both gap gates with an inference threshold
  of 0.001; gap analysis applies 0.3 or 0.8 afterward.

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
`scripts/select_deft_od_aoi_probe.py`. Train once from the frozen base checkpoint with
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

## Training recovery and common runtime issues

Use this section after the input and nested-spec gates have passed. Preserve
the first failing rank's traceback and the latest complete checkpoint before
changing the spec.

### Triage procedure

1. Poll the backend, then compare the last status timestamp with the latest
   train log and checkpoint timestamps. A live allocation with no advancing
   status is stalled, not healthy.
2. Search every rank log backward from the first error. Classify on the first
   local exception, not a later NCCL watchdog timeout.
3. Verify the latest `model_epoch_*.pth` is nonempty and was closed before the
   failure. Never resume from a partially copied checkpoint.
4. Change only the setting named by the known issue below. Keep the target
   epoch, optimizer settings, data, and output identity fixed.
5. Require `Restored all states from the checkpoint` in the resume log and a
   new status or checkpoint timestamp before declaring recovery.

### DataLoader shared-memory unlink or multi-worker stall

Signatures include `could not unlink the shared memory file /torch_*`, workers
exiting unexpectedly, one rank progressing while peers stop, or a live job
with no new status rows. Treat this as DataLoader IPC cleanup pressure, not a
model, dataset, CUDA, NCCL, or checkpoint-corruption failure.

Confirm that checkpoints and validation metrics have stopped advancing; do not
infer a hang from scheduler state alone. Cancel the hung job, open a new
job-record, and resume the same iteration from the latest complete checkpoint.
Keep the same nested spec except for
`train.resume_training_checkpoint_path: <latest-complete-checkpoint>` and a
reduced worker count. Use `dataset.workers: 1` for a first occurrence. Use
`dataset.workers: 0` after a recurrence or on a cluster/run already known to
exhibit the issue. Prefer
`write_rtdetr_specs.py --training-workers <count>` so the operational override
is recorded in `probe_manifest.json`.

Keep the original `train.num_epochs`. A validation pass during restore is not
a newly completed epoch. Do not restart the iteration from the frozen base
checkpoint or initialize the next iteration from interrupted weights. Keep the
effective worker override for any extension and, after recurrence, for later
iterations of the same run.

### Pillow decompression-bomb rejection on trusted inspection imagery

The first rank reports `PIL.Image.DecompressionBombError`; other ranks may
later report NCCL timeouts. Validate image dimensions and provenance before
bypassing the guard. For trusted, intentionally large inspection images only,
stage a small `sitecustomize.py` in node-local code that sets
`PIL.Image.MAX_IMAGE_PIXELS = None`, put that node-local directory on
`PYTHONPATH`, and assert the value inside the container before launch. Never
disable this protection for untrusted uploads.

### AF_UNIX socket path too long

Signatures include `AF_UNIX path too long`, rendezvous/socket creation failure,
or failure before the first epoch. A long job name nested below a deep
`/raid/scratch/...` path can exceed the kernel socket limit even when the
filesystem accepts the directory.

- Use a short job-specific scratch basename such as `/raid/scratch/d<jobid>`.
- Point `TMPDIR` and runtime caches under that short root.
- Print the resolved `TMPDIR` and its character count before `torchrun`.
- Do not fall back to Lustre for sockets, caches, code, or environments.

### Secondary NCCL timeout

An NCCL watchdog line often appears well after another rank failed. Inspect at
least the preceding 30 minutes and all rank logs. Classify it as infrastructure
only when there is no earlier data, Python, Pillow, socket, CUDA, or filesystem
exception.

### Wall time is shorter than the measured training ETA

This commonly appears after setting `dataset.workers: 0`: the safety override
can materially increase epoch time even though GPU training is healthy. Do not
wait for SLURM to kill the job or estimate from an earlier iteration with a
different assembled dataset size.

- After epoch 0, read `time_per_epoch` and `eta` from `train/status.json`. Add
  staging, final validation, checkpoint serialization, bounded copy-back, and
  at least a 10% margin.
- Compare that bound with the backend's actual `TimeLimit`, not only the
  rendered script. The packaged SLURM default is four hours.
- If permitted, extend the exact running job and verify the new limit with
  `scontrol show job`. `Access/permission denied` means the limit did not
  change.
- When the limit cannot be extended, cancel early only after a complete
  checkpoint exists. Verify the EXIT trap copied it cleanly, open a
  retry-linked job-record, and resume with the same optimizer, data, target
  epoch, and output identity under an adequate reviewed limit.
- Treat a time-limit termination as capacity planning, not a model-program
  error. Preserve the measured epoch timing in the durable timing report.

### Resume and copy-back gate

Execute code, scripts, environments, sockets, and caches from node-local
storage. Keep durable datasets, checkpoints, and results on shared storage only
where needed. Record allocation, staging complete, workload start/end, and
copy-back complete timestamps plus setup/teardown seconds. An EXIT trap must
preserve the workload exit status and copy back only allowlisted results.

## Selection and reporting

KPI validation AP50 selects probe configuration when probes ran, plus epoch,
extension, and the checkpoint carried into the next inference stage. Test
metrics are computed only after the checkpoint is fixed and cannot influence
any choice.

## Final model soup

The frozen default enables one greedy `tao-model-soup` stage after the last
iteration. Its candidate set is the KPI-selected main checkpoint from every
completed training iteration. All candidates share the RT-DETR ResNet-50
binary head and frozen base initialization, but exact state-dict compatibility
is still a hard runtime gate.

Reevaluate individuals and proposed soups on KPI AP50. Seed with the best
individual and accept an additional original checkpoint only when the equal-
weight soup strictly improves KPI (`min_improvement: 0.0`). Test remains
report-only and runs after the final soup manifest freezes the method,
ingredients, weights, and hash. The soup checkpoint is an evaluate/inference/
export artifact, not a resumable average of optimizer or trainer state.
