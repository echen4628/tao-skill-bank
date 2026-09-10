# Local-only YOLO backend for DEFT OD AOI

This runbook demonstrates how the binary DEFT OD AOI application can use YOLO
instead of RT-DETR. It is intentionally kept on a local branch until the
Ultralytics/YOLO license and redistribution plan are approved.

## What changes

The dataset preparation, SigLIP routing, AnomalyGenNext option, cumulative COCO
assembly, dual gap thresholds, and admission logic do not change. The frozen
`model.backend` selects only the detector leaf:

- RT-DETR: existing TAO train/inference/evaluate path, probes, optional +12
  extension, and final model soup.
- YOLO: `tao-train-yolo` train plus separate KPI/test evaluate actions. The
  first validated profile disables probes, extension, and model soup.

YOLO KPI evaluation exports scored KITTI labels, so both detector backends feed
the same loose-0.3 and strict-0.8 gap analysis.

## Inputs to freeze

- Selected execution platform and resource request.
- Binary COCO KPI, sealed test, defect-source, and known-clean roles.
- A user-supplied YOLO initializer compatible with the chosen architecture.
- Iteration count and synthesis decision. Synthesis retains all existing
  AnomalyGenNext mask, recipe, checkpoint/bootstrap, and type gates.
- YOLO image/container version and reviewed license status.
- On edge-AI: OneLogger login presence, callback adapter module, and enabled
  job configuration.
- Durable result root and node-local runtime root.

Do not download an initializer, pull an image, submit a job, or push the branch
before the corresponding user/launch approval.

## Freeze the backend policy

Copy the current application policy template, select the backend, and fill the
same four normalized roles used by RT-DETR:

```yaml
model:
  backend: yolo
  architecture: yolo26x
max_iterations: 3
base_checkpoint: /durable/checkpoints/yolo26x.pt
synthesis:
  enabled: false
```

Then initialize through the refactored application entry point:

```bash
skills/applications/tao-run-deft-od-aoi/scripts/init_deft_od_aoi.py \
  --config /workspace/deft_policy.yaml \
  --output-dir /durable/run/deft_contract
```

The nested `yolo` policy section freezes the customer-comparison profile:
100 epochs, patience 20, image size 640, batch/nbs 32, four devices, optimizer
`auto`, AMP, deterministic seed 0, per-epoch validation, and per-epoch periodic
checkpoints. Explicit reviewed user overrides should be recorded as such; these
are not general recommendations for arbitrary YOLO datasets.

## Generate the leaf specs

For iteration 0, call the same helper with `--phase baseline --iteration 0` and omit
`--train-coco` and `--train-images`. It writes only KPI/test evaluation specs against the supplied
initializer. Run them separately, use KPI KITTI labels for the baseline gaps,
and keep test report-only.

After cumulative COCO assembly for iteration N:

```bash
.venv/deft/bin/python \
  skills/applications/tao-run-deft-od-aoi/scripts/write_yolo_specs.py \
  --policy /durable/run/deft_contract/deft_od_aoi_policy.yaml \
  --phase train \
  --iteration 1 \
  --train-coco /durable/run/iteration_1/assembly/train.json \
  --train-images /durable/run/iteration_1/assembly/images \
  --output-dir /node/local/spec-staging \
  --published-output-dir /durable/run/iteration_1/specs \
  --runtime-root '${TAO_RUNTIME_ROOT}' \
  --onelogger-enabled true \
  --onelogger-callback-module site_onelogger_adapter
```

Copy each phase's YAMLs plus `yolo_spec_manifest.json` to the exact published
directory. After training completes, invoke the helper again with
`--phase measure --selected-checkpoint <train-job-results>/selected.pt`; it
emits the KPI/test specs. Each action keeps `results_dir: "{results_dir}"` so
the job record remains authoritative. Reject every durable manifest containing
a resolved node-local path.

## Dispatch order

Each action receives its own job-record and immutable result directory:

1. Submit `train.yaml` as `tao-train-yolo` action `train`.
2. Gate `selected.pt`, `terminal_resume.pt`, `results.csv`, `selection.json`,
   hashes, timing, and `status.json: COMPLETE`.
3. Submit `kpi_evaluate.yaml` as action `evaluate`.
4. Gate common COCO metrics, native metrics, `predictions.json`, and
   `kitti_labels/`; run both shared gap jobs from those KPI labels.
5. Submit `test_evaluate.yaml` separately. It is report-only and cannot change
   checkpoint selection, routing, stopping, or configuration.
6. Continue the DEFT loop. Start the next train action from the same frozen
   initializer, not the previous iteration's `selected.pt`.

Use the selected platform skill's `submit`, `status`, `logs`, and `cancel`
verbs. The platform backend is the source of live status.

## SLURM requirements

Set `TAO_RUNTIME_ROOT` to a short job-specific directory below
`/raid/scratch`. Stage custom code and the hot dataset there; execute no custom
code, environment, cache, socket, or database from Lustre. Put all framework
caches under the runtime root. The batch wrapper must record allocation,
staging-complete, workload-start/end, and copy-back-complete timestamps and use
an `EXIT` trap that preserves the workload exit status while copying back only
the declared artifacts.

## Recovery

Resume only an interrupted iteration. Set `train.resume: true`, use that run's
`terminal_resume.pt` as `model.checkpoint`, and add:

```yaml
train:
  prior_results_csv: /durable/run/iteration_1/train/results.csv
  prior_best_checkpoint: /durable/run/iteration_1/train/selected.pt
```

Publish to a new retry-linked job-record/results directory. The runner merges
curves by reported epoch and selects KPI AP50 across both segments. Never use a
stripped KPI-best checkpoint as the recovery checkpoint.

## Customer evidence package

Retain the frozen policy/spec manifest, exact image identity, initializer and
selected-checkpoint hashes, complete training curve, KPI/test common COCO
metrics, KPI KITTI predictions, timing summary, job records, and the statement
that test was report-only. Compare YOLO and RT-DETR on the same frozen data
roles and KPI/test definitions; do not tune either backend from test results.
