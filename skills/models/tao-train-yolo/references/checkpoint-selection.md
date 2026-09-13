# Checkpoint selection and recovery

Ultralytics writes one row per completed epoch to `results.csv`. Rank rows by
`metrics/mAP50(B)`, maximize, and use the earliest epoch as the deterministic
tie-breaker. The CSV row index is zero-based for `weights/epochN.pt`; the
human-readable `epoch` column may be one-based.

Publish two different identities:

- `selected.pt`: the KPI-best checkpoint used for inference and evaluation.
- `terminal_resume.pt`: the last periodic checkpoint, retaining the trainer
  state required to continue the same interrupted run.

Also publish `last.pt`, the complete curve, hashes, and the selected row in
`selection.json`. A recovery run must merge its new curve with
`train.prior_results_csv` before selecting across the full history. It must not
restart the DEFT iteration or initialize the next iteration from recovered
weights.

Optimizer evidence in `selection.json` is provenance-aware. A single-process
runtime callback records the concrete class. Ultralytics does not propagate
model-local callbacks into its multi-GPU DDP child, so a fresh run with an
explicit non-`auto` optimizer records that deterministic effective name with
`optimizer_evidence_source: explicit_fresh_config` while leaving
`optimizer_observed_class` as `unknown`. Consumers must gate the effective name
and evidence source, not require a runtime callback for DDP. Auto-selected and
resumed optimizers remain `unknown` unless the runtime callback observes them.

Require a contiguous one-based epoch sequence. When a restored trainer repeats
rows already present in the durable prior curve, keep the prior rows as the
authoritative metrics for those completed epochs; new rows begin after the
prior terminal epoch.

If the terminal periodic checkpoint is missing, fail publication. Do not call
`last.pt` an optimizer-safe periodic checkpoint without verifying it.
