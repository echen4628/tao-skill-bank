# DEFT OD AOI inputs and defaults

Resolve everything possible before asking. Ask one consolidated question only
for missing required inputs. Never ask about a parameter that has a default.
Preserve every explicit user value; a documented default applies only when the
user did not supply that parameter. Show the source of every run parameter
(`user` or `default`) in the launch review, then freeze the approved values
once in `deft_od_aoi_policy.json`.

## Required inputs — no default, ask

- **Platform.** Installed execution platform. Do not default among Docker,
  SLURM, Kubernetes, Brev, virtualenv, or an external platform.
- **`max_iterations`.** Number of training iterations after iteration 0. Ask if
  not supplied and do not proceed past preflight without it. If the user gives
  a time or GPU-hour budget instead, convert it to an estimated iteration count
  and confirm.
- **Four disjoint pools.** KPI, test, real-defect source, and clean-negative
  binary COCO plus image roots. Validate with `scripts/validate_deft_od_aoi_inputs.py`.
- **Base checkpoint, if any.** Ask what RT-DETR checkpoint they already have.
  Do not assume a warehouse file is on disk. Freeze one path as the train
  initializer reused every iteration, plus the one-line `defect` class map.
  Record the source as `user` or `ngc`.
  - a user path whose head already matches binary `defect` → run iteration 0
    inference on it;
  - a user path that is a raw warehouse/NGC checkpoint with a mismatched head
    → cold start: skip invalid baseline inference, seed iteration 1 from all
    KPI GT boxes as FNs, and train from that file;
  - no checkpoint → download the published ResNet-50 warehouse trainable from
    NGC. Use the **trainable** version (not `deployable`; that is
    TensorRT-only):

    ```bash
    ngc registry model download-version \
      nvidia/tao/rtdetr_2d_warehouse:trainable_rn50_v1.0.2 \
      --dest "${WORKSPACE}/checkpoints/rtdetr_warehouse"
    ```

    Requires the `ngc` CLI and a configured account. Reuse an existing `.pth`
    under that dest if Pre-Flight already fetched it. On an air-gapped host,
    the user supplies the file. That warehouse head is not binary `defect`,
    so this path is always a cold start.
- **KPI vs test roles.** One KPI set for selection; a separate test set for
  reporting only.
- **Synthesis enablement.** `true` or `false`. If `true`, also require the
  frozen generator-type list and AnomalyGenNext recipes, checkpoints, masks,
  and defect specifications. Do not enable synthesis when those assets are
  absent.

## Defaults — never ask

These are the packaged algorithm constants. Surface them in the launch review
so the user can override them; do not interrogate field by field.

### Gap matching

- `inference.confidence_threshold` — `0.001` (keep scores below both gap gates)
- `gap.loose_confidence` — `0.3` (FP routing)
- `gap.strict_confidence` — `0.8` (FN routing)
- `gap.match_iou` — `0.5`
- `gap.background_iou_upper` — `0.05` (below this is a background-like FP)
- `gap.near_miss_iou_upper` — `0.5` (from 0.05 inclusive is a near miss)

### Routing and doses

- `routing.uniform_mine` — 12 real images per pocket in iterations 1–2; `0`
  afterward. Pass `--uniform-mine-per-pocket 0` to disable the gap-independent
  top-up. A positive constant overrides the schedule for every iteration.
- `routing.adaptive_conversion_prior` — `0.33` until a pocket has a trackable
  conversion rate
- `routing.real_mine_factor_min` / `max` — `1` / `6` (about `1/conversion`)
- `routing.near_miss_real_factor` — `2` per near-miss FP
- `routing.near_miss_real_cap_per_pocket` — `20` per iteration
- `routing.clean_factor` — `2` per background-like FP
- `routing.clean_cumulative_cap_per_real` — `1.0`

### Synthetic doses (only when synthesis is enabled)

- `synthetic.ratio_per_admitted_real` — `0.5`
- `synthetic.shortfall_fill_multiplier` / `minimum` — `2.0` / `20`
- `synthetic.conversion_freeze_below` — `0.05` with at least 4 trackable boxes
- `synthetic.cumulative_fraction_of_defective` — `0.25`
- `synthetic.per_iteration_request_cap` — `1500`

- `training.probes_enabled` — `true`. Pass `--probes-enabled false` to skip the
  iteration-3+ LR bake-off. The size-based main epoch budget still applies.

### Training epochs — do not ask for a per-iteration epoch count

Epochs are a function of iteration index and training-set size, not a user
knob. Freeze this policy and let `scripts/write_rtdetr_specs.py` compute the
count:

- Iterations 1–2: skip probes; train **36** epochs from the warehouse
  checkpoint; pick the KPI-best epoch.
- Iterations 3 and later, probes enabled: run **3** probes of **10** epochs
  (incumbent, data-growth-scaled, deterministic jitter). Keep the KPI-best
  probe, then train the main run for the budget below.
- Iterations 3 and later, probes disabled: skip the bake-off and train once
  with frozen learning rates for the same budget.

  ```text
  round(36 * sqrt(10000 / training_images))
  ```

  clipped to **24–48** epochs. If that best epoch is among the last **3**,
  extend the same run by **12** epochs and reselect on KPI.

Do not ask “how many epochs?” A user value such as `epoch 20` still wins for
that run and must be recorded as `user`, but the default path above is the
algorithm. Learning rates, GPU shape, and probe jitter ranges stay in
`references/training-policy.md`.

### Model and selection (also default)

- Architecture — RT-DETR ResNet-50, binary `defect`
- Train GPUs / batch — `4` / `8`
- Base LR / backbone LR — `1e-4` / `1e-5`
- Checkpoint selection — KPI validation AP50
- Test — report-only
