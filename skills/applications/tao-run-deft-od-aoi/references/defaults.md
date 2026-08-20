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
- **Dataset inputs.** Accept one or many dataset paths. Resolve a distinct KPI
  role and test role, plus mining sources and known-clean sources. Paths may be
  COCO files, dataset roots, or shard roots such as
  `<root>/<benchmark>/{train,mine}.json`; do not ask the user to merge or split
  them manually. Inventory paths with `inspect_deft_od_aoi_sources.py`, encode
  path-to-texture/defect rules and clean provenance in `dataset_sources.json`,
  prepare with `prepare_deft_od_aoi_sources.py`, then validate the four
  canonical roles. Ask only when KPI/test identity or clean provenance cannot
  be resolved safely from the inventory and user intent.
- **Base checkpoint, if any.** Ask what RT-DETR checkpoint they already have.
  Do not assume a file is on disk. Freeze one path as the train initializer
  reused every iteration, plus the one-line `defect` class map. Record the
  source as `user` or `ngc`.
  - a user path whose head already matches binary `defect` → run iteration 0
    inference on it;
  - a user path whose head does not match binary `defect` → cold start: skip
    invalid baseline inference, seed iteration 1 from all KPI GT boxes as FNs,
    and train from that file;
  - no checkpoint → download the published NGC trainable (not `deployable`;
    that is TensorRT-only) and treat it as the frozen base. This head is not
    binary `defect`, so the path is a cold start:

    ```bash
    ngc registry model download-version \
      nvidia/tao/rtdetr_2d_warehouse:trainable_rn50_v1.0.2 \
      --dest "${WORKSPACE}/checkpoints/rtdetr_base"
    ```

    Requires the separately installed `ngc` CLI and a configured model-registry
    account. Follow the NGC CLI gate in `preflight.md`; `NGC_KEY` availability
    for container pulls is not an installation/configuration check for the CLI.
    Reuse an existing `.pth` under that dest if Pre-Flight already fetched it.
    On an air-gapped host, the user supplies a trainable checkpoint instead.
- **KPI vs test roles.** One KPI set for selection; a separate test set for
  reporting only.
- **Synthesis enablement.** `true` or `false`. Ask this only after the mask
  prerequisite below is answered. If `true`, also require these AnomalyGenNext
  assets and do not enable synthesis when any are absent. This loop never
  fine-tunes AnomalyGenNext.
  - **Pixel defect masks.** A prerequisite, not a default. AnomalyGenNext
    stamps a defect onto a retrieved clean image with automatic mask
    placement. A COCO box is not a mask. Ask whether the user's dataset has
    per-pixel defect masks for the mining-pool defects — typically sibling
    files next to those images (`ground_truth/<defect>/`, `*_mask.png`, or a
    parallel mask tree). Those mining defects are the usual same-type mask
    pool: one mask isolated to the missed FN box, plus other masks of the
    same `TEXTURE+TYPE` sampled as placement templates. If the dataset has
    boxes only, keep synthesis `false`. The derived `anomalygen_clean/` view
    is clean images only and does not replace this pool.
  - frozen generator-type list: the exact `TEXTURE+TYPE` strings the producer
    can emit; routing requests only those types;
  - `defect_spec.jsonl`: one row per selected type. A `spatial_dependency:
    text` row must include a non-empty `roi_prompt_defect_location`. Do not
    invent or repair prompts from the OD false negative;
  - a task-fine-tuned AnomalyGenNext checkpoint and its matching recipe for
    each dataset route. The recipe must declare the same types as
    `defect_spec` and the frozen generator-type list;
  - the Cosmos3-Nano base checkpoint and AnomalyGenNext checkout used by
    `tao-generate-od-defects`.

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

- `retrieval.mode` — `siglip_only`; uniform/random bootstrap is disabled
- `retrieval.model_path` — `google/siglip-base-patch16-224`
- `retrieval.defect_context_scale` — `1.5`
- `retrieval.clean_grids` — `[1, 2]` (whole image plus four quadrants)
- `retrieval.output_size` — `224`
- `retrieval.minimum_similarity` — `-1.0` for the first calibration run;
  inspect the frozen retrieval audit before adopting a stricter cutoff
- `retrieval.candidate_overfetch` — `15`
- `retrieval.audit_top_k_per_query` — `20`

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

- Iterations 1–2: skip probes; train **36** epochs from the frozen base
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
