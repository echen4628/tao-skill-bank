# DEFT OD AOI preflight and launch review

Complete this gate before creating a job-record or submitting work.

## Required decisions

- Select an installed execution platform. Read its `SKILL.md` and run its
  Preflight; do not default to a platform.
- Select `deft_od_aoi_reference` or `configurable`. The configurable profile requires
  an explicit uniform-mining value; zero disables the gap-independent top-up.
- Confirm the RT-DETR checkpoint the user already has, if any, the iteration
  count, and whether AnomalyGenNext is enabled. If they have no checkpoint,
  download `nvidia/tao/rtdetr_2d_warehouse:trainable_rn50_v1.0.2` with
  `ngc registry model download-version` and treat the run as a cold start.
- Confirm one KPI set used for selection and a separate test set used only for
  reporting.

## Required data

Resolve absolute paths for:

- KPI and test binary COCO plus image roots;
- a real-defect source COCO plus image root;
- a clean-negative COCO plus image root;
- the RT-DETR base checkpoint (user path or NGC
  `rtdetr_2d_warehouse:trainable_rn50_v1.0.2`) and one-line `defect` class map;
- AnomalyGenNext recipes, checkpoints, masks, and defect specifications when
  synthesis is enabled.

Run the packaged validator before launch:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/validate_deft_od_aoi_inputs.py \
  --kpi-coco "${KPI_COCO}" \
  --kpi-images-dir "${KPI_IMAGES}" \
  --test-coco "${TEST_COCO}" \
  --test-images-dir "${TEST_IMAGES}" \
  --source-coco "${SOURCE_COCO}" \
  --source-images-dir "${SOURCE_IMAGES}" \
  --clean-coco "${CLEAN_COCO}" \
  --clean-images-dir "${CLEAN_IMAGES}" \
  --output "${RESULTS_DIR}/input_validation.json"
```

Read `data-contract.md` for the metadata contract. Stop on any overlap between
KPI, test, real source, and clean source pools.

## Single launch review

Present one review containing:

- platform and native resource request;
- resolved container images;
- all four data-pool identities and validation counts;
- base checkpoint (user path or resolved pretrained), whether iteration 0
  inference can load it, and proof every training iteration reuses that same
  file;
- profile, iteration count, uniform-mining policy, probe enablement, dual gap
  thresholds, IoU thresholds, synthesis enablement, admission caps, and
  training policy;
- the frozen list of generator types when synthesis is enabled;
- KPI selection role and test report-only role;
- result root, estimated GPU jobs, and credential variable names only.

Wait for explicit user confirmation. Then invoke `tao-launch-workflow`, open
each job-record before submission, and operate every backend through
`submit`/`status`/`logs`/`cancel`.

## Hard stops

Stop without auto-repair when validation fails, an enabled producer returns no
admitted rows, a job completes without its declared artifacts, the frozen
policy changes, training produces no checkpoint, or category id/name drift is
detected. A naturally empty gap branch is valid and should emit an empty plan.
