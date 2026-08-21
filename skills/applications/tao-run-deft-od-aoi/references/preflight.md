# DEFT OD AOI preflight and launch review

Complete this gate before creating a job-record or submitting work.

## Required decisions

- Select an installed execution platform. Read its `SKILL.md` and run its
  Preflight; do not default to a platform.
- Confirm that the run uses SigLIP-only, gap-driven retrieval. Uniform/random
  bootstrap mining is not part of this contract.
- Confirm the RT-DETR base checkpoint the user already has, if any, and the
  iteration count. If they have no RT-DETR checkpoint, resolve the NGC
  fallback in `defaults.md`.
- Before enabling AnomalyGenNext, ask whether the mining-pool defects have
  per-pixel masks. Boxes alone are not enough. If masks are absent, keep
  synthesis off. If they are present, also confirm `defect_spec` and the
  AnomalyGenNext checkpoint required below.
- Confirm one KPI set used for selection and a separate test set used only for
  reporting.
- Surface the default final greedy model-soup stage. Freeze whether it is
  enabled before launch; never decide after seeing test results.

## NGC CLI gate for automatic checkpoint resolution

This gate applies only when the user did not supply a base checkpoint and the
published RT-DETR trainable must be downloaded. Run it on the host that writes
the shared checkpoint destination (for SLURM, normally the cluster login host),
not merely on the agent's local machine:

```bash
command -v ngc >/dev/null 2>&1 || echo "MISSING: ngc CLI"
ngc --version
```

The NGC CLI is a separately installed executable. An `NGC_KEY` that is present
for `nvcr.io`/Pyxis image pulls does not prove that the CLI is installed or that
its model-registry account is configured. If `ngc` is missing, stop before
creating artifacts or submitting jobs and direct the user to NVIDIA's NGC CLI
[installer](https://org.ngc.nvidia.com/setup/installers/cli). After
installation, the user verifies `ngc --version` and runs `ngc config set` in
their own shell; never request or read the API key in chat. Use the official
[NGC Catalog User Guide](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
for CLI configuration and model-registry troubleshooting.

If the CLI cannot be installed or the execution environment is air-gapped,
require a user-supplied trainable checkpoint path instead. Whether downloaded
or supplied, hard-stop until the resolved `.pth` exists on the execution host.

## Required data

Resolve absolute paths for:

- one or many dataset paths containing the KPI, test, mining, and known-clean
  sources; already-normalized four-role COCO is also accepted;
- the RT-DETR base checkpoint and one-line `defect` class map;
- when synthesis is enabled: the mining-pool pixel-mask directory (or an
  equivalent same-type mask pool), `defect_spec.jsonl`, a task-fine-tuned
  AnomalyGenNext checkpoint plus matching recipe for each dataset route, the
  Cosmos3-Nano base checkpoint, and the AnomalyGenNext checkout. Resolve
  absolute paths and stop if any are missing;
- one resolvable SigLIP encoder (`google/siglip-base-patch16-224` by default)
  and one data-services container image used identically for candidate and
  query embeddings.

For path-based intake, inventory every path without changing it:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/inspect_deft_od_aoi_sources.py \
  --dataset-path "${DATASET_PATH_1}" \
  --dataset-path "${DATASET_PATH_2}"
```

Author `dataset_sources.json` from that inventory as specified in
`source-manifest.md`. Keep `strict_metadata: true` and test the complete mapping
without materializing outputs:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/prepare_deft_od_aoi_sources.py \
  --manifest "${DATASET_SOURCES_JSON}" \
  --check-only
```

Show role counts, each mining and clean source, texture/defect counts,
`metadata_sources`, ignored boxless counts, and the overlap result in the
launch review. Strict preparation must have no fallback counts. Do not treat
all boxless images as clean unless the source manifest explicitly authorizes
that source.

After launch approval, materialize the exact canonical inputs:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/prepare_deft_od_aoi_sources.py \
  --manifest "${DATASET_SOURCES_JSON}" \
  --output-dir "${RESULTS_DIR}/normalized_inputs" \
  --link-mode symlink
```

Then run the packaged validator. For normalized outputs, every image has an
absolute `source_path`, so the normalized directory is a valid fallback image
root for the mining roles. KPI/test use their generated views. The preparer
also emits `anomalygen_clean/<benchmark>_<texture>/clean_image/` as a derived
view when synthesis needs the legacy directory convention:

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/validate_deft_od_aoi_inputs.py \
  --kpi-coco "${RESULTS_DIR}/normalized_inputs/kpi.json" \
  --kpi-images-dir "${RESULTS_DIR}/normalized_inputs/kpi_images" \
  --test-coco "${RESULTS_DIR}/normalized_inputs/test.json" \
  --test-images-dir "${RESULTS_DIR}/normalized_inputs/test_images" \
  --source-coco "${RESULTS_DIR}/normalized_inputs/source.json" \
  --source-images-dir "${RESULTS_DIR}/normalized_inputs" \
  --clean-coco "${RESULTS_DIR}/normalized_inputs/clean.json" \
  --clean-images-dir "${RESULTS_DIR}/normalized_inputs" \
  --output "${RESULTS_DIR}/input_validation.json"
```

Read `data-contract.md` for the metadata contract. Stop on any overlap between
KPI, test, real source, and clean source pools.

## Single launch review

Present one review containing:

- platform and native resource request;
- resolved container images;
- source dataset layout, the frozen source manifest, all four canonical role
  counts, clean provenance, exact metadata sources, and identity-overlap result;
- base checkpoint (user path, or the NGC fallback from `defaults.md` when none
  was supplied), whether iteration 0 inference can load it, and proof every
  training iteration reuses that same file;
- iteration count, SigLIP encoder/crop/patch settings, the absence of uniform
  bootstrap, probe enablement, dual gap
  thresholds, IoU thresholds, synthesis enablement, admission caps, and
  training policy;
- final model-soup enablement, greedy method, candidate scope (all compatible
  iteration-selected checkpoints), KPI AP50 direction, and strict improvement
  gate;
- when synthesis is enabled: proof the mining-pool defects have pixel masks,
  the mask-pool path, the frozen generator-type list, `defect_spec` path,
  per-route AnomalyGenNext checkpoint/recipe pairs, Cosmos3-Nano base
  checkpoint, and AnomalyGenNext checkout;
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
