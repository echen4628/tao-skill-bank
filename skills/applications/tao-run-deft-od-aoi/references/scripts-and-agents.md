# Bundled scripts and stage skills

Run every bundled script through `scripts/deft_python.sh`. Resolve every path
argument to an absolute host path before calling. GPU work uses the selected
platform's `submit` / `status` / `logs` / `cancel` verbs and a job-record, not
`deft_state.json`.

Read only the current stage's overlay and mapped skill, then invoke. Never
preload every reference or leaf skill.

## Bundled scripts

| Script | Stage | Purpose |
|---|---|---|
| `deft_python.sh` | any | Select an already-provisioned host Python. Never installs packages. |
| `inspect_deft_od_aoi_sources.py` | preflight | Inventory one or many dataset paths, COCO contents, metadata hints, sample source paths, and candidate clean directories without mutation. |
| `prepare_deft_od_aoi_sources.py` | preflight | Apply the frozen strict source manifest and emit exact KPI, test, defective, and curated-clean roles. |
| `init_deft_od_aoi_policy.py` | preflight | Freeze `deft_od_aoi_policy.json` once. Refuses to replace different values. |
| `validate_deft_od_aoi_inputs.py` | preflight | Validate the four binary COCO pools and reject identity overlap. |
| `write_gap_specs.py` | gap | Project frozen binary KPI COCO to KITTI, emit both specs, and optionally run deterministic binary matching with `--analyze-binary`, writing both artifact sets in one command. |
| `prepare_deft_od_aoi_siglip_candidates.py` | preflight | Build contextual defect crops and multiscale verified-clean patches for the reusable candidate index. |
| `route_deft_od_aoi_iteration.py` | route | Public routing lifecycle driver. `prepare` writes query crops plus the embedding spec; `finalize-embeddings` converts the tracked embedding result to durable provenance; `commit` routes and validates the complete artifact set. `stage-coco` is the node-local frozen-COCO handoff for platforms such as SLURM. |
| `prepare_deft_od_aoi_siglip_queries.py` | route library | Query-crop implementation used by the public routing driver. |
| `route_deft_od_aoi_siglip.py` | route library | Ranking/admission implementation used by the public routing driver. |
| `route_deft_od_aoi.py` | legacy | Historical pocket-local DCT/uniform router; do not use for the SigLIP-only contract. |
| `assemble_deft_od_aoi_coco.py` | assemble | Cumulative binary COCO from the previous assembly plus current admissions, with router-ledger count gates and explicit clean negatives. |
| `write_rtdetr_specs.py` | train | Nested train, inference, and optional probe specs. |
| `select_deft_od_aoi_probe.py` | train | KPI-best probe deltas and size-based epoch budget. Skip when probes are disabled. |
| `select_deft_od_aoi_checkpoint.py` | train | KPI-best `model_epoch_*.pth` and optional +12 epoch extension. |
| `prepare_rtdetr_extension_spec.py` | train | Build the one policy-approved dynamic-budget +12 resume spec from the terminal checkpoint. |
| `prepare_rtdetr_measurement_specs.py` | measure | Freeze KPI/test inference and evaluation specs plus a manifest for the selected checkpoint; supports separate staged and published output identities. |
| `deft_od_aoi_policy.py` | library | Policy construction, SigLIP-only invariants, validation, and epoch budget. |

## Stage reference modules

Each GPU or data-services stage reuses a bank skill. The DEFT OD AOI overlay adds
thresholds, paths, and admission rules. The leaf skill owns the container
command, spec schema, and pitfalls.

| Stage | DEFT OD AOI overlay | Underlying skill | Owns |
|---|---|---|---|
| Prepare inputs | `references/source-manifest.md`, `references/data-contract.md` | *(bundled)* | Dataset discovery, provenance metadata, curated clean declarations, and strict validation. Glue: `inspect_deft_od_aoi_sources.py`, `prepare_deft_od_aoi_sources.py`. |
| Launch review | `references/preflight.md` | selected platform + `tao-launch-workflow` | Platform preflight, job-record, four verbs. |
| `inference` | `references/pipeline.md` §1, `references/training-policy.md` | `tao-train-rtdetr` action `inference` | KITTI predictions. Threshold 0.001 so both gap passes keep candidates. `automl_policy: off`. |
| Dual gap | `references/gap-routing.md` | bundled binary matcher; optional `tao-analyze-gaps-od-map` | Loose 0.3 FP rows and strict 0.8 FN rows. `write_gap_specs.py --analyze-binary` is self-contained; the leaf action is equivalent when its pinned image is accessible. |
| Candidate/query embedding | `references/gap-routing.md` | `tao-generate-image-embeddings` | One frozen SigLIP encoder for both role-separated candidate indexes and every gap query. `route_deft_od_aoi_iteration.py prepare` writes the query spec; after the tracked leaf job, `finalize-embeddings` records durable paths. |
| Route / admit | `references/gap-routing.md`, `references/data-contract.md` | *(bundled — no miner skill)* | Global role-separated cosine rank, gap-driven doses, audit, and admission. On node-local platforms, `stage-coco` materializes only the frozen source/clean views; `commit` gates the outputs. |
| Synthetic (optional) | `references/pipeline.md` §4 | `tao-prepare-anomalygennext-inputs`, then `tao-generate-image-embeddings`, then `tao-generate-od-defects` | FN testcases, embeddings, generation in the leaf's pinned AnomalyGenNext 1.1 container, and validated binary COCO. The leaf owns the exact Docker invocation; DEFT OD AOI admits only that COCO. |
| Assemble | `references/data-contract.md` | *(bundled)* | Cumulative train COCO. Glue: `assemble_deft_od_aoi_coco.py`. |
| Probe / train / select | `references/training-policy.md` | `tao-train-rtdetr` action `train` | Fresh-base training from the frozen checkpoint. Glue: `write_rtdetr_specs.py`, `select_deft_od_aoi_probe.py`, `select_deft_od_aoi_checkpoint.py`, `prepare_rtdetr_extension_spec.py`. `automl_policy: off`. |
| Selected measurement | `references/pipeline.md` §7, `references/training-policy.md` | `tao-train-rtdetr` actions `inference` and `evaluate` | Freeze four selected-checkpoint specs with `prepare_rtdetr_measurement_specs.py`; KPI selects and routes, test reports only. |
| Final model soup | `references/pipeline.md` §8, `references/training-policy.md` | `tao-model-soup` | Greedy KPI-only averaging of compatible iteration-selected checkpoints into one final RT-DETR checkpoint. |

If an overlay or mapped skill is missing, stop. Do not substitute guessed
`docker run` lines or inline Python for a leaf skill.

## Invocation

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/<script>.py \
  --policy "${RESULTS_DIR}/deft_od_aoi_policy.json" \
  ...
```

After a GPU stage, poll the platform `status` verb until `COMPLETE` or `ERROR`.
Gate on the artifacts named in `references/pipeline.md` before the next stage
reads them.

## Pitfall — AutoML in the spec

Do not write `automl_policy` or a `workflow:` block into an RT-DETR YAML spec.
Pass `automl_policy: off` at the skill/workflow boundary. Direct `rtdetr train`
is already non-AutoML.
