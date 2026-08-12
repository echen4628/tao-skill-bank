# AnomalyGenNext input-preparation contract

Read this when adapting a new OD dataset layout or troubleshooting an eligibility,
mask, retrieval, or frozen-manifest failure.

## Configuration

Use nested YAML. Dataset entries declare how to identify an image path, derive
the AnomalyGenNext texture and defect class, locate the matching mask, and route
the result to a matched inference checkpoint and recipe. The checkpoint must
already be fine-tuned for the specific anomaly types in the recipe; this skill
does not fine-tune AnomalyGenNext.

`defect_spec` is required. It must contain one row for every selected
`TEXTURE+TYPE`. For `spatial_dependency: text`, require a non-empty
`roi_prompt_defect_location`. Treat that field as the authoritative placement
prompt and never synthesize a replacement. The preparation gate rejects a text
entry with a missing or blank prompt.

`selection.mode=per_dataset` requires `split` and `per_dataset` for bounded
tests. `selection.mode=all_eligible` selects every compatible FN across the
listed datasets.

`embedding.model` and `embedding.model_path` are copied into both emitted
embedding specs. Do not modify one spec independently.

`retrieval.metric` must be `cosine`. `candidate_topn` controls the candidates
tested per FN; `max_neighbors_per_fn` controls retained clean pairs after AMP;
`min_similarity` is the acceptance floor.

## FN independence and allowed reuse

The unit of retrieval expansion, AMP, provenance, and generation is the
box-level `fn_id`, not the unique source filepath. Several FNs from one image
may share a cached embedding and the same ordered clean-neighbor search. They
may also retain the same clean image. Clean reuse across FN ids is intentional
and must not be rejected by a global-uniqueness gate.

Every FN still owns a distinct mask pair: its exact bbox-isolated `fn_mask` and
a same-type sampled mask selected independently under that FN's identity.
Same-image FNs must not inherit the representative FN's mask files; select
distinct sampled masks between them when the pool permits. Run AMP separately
for every `(fn_id, clean_image, mask_branch)` tuple, and include `fn_id` in
stable ids and output paths so shared clean images cannot overwrite or collapse
generated rows.

## Dataset layout

The clean and same-type mask pools resolve beneath `pool_dataset_root`:

```text
TEXTURE/clean_image/IMAGE
TEXTURE/mask/DEFECT/MASK
```

Use `split_components`, `defect_class_fixed`, `mask_component_replacements`,
`mask_suffix`, and `mask_extensions` for layouts that do not use the default
`test` and `ground_truth` components.

## Staged actions

`prepare-inputs` emits unique image rows for embedding and a separate
`selected_fn_queries.parquet` that retains every box-level FN. After embedding,
`build-knn-and-amp` joins each unique image embedding back to every query and
emits per-FN AMP requests. `finalize-inputs` retains only pairs whose two mask
branches both produced valid aligned masks and writes the frozen manifest.
Validation must prove that every eligible `fn_id` is either represented by its
own output rows or has an explicit skip reason.

## Frozen artifacts

The manifest hashes all files consumed by generation. Absolute paths are part
of the contract, so move neither the input root nor its referenced source files
after finalization.

Write them beneath `prepared_anomalygennext_inputs/`; do not use numbered phase
directories. A preparation output root is immutable and must not be reused.
