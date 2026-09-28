# DEFT OD AOI data contract

Read this before freezing the application policy or admitting new training
images.

## Normalized roles

The application consumes four normalized COCO roles:

| Role | Boxes | May enter training | Purpose |
|---|---:|---:|---|
| KPI | labeled or empty | never | gap queries and checkpoint selection |
| Test | labeled or empty | never | report-only measurement |
| Real | at least one per image when nonempty | after retrieval and admission | positive mining |
| Clean | exactly zero per image when nonempty | after retrieval and admission | negative mining |

Every COCO document declares exactly one foreground category named `defect`.
Background is implicit. Clean images remain explicit `images` rows with zero
annotations; files absent from the COCO document do not train the detector.
Every bbox must have a nonnegative origin, positive dimensions, and remain
fully within its image dimensions. Boundary overflow is an input-contract
violation; source ground truth is rejected rather than clipped or rewritten.

Each image resolves through `source_path` when present, otherwise
`images_dir/file_name`. Resolved identities must be disjoint across KPI,
test, real, and clean roles. The initializer verifies paths, IDs, boxes,
category names, role-specific annotation counts, and cross-role overlap before
freezing hashes in the policy.

Real and clean roles may start with zero images. Initialization emits a warning
and a typed `UNAVAILABLE` retrieval capability for each empty role, while all
nonempty roles retain the same strict validation. The clean retrieval role is
independent from the AnomalyGenNext synthesis reference pool.

## Optional synthesis metadata

Each eligible KPI annotation/image pair must resolve `dataset_id`,
`texture_id`, `defect_class`, and a real pixel `fn_mask_source`. Metadata
may appear directly on the record or under `deft_od_aoi`. A box never
substitutes for the mask.

`dataset_id` selects a route, and `texture_id+defect_class` must match a
type declared by that route's recipe and defect specification.

## Cumulative admission

`admit_deft_od_aoi_coco.py` recomputes similarity from the frozen embeddings,
deduplicates selected crops to source images, rejects previously admitted
sources, and publishes a new binary COCO. Pass `--previous-coco` from
iteration 2 onward. Clean negatives are capped by
`routing.clean_cumulative_cap_per_real`. When synthesis is enabled, pass both
the generated binary COCO and its image root. Synthetic admission is capped so
that synthetic defects occupy at most
`synthesis.cumulative_fraction_of_total_defects` of the combined real and
synthetic defect pool. Existing records remain unchanged. The legacy
`cumulative_fraction_of_real_defects` key remains readable with its original
synthetic-to-real meaning. Admission also emits `admission_preview.json`. When
overfetched crops
do not contain enough novel parent images for a branch target, it admits the
available parents and records the shortfall instead of failing the iteration.
