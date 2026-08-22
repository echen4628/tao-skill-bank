# DEFT OD AOI data contract

## Accepted input layouts

Do not require users to pre-split or rewrite a generic binary object-detection
dataset. Accept either:

1. four normalized roles: KPI, test, defective source, and clean source; or
2. one or many paths containing KPI, test, mixed mining COCO shards, and
   known-clean sources.

Mixed shards may contain both annotated defective images and boxless images.
They may use absolute `source_path` values that point outside the COCO
directory; preparation preserves those paths and does not copy mining images.
Inventory them with `inspect_deft_od_aoi_sources.py`, write the common manifest
defined in `source-manifest.md`, and run
`prepare_deft_od_aoi_sources.py --check-only` before the launch review.

Preparation classifies any mining image with at least one box as defective.
It classifies a boxless image as clean only when the manifest selects its COCO
through `boxless_clean_coco` or explicitly enables `boxless_as_clean`. Raw
known-clean directories are separate `inputs.clean` entries. It remaps all
foreground annotations to the single `defect` category and emits `kpi.json`,
`test.json`, `source.json`, `clean.json`, and
`source_preparation_report.json`.

## Exact metadata rules

Preserve supplied metadata. Otherwise use benchmark-specific named regex rules
over the original `source_path` to derive `texture` and `defect_type`. Use an
explicit default defect type only for datasets whose defect pocket truly has a
single type, and use `generator_type_template` plus optional
`defect_type_renames` to match configured synthesis routes. Keep
`strict_metadata: true`: a missing rule, unmatched path, unresolved texture, or
unresolved defect type is an error.

Real candidate retrieval itself is global within the `defect` or `clean` role;
metadata does not partition the SigLIP index.

If one image has multiple annotation labels, join the sorted labels into one
stable defect type only when no path rule or explicit default resolved the
type. Review `metadata_sources`, texture counts, defect-type counts, and every
source's admitted/ignored counts in the preparation report.

## Binary COCO contract

Every pool declares exactly one category named `defect`. Background is
implicit. Do not create a background category. Each image uses a stable
`source_path` when possible and includes DEFT OD AOI metadata either at the image
top level or under `image.deft_od_aoi`:

```json
{
  "benchmark": "benchmark-id",
  "texture": "texture-id",
  "defect_type": "defect-id",
  "generator_type": "optional-generator-route"
}
```

`benchmark` and `texture` are required everywhere. `defect_type` is required
for images with defect boxes. `generator_type` is optional and defaults to the
joined benchmark/texture/defect pocket name. These fields support audit,
adaptive FN dosing, and optional synthesis; they do not restrict which source
dataset SigLIP may retrieve from.

## Pool roles

| Pool | Annotations | Enters training | Role |
|---|---:|---:|---|
| KPI | Labeled | Never | Gap queries and KPI checkpoint selection |
| Test | Labeled | Never | Final/report-only measurement |
| Real source | At least one defect box per image | After admission | Positive mining |
| Clean source | Exactly zero boxes per image | After admission | Negative mining |
| Synthetic | At least one validated defect box | After admission | FN shortage coverage |

No image identity may occur in more than one of the first four pools. KPI
crops are similarity queries only; neither the crop nor its parent KPI image
may be copied into training.

## Clean negatives and orphans

A clean training image must have an `images` entry in the assembled COCO and
zero corresponding annotation rows. An image file copied beside the dataset
but absent from the COCO `images` array is an orphan; TAO does not train on it.

The canonical real source pool is defective-only and the canonical clean pool
is separate. Original mining input may mix both, but boxless does not
automatically mean verified clean. DEFT OD AOI routes background-like loose
false positives into one global verified-clean SigLIP index. Benchmark and
texture are retained for reporting, not eligibility. The detector still has
one foreground class; clean examples teach background through absence of boxes.

## Gap-analysis inputs

RT-DETR inference produces the KITTI prediction labels consumed by
`tao-analyze-gaps-od-map`. The KPI ground truth used for that stage must be the
matching KITTI projection of the frozen KPI COCO. Preserve image stems and the
single `defect` label across COCO, KITTI, inference class map, and gap output.

## Assembly

For iteration 1, pass the current committed route manifest, current routing
report, and current admitted synthetic COCO to
`scripts/assemble_deft_od_aoi_coco.py`. For every later iteration, also pass
the immediately previous assembled COCO through `--previous-assembled-coco`;
do not reconstruct cumulative state by remembering only prior synthetic roots.
The script deduplicates by resolved source path, assigns category id 1, retains
clean images with zero boxes, and refuses output when real/clean counts differ
from the current routing report's cumulative ledger totals. Use symlinks only
when the selected platform mounts the source paths; otherwise pass
`--link-mode copy`.
