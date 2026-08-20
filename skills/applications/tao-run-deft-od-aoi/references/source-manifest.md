# Dataset source manifest

Use one manifest to convert one or many binary object-detection datasets into
the four canonical DEFT OD AOI roles. The user supplies dataset paths; the
agent inventories those paths and authors this file. Do not ask the user to
merge COCO shards or manually annotate every image with routing metadata.

## Contract

```json
{
  "schema_version": 1,
  "strict_metadata": true,
  "metadata_rules": {
    "plant_a": {
      "defect_path_regex": "/PlantA/(?P<texture>[^/]+)/test/(?P<defect_type>[^/]+)/",
      "clean_path_regex": "/PlantA/(?P<texture>[^/]+)/(?:train|test)/[^/]+/",
      "generator_type_template": "{benchmark}_{texture}+{defect_type}"
    },
    "texture_set": {
      "defect_path_regex": "/TextureSet/(?:images/)?(?P<texture>[^/]+)/",
      "clean_path_regex": "/TextureSet/images/(?P<texture>[^/]+)/Train/",
      "default_defect_type": "defect",
      "generator_type_template": "{benchmark}_{texture}+{defect_type}"
    }
  },
  "inputs": {
    "kpi": [
      {"coco": "/data/kpi.json", "images_dir": "/data/kpi/images"}
    ],
    "test": [
      {"coco": "/data/test.json", "images_dir": "/data/test/images"}
    ],
    "mining": [
      {
        "benchmark": "plant_a",
        "coco": ["/pool/plant_a/train.json", "/pool/plant_a/mine.json"],
        "boxless_clean_coco": ["/pool/plant_a/mine.json"]
      }
    ],
    "clean": [
      {
        "benchmark": "texture_set",
        "glob": "/raw/TextureSet/images/*/Train/*.PNG",
        "exclude_if_exists": "/raw/TextureSet/masks/{texture}/Train/{stem}_label.PNG"
      }
    ]
  }
}
```

All paths may be absolute or relative to the manifest. `coco` accepts one path
or a list. A mining entry always contributes annotated images. It contributes
boxless images only when `boxless_as_clean` is `true` or their COCO is listed
in `boxless_clean_coco`. This reproduces a curated-clean policy such as using
only mine-bucket clean images while retaining both train and mine defects.

A clean entry accepts a zero-annotation `coco` (plus optional `images_dir`),
`glob`, or `paths`. `exclude_if_exists` supports `{benchmark}`, `{texture}`,
`{stem}`, and `{name}` and excludes an image when the rendered path exists.
This supports datasets where a missing mask denotes a clean image.

## Common source patterns

Prefer adapting paths in the manifest over restructuring or copying the raw
dataset. These layouts cover common industrial anomaly-detection sources:

| Source layout | Defect pocket | Known-clean declaration |
|---|---|---|
| `<root>/<texture>/test/<defect>/...` | texture + defect | selected boxless mine COCO |
| `<root>/<texture>/test/<defect>/...` | texture + defect | the source's labeled normal split, or `train/<texture>/good` |
| `<root>/images/<texture>/test/<defect>/...` | texture + defect | raw `train/ok` or `train/good` |
| `<root>/<class>/Train/...` with masks in a sibling tree | class + configured default | image is clean only when its corresponding mask does not exist |

The paths above are examples, not hardcoded behavior. Inspect actual
`source_path` samples and write named captures matching the installed layout.
For a new dataset, the minimum rule is one stable texture capture and either a
defect-type capture or a truthful constant `default_defect_type`.

Do not create a background class. A clean image is represented by a COCO image
record with zero annotations. Keep the source's clean proof in the manifest so
the resulting report can distinguish selected boxless records, explicit normal
directories, and mask-absence rules.

## Metadata rules

Rules are selected by each image's existing `benchmark`, then by the source
entry's `benchmark`, COCO `info.bench`, or the shard directory. Regexes must use
the named group `texture`; `defect_type` is optional when
`default_defect_type` is supplied.

Resolution order is:

1. existing image `deft_od_aoi` metadata;
2. named regex captures from `source_path`;
3. `default_defect_type` for a defective image;
4. annotation labels only when neither regex nor a default resolved the type.

Use `defect_type_renames` for canonical aliases and
`generator_type_template` to map an FN query onto configured synthesis types.
The template fields are `{benchmark}`, `{texture}`, and
`{defect_type}`.

The manifest defines data roles and provenance, not SigLIP routing groups.
Candidate search is global inside the defect or verified-clean role, so mixed-
texture datasets need no extra grouping field.

Keep `strict_metadata: true`. Any missing rule, unmatched path, missing
texture, or missing defect type is a hard error. Do not disable strict mode to
guess texture or defect type.

## Agent workflow

1. Run `inspect_deft_od_aoi_sources.py` on every user-supplied path.
2. Assign KPI/test roles from the user's intent. Infer mining shards and clean
   sources from the inventory; ask only when KPI/test identity is ambiguous.
3. Author `dataset_sources.json` with named regex captures. Test the rules with
   `prepare_deft_od_aoi_sources.py --check-only`.
4. Require zero metadata fallbacks in strict mode. Show source counts, clean
   provenance, texture counts, defect-type counts, and overlap results.
5. After launch approval, materialize canonical COCO and KPI/test image views
   into the result directory. Downstream stages consume only that frozen view.

## Recommended migration for an existing sharded setup

Keep the existing train/mine COCO shards and their absolute `source_path`
values. Add one version-controlled `dataset_sources.json` beside the run:

- list all shard paths under `inputs.mining`;
- explicitly identify only the COCO shards whose boxless records are trusted
  clean;
- declare raw normal-image globs for sources where clean images live outside
  COCO, including mask-exclusion rules where necessary;
- encode the old hardcoded path parser as named regex rules and the old
  synthesis naming as `generator_type_template` plus renames.

This removes the need to maintain a second hand-built clean COCO. The preparer
also materializes the derived generator view
`anomalygen_clean/<benchmark>_<texture>/clean_image/` from canonical
`clean.json`; do not make that directory the source of truth.
