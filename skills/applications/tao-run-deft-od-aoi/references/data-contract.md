# DEFT OD AOI data contract

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
joined benchmark/texture/defect pocket name.

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

The real source pool is defective-only. The clean source pool is separate by
design because DEFT OD AOI routes background-like loose false positives to known
clean textures. The detector still has one foreground class; clean examples
teach background through absence of boxes.

## Gap-analysis inputs

RT-DETR inference produces the KITTI prediction labels consumed by
`tao-analyze-gaps-od-map`. The KPI ground truth used for that stage must be the
matching KITTI projection of the frozen KPI COCO. Preserve image stems and the
single `defect` label across COCO, KITTI, inference class map, and gap output.

## Assembly

Pass every committed iteration manifest, plus every admitted synthetic COCO,
to `scripts/assemble_deft_od_aoi_coco.py`. It deduplicates by resolved source path,
assigns category id 1, retains clean images with zero boxes, and emits one
cumulative training COCO. Use symlinks only when the selected platform mounts
the source paths; otherwise pass `--link-mode copy`.
