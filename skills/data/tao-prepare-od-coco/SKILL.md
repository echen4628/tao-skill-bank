---
name: tao-prepare-od-coco
description: >-
  Project a multiclass COCO object-detection dataset to one class, or stage a
  selected image manifest as a compact COCO training source, using NVIDIA TAO
  Data Services. Use for detector taxonomy projection, AOI defect collapse,
  or materializing mined COCO data; do not use for model training or gap analysis.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
---

# Prepare Object-Detection COCO Data

Use TAO Data Services for the two detector-data boundaries below. The source
implementation lives in `tao-data-services`; this skill owns only the execution
contract and launch guidance.

Read `references/skill_info.yaml` before constructing a run. Use nested YAML
objects in the actual config, never dotted keys.

## Actions

| Action | Command | Purpose |
|---|---|---|
| `project` | `annotations project -e <spec>` | Collapse all COCO categories to one detector class while retaining original per-box identity. |
| `stage` | `annotations stage -e <spec>` | Materialize the files selected by a manifest and their matching COCO records. |

Both actions are CPU-only. Dispatch them through the selected platform's
four-verb contract with `gpu_spec_key: null`.

## Image capability gate

The selected Data Services image must advertise both actions. After the user
approves the launch review and before writing a job record, verify once:

```bash
docker run --rm "$TAO_DS_IMAGE" annotations project --help
docker run --rm "$TAO_DS_IMAGE" annotations stage --help
```

If either command is absent, stop. Do not fall back to a host-side reimplementation.
Use an image built from a `tao-data-services` revision that contains the two
actions, then update the pinned image before release.

## Project categories

Start from `assets/project_coco.yaml`. Set:

- `data.annotation_file` to the raw COCO JSON;
- `results_dir` to a new output directory;
- `projection.target_id`, `projection.target_name`, and a stable
  `projection.source_tag` when the defaults are not the desired contract.

The action writes:

- `projected_coco.json` for detector training;
- `annotation_metadata.jsonl` with stable per-box provenance;
- `classmap.txt` in detector class order;
- `kpi_mapping.yaml`, grouping the projected class with the original names.

Projection changes only the training taxonomy. It retains box geometry, image
records, unrelated annotation fields, and original category/defect type under
each annotation's `deft` metadata.

## Stage selected data

Start from `assets/stage_coco.yaml`. Set:

- `data.source_coco` to the prepared source COCO;
- `data.selection_manifest` to parquet, CSV, JSONL, JSON, or a newline list;
- `data.filepath_column` only when neither `filepath` nor `source_filepath`
  names the selected-path column;
- `results_dir` to the stage output directory;
- output filenames and `min_success_rate` required by the caller.

The action copies selected images, reindexes image and annotation IDs, preserves
the frozen category order and custom annotation metadata, and writes the staged
COCO, classmap, and report. Duplicate basenames, zero resolved annotations, or a
success rate below the configured minimum are hard errors.

## Validation

Before handing outputs to training, verify that all four project artifacts or
all four stage artifacts exist and are non-empty. For staged data, also verify
that the report's category IDs/names match the detector contract frozen by the
calling workflow.
