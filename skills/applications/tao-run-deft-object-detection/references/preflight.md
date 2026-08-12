# Pre-Flight Checks and Summary

Resolve everything you can before asking the user. Parameter precedence is strict: **explicit user value → workspace spec → documented default.** Never replace an explicit value with a recommendation. Record the winning value and its source in the Summary.

## Checks, in order

1. **Workspace and run directory.** Resolve to an absolute path (`WORKSPACE=$(realpath -m <workspace>)`); never hand a quoted `~/...` path to Python. Derive `RESULTS_DIR=${WORKSPACE}/results/run_$(date +%Y%m%d_%H%M%S)`. Do **not** create it before the user gate. If resuming, set `RESULTS_DIR` to the existing run directory (detect via `results/run_*/deft_state.json`).

2. **Host Python.** Probe with the bundled launcher:

   ```bash
   <skill_root>/scripts/deft_python.sh -c "import pandas,numpy,matplotlib,pyarrow,PIL,yaml"
   ```

   **All six.** `deft_python.sh` selects an interpreter only when *every* one of
   `pandas numpy matplotlib pyarrow PIL yaml` imports; a host missing just `matplotlib`
   makes it exit 2 with no interpreter at all, and every bundled script becomes unrunnable.
   Probing a shorter list will pass here and then fail at the first real call. If the probe
   fails, provision a venv and re-probe:

   ```bash
   python3 -m venv "$WORKSPACE/.venv" && \
     "$WORKSPACE/.venv/bin/pip" install pandas numpy matplotlib pyarrow pyyaml pillow
   ```

   `deft_python.sh` finds `$WORKSPACE/.venv` only when `WORKSPACE` (or `WORKSPACE_DIR`) is
   exported in the calling shell — otherwise pass the interpreter as `DEFT_PYTHON`.

   `deft_python.sh` auto-selects `$WORKSPACE/.venv/bin/python` once it exists. Pre-Flight is incomplete until the probe exits zero.

3. **Credentials.** Check presence only; never print a value.

   ```bash
   for var in NGC_KEY HF_TOKEN; do
     [ -n "${!var:-}" ] && printf '%s SET\n' "$var" || printf '%s UNSET\n' "$var"
   done
   ```

   | Variable | Required for |
   |---|---|
   | `NGC_KEY` | Always — every `nvcr.io` pull (TAO PyTorch and data-services). |
   | `HF_TOKEN` | **Conditionally** — only when the encoder resolves to a HuggingFace id rather than a local snapshot directory. See check 9. |

   If `NGC_KEY` is unset, tell the user which variable to export and relaunch. Defer the `HF_TOKEN` verdict until check 9 has resolved the encoder; a run using a local snapshot needs no HuggingFace access at all.

4. **Resolve and export the pinned images.** Hard-stop if either export is empty — bash silently substitutes `""` and the next `docker image inspect` then reports a misleading failure.

   ```bash
   VR=<skill_root>/scripts/resolve_versions_key.py
   export TAO_PYT_IMAGE=$(<skill_root>/scripts/deft_python.sh "$VR" images.tao_toolkit.pyt)
   export TAO_DS_IMAGE=$(<skill_root>/scripts/deft_python.sh "$VR" images.tao_toolkit.data_services)
   ```

   | Env var | versions-key | Used by |
   |---|---|---|
   | `TAO_PYT_IMAGE` | `images.tao_toolkit.pyt` | `train`, `inference` |
   | `TAO_DS_IMAGE` | `images.tao_toolkit.data_services` | `gap_analysis`, `embed`, `mine`, `kpi_analyze` |

5. **Image presence.** `docker image inspect "$TAO_PYT_IMAGE" "$TAO_DS_IMAGE"`. Record anything missing as `WILL_PULL_AFTER_APPROVAL`; do not pull before the gate.

6. **Detector and base checkpoint.** Resolve `detector=grounding_dino|rtdetr`
   explicitly and freeze it in state. The baseline scores the base checkpoint
   without training and every iteration fine-tunes from it.

   For Grounding DINO, the user's path wins; otherwise fetch the published checkpoint:

   ```bash
   ZERO_SHOT_CHECKPOINT=$(<skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/fetch_gdino_checkpoint.py --dest "$WORKSPACE/checkpoints/gdino")
   ```

   That resolves `nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_trainable_v1.1`
   (1.93 GB, ~20s) and prints the checkpoint path on stdout. It is idempotent — an existing
   download is reused, so a resumed run re-costs nothing. Use a **`trainable`** release; the
   sibling `deployable` one is for TensorRT export and cannot be fine-tuned.

   This is verified, not assumed: the NGC file was compared tensor-by-tensor against a
   hand-staged copy that had been circulating as `fixed_ckpt.pth`, and all 951 tensors are
   bit-identical with no key differing in either direction. The rename was not a repair.

   Requires the `ngc` CLI and a configured account (check 3). On an air-gapped host, or any
   time the download cannot run, the user supplies `--zero-shot-checkpoint` directly — the
   script says so rather than failing obscurely.

   Record in the Summary which source won (`user` or `NGC <version>`), and hard-stop if the
   resolved path does not exist.

   For RT-DETR, require a user-supplied task-compatible checkpoint; there is no
   open-vocabulary zero-shot fallback. Read `references/rtdetr.md`. The checkpoint
   head, source COCO categories, and train template must describe the same class set.

   For either detector, confirm the spec matches the checkpoint's architecture. `model.backbone`,
   `num_queries`, `enc_layers`, `dec_layers`, `num_feature_levels`, and especially
   detector geometry must agree, or the run dies at load time — after the container has
   started, with an error that reads like a bad checkpoint rather than a bad spec.

   `class_embed_bias` is the one that actually bites: it defaults to `False`, is absent from
   the shipped `infer.yaml` template, and a checkpoint trained with it `True` fails with
   *only* `class_embed.*.bias` keys reported unexpected. Read the values out of the
   checkpoint and set them explicitly — see `references/grounding-dino.md`.

   The checkpoint is coupled to the container, not just the spec. Record which TAO PyTorch
   image the checkpoint was trained with in the Summary; the pinned image is not
   automatically the right one.

7. **Train-spec template.** Must exist and parse as YAML, and
   `dataset.train_data_sources` must be a **list** for both detectors. Entries are
   `{image_dir,json_file,label_map}` for Grounding DINO/ODVG and
   `{image_dir,json_file}` for RT-DETR/COCO.

   **Seed training data is optional.** Inspect the list and branch:

   - **Non-empty** → validate every entry's `image_dir` and `json_file`; also
     require `label_map` for Grounding DINO. Report source and annotation counts.
   - **Empty or absent** → note in the Summary that iteration 1 trains on mined data alone, and that the first iteration's dataset will be small. Not an error.

   Either way the source pool (check 8) stays mandatory — without it there is nothing to mine and the loop cannot add data at all.

8. **Source pool.** The embedding parquet and ODVG tree remain required for the
   shared miner/staging audit. RT-DETR additionally requires the prepared pool COCO:
   - `source_pool_embeddings` parquet — must be non-empty and carry `filepath` and `embedding`.
   - `source_pool_annotations` — an ODVG tree containing `*.jsonl` records keyed by `file_name`, and ideally a `*labelmap.json`. Staging synthesizes a labelmap from observed categories when none is found, but an explicit one is preferred.
   - `source_detection_file` — required for RT-DETR even with global allocation.
     Its COCO categories must exactly match target classes and use dense ids
     `0..N-1` or `1..N`. This file is the RT-DETR class and mined-staging contract.

   Hard-stop if either is missing or the parquet has zero rows. **The loop consumes a
   prepared pool; it does not build one.** Preparing the pool is its own run, completed
   before the loop launches — see `references/prep-source-pool.md`. `init_deft_state.py`
   refuses to write state without these paths, because the alternative is discovering the
   corpus is absent at `mine`, six stages and a training run later.

   Pass `--pool-report` with the prep run's `pool_report.json`. It records which classes the
   pool actually holds annotations for, and init cross-checks that against the target classes.
   A pool prepared for a different class set does not make mining fail — it makes mining
   return neighbours of something else, and the affected class simply never improves.

9. **Resolve the encoder — local snapshot first, never an implicit online default.**

   The encoder that embeds each iteration's weak images must be the *same* one that produced the source-pool parquet. A mismatch is silent: mining succeeds and returns confidently wrong neighbours. Record the resolved values as `config.embedding_model` / `config.embedding_model_path` and reuse them verbatim on every iteration.

   Resolution order — keep an already-set `SIGLIP_MODEL_PATH` only when it is a directory containing `config.json`; otherwise search the known cache roots:

   ```bash
   resolved_siglip=""
   if [ -n "${SIGLIP_MODEL_PATH:-}" ] && [ -f "$SIGLIP_MODEL_PATH/config.json" ]; then
     resolved_siglip="$SIGLIP_MODEL_PATH"
   else
     for cache_root in \
       "${HF_HOME:+$HF_HOME/hub}" \
       "${HUGGINGFACE_HUB_CACHE:-}" \
       "$WORKSPACE/source_pool/hf_cache/hub" \
       "$WORKSPACE/source_pool/hf_cache"; do
       [ -n "$cache_root" ] || continue
       for snapshot in "$cache_root/models--google--siglip-base-patch16-224/snapshots/"*; do
         if [ -f "$snapshot/config.json" ]; then
           resolved_siglip="$snapshot"
           break 2
         fi
       done
     done
   fi
   [ -n "$resolved_siglip" ] && export SIGLIP_MODEL_PATH="$(realpath "$resolved_siglip")"
   ```

   Rules:

   - Select a snapshot **only** from `models--google--siglip-base-patch16-224/snapshots/*` and verify its `config.json`. Never substitute a DINO, C-RADIO, or other model that happens to be cached — it will embed successfully and produce garbage matches.
   - `HF_HOME` may point outside the workspace, so do not limit the search to the workspace tree.
   - Fall back to the bare HuggingFace id `google/siglip-base-patch16-224` **only after** outbound HuggingFace access has been verified, and only then require `HF_TOKEN`. Never let the embed stage pick an online default implicitly.
   - If no local snapshot resolves and outbound access cannot be verified, hard-stop. Report the cache roots searched.

   Record the resolved snapshot path (or the verified HF id) in the Summary. If the user cannot say which encoder produced the source pool, surface that as an explicit risk row rather than assuming SigLIP.

10. **KPI inputs.** Image directory, ground-truth KITTI label directory, and class-mapping YAML must all exist. `image_dir` must not end in `/` — `kpi_analyze` derives its `Sequence Name` from the second-to-last path component.

11. **Class thresholds and mining config.** Per-class AP50 thresholds and the mining `multiplier` both have reference defaults — do not interrogate the user for them. Omitting `--ap50-thresholds-json` gates each target class at the reference ITS value (`car 0.99`, `bicycle 0.7`, `person 0.7`) and any other target class at `0.7`; `--multiplier` defaults to `3`. Surface the defaulted values in the Pre-Flight Summary so the user can override them, and treat a class gated by assumption as worth flagging: too loose a gate marks no image weak and the iteration mines nothing. If rare classes are configured, also require `target_detection_file` as COCO; RT-DETR already requires `source_detection_file` in check 8.

12. **GPU count.**

    ```bash
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --list-gpus | wc -l
    else docker run --rm --gpus all "$TAO_PYT_IMAGE" python3 -c 'import torch; print(torch.cuda.device_count())'
    fi
    ```

13. **Spec sanity.** `train.checkpoint_interval` must be `<= train.num_epochs`. `update_train_spec.py` lowers it automatically when an explicit epoch override would violate this, but flag the adjustment in the Summary so it is not a surprise.

14. **Optional AnomalyGenNext producer.** Disabled unless the user requests it.
    When enabled, read `references/anomalygen-next.md` and resolve every item in
    its preflight section: nested config template, installation/repository,
    Cosmos3-Nano base checkpoint, GPU count, optional provenance `source_tag`,
    and detector target class. This route is inference-only: validate that
    every dataset checkpoint already exists, was fine-tuned for the requested
    anomaly types, and matches its recipe. Require `defect_spec`; reject any
    selected `spatial_dependency: text` entry without a non-empty
    `roi_prompt_defect_location`. Confirm that the target class is in the
    detector target set. The per-iteration `gap_parquet` is the only value
    replaced later.

**Required input — `max_iterations`.** No default. Ask if not supplied and do not proceed past Pre-Flight without it.

## Defaults

- `train.num_epochs` — from the train-spec template
- `train.optim.lr` — from the train-spec template
- `multiplier` — `3`
- `allocation_policy` — `class_stratified` when rare classes are given, else `global`
- `distance_metric` — `euclidean`
- `candidate_expansion_factor` — `5`
- `embedding_model` — `SigLIP`; `embedding_model_path` is **resolved**, not defaulted (check 9)
- `iou_threshold` — `0.5`
- `kpi.conf_threshold` — `0.3`
- workspace root — user prompt, else `~/workspace`

## Pre-Flight Summary

Print this and **STOP — wait for explicit approval.** This is the only user gate.

```
## DEFT OD Loop — Pre-Flight Summary

### Run config
| Field                  | Value                                          | Source            |
| ---------------------- | ---------------------------------------------- | ----------------- |
| Model                  | Grounding DINO (ODVG) / RT-DETR (COCO)        | user              |
| Max iterations         | N                                              | user              |
| Stop condition         | max_iterations reached, or zero weak images.
                          mAP is reported, not gated — no target.         | workflow          |
| Epochs / LR            | N / X                                          | user/spec         |
| Encoder                | <model> @ <resolved snapshot or verified HF id>| resolved          |
| Allocation policy      | class_stratified / global                      | user/default      |
| Rare classes           | <list or none>                                 | user              |
| Mining multiplier      | N (budget = iter1 weak count x N)              | user/default      |
| AP50 thresholds        | {"car": 0.99, ...}                             | user/default      |
| GPUs                   | N                                              | detected          |
| AnomalyGenNext         | disabled / enabled: <target class>, N GPUs     | user/preflight    |
| Resuming               | yes — iter N complete / no                     | disk              |

### Inputs
| Field                     | Value                                        |
| ------------------------- | -------------------------------------------- |
| Base checkpoint           | <path> (NGC/user; target-compatible for RT-DETR) |
| Train spec template       | <path> (N base source(s); 0 = mined-only)    |
| Source pool embeddings    | <path> (N rows, encoder: <model>)            |
| Source pool annotations   | <path> (N jsonl, labelmap: found/synthesized)|
| Source pool COCO          | <path> (required for RT-DETR)                |
| RT-DETR class contract    | disabled / ids, num_classes, classmap order  |
| KPI images                | <path>                                       |
| KPI ground truth          | <path> (N label files)                       |
| Class mapping             | <path>                                       |
| AnomalyGen config         | disabled / <validated template path>         |
| AnomalyGen install/base   | disabled / <repo> / <base checkpoint>        |

### Docker images
| Env var         | Image              | Status     |
| --------------- | ------------------ | ---------- |
| `TAO_PYT_IMAGE` | `<$TAO_PYT_IMAGE>` | OK/MISSING |
| `TAO_DS_IMAGE`  | `<$TAO_DS_IMAGE>`  | OK/MISSING |

### Per-iteration stages
gap_analysis -> embed -> mine -> stage -> train -> inference -> kpi_analyze
(baseline runs inference -> kpi_analyze only; no training)
(stage admits detector-native mined and optional synthetic annotations;
train appends both to the cumulative source list)
```

Remind the user to enable auto-mode (shift+tab) before approving — the post-gate loop is continuously side-effecting.

## Immediately After Approval

Perform the planned pulls and directory creation, then initialize state once:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/init_deft_state.py \
  --workspace "$WORKSPACE" \
  --results-dir "$RESULTS_DIR" \
  --max-iterations "$MAX_ITERATIONS" \
  --detector "$DETECTOR" \
  --num-gpus "$NUM_GPUS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --zero-shot-checkpoint "$ZERO_SHOT_CHECKPOINT" \
  --train-spec-template "$TRAIN_SPEC_TEMPLATE" \
  --source-pool-embeddings "$SOURCE_POOL_EMBEDDINGS" \
  --source-pool-annotations "$SOURCE_POOL_ANNOTATIONS" \
  --source-detection-file "$SOURCE_DETECTION_FILE" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  --kpi-images-dir "$KPI_IMAGES_DIR" \
  --ground-truth-labels-dir "$GROUND_TRUTH_LABELS_DIR" \
  --class-mapping "$CLASS_MAPPING" \
  --ap50-thresholds-json "$AP50_THRESHOLDS_JSON" \
  --multiplier "$MULTIPLIER" \
  --allocation-policy "$ALLOCATION_POLICY" \
  [--anomalygen-config-template "$ANOMALYGEN_CONFIG_TEMPLATE" \
   --anomalygen-repo "$ANOMALYGEN_REPO" \
   --anomalygen-base-checkpoint "$ANOMALYGEN_BASE_CHECKPOINT" \
   --anomalygen-target-class "$ANOMALYGEN_TARGET_CLASS" \
   --anomalygen-num-gpus "$ANOMALYGEN_NUM_GPUS"]

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/audit_deft_run.py \
  --results-dir "$RESULTS_DIR"
```

Then copy the train-spec template to `${RESULTS_DIR}/train_${DETECTOR}.yaml` and begin the baseline using the detector-specific overlay.
