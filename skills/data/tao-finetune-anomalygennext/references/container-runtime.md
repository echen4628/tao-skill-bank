# Container runtime

Read this before submitting AnomalyGenNext fine-tuning.

The action uses the public image declared by `skill_info.yaml`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

Use the source tree and Python environment baked into that image at
`/workspace/paidf-anomalygen`. Do not overlay a host checkout or virtualenv.
Mount or stage the dataset, canonical recipe, validation JSONL, Cosmos3-Nano
checkpoint, VAE, complete checkpoint root, and durable results directory.
The Cosmos3-Nano checkpoint must retain its DCP layout: `checkpoint.json`,
`model/.metadata`, and at least one `model/*.distcp` shard.

When the complete checkpoint root is unavailable, create it with the canonical
full installer documented in
[`tao-prepare-anomalygennext-inputs`](../../tao-prepare-anomalygennext-inputs/references/checkpoint-install.md).
The installer runs inside this pinned image and produces the DCP checkpoint and
the fixed companion tree together. Do not substitute a host checkout or an ad
hoc download sequence.

When the companion root is already complete but the Cosmos3-Nano base
checkpoint is available only in Hugging Face format rather than the required
DCP format, run the converter baked into the pinned image. Mount `/models` to
persistent writable storage and run:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o /models/Cosmos3-Nano-dcp \
  --checkpoint-path Cosmos3-Nano
```

The converter runs on CPU, resolves the registered official model, and writes
the DCP checkpoint under the output directory. `--checkpoint-path` may instead
name an absolute, container-visible Hugging Face checkpoint directory. Write
the converted checkpoint to persistent mounted storage; recipe preparation and
training both require that directory as the `base_checkpoint` input.

The upstream trainer resolves the required Qwen tokenizer model assets and
DINOv2 from the image repository's own checkpoint tree. Expose the complete
selected tree read-only at:

```text
/workspace/paidf-anomalygen/checkpoints
```

It must contain the `Qwen/Qwen3-VL-8B-Instruct` assets under `hf/` plus
`facebook/dinov2-large/config.json` and either
`facebook/dinov2-large/model.safetensors` or
`facebook/dinov2-large/pytorch_model.bin`. The `hf/` name is fixed by the
upstream image. Separate Qwen or DINOv2 mounts are insufficient because the
upstream paths are fixed.

The selected platform owns image import, execution storage, temporary runtime
storage, GPU allocation, and result publication. Keep credentials out of recipes,
commands, logs, and job records. Preserve only the canonical recipe, selected
adapter, validation metrics, status, training curves, and
`training_handoff.json` declared by the action.

A 1000-step run is the minimum quality smoke because it includes iteration-zero
and later validation. Shorter runs demonstrate wiring only.
