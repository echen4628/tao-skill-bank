# AnomalyGenNext 1.1 checkpoint installation

The container image declared by `skill_info.yaml` includes
`/workspace/paidf-anomalygen/scripts/download_checkpoints.sh`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

## Run

Set `CKPT_DIR` to the writable destination for the checkpoint tree:

```bash
export CKPT_DIR=/workspace/paidf-anomalygen/checkpoints
bash /workspace/paidf-anomalygen/scripts/download_checkpoints.sh
```

AMP, fine-tuning, and generation use that root at:

```text
/workspace/paidf-anomalygen/checkpoints
```

Some downloaded repositories are gated and require `HF_TOKEN` after their
Hugging Face licenses have been accepted. Pulling the container image may
separately require `NGC_KEY`.

## Installed assets

The upstream script installs the complete 1.1 asset set, including:

- native DCP trees for Cosmos3-Nano and Cosmos3-Edge;
- a direct Transformers-format Cosmos3-Nano model for AMP grounding;
- DINOv2 and C-RADIO KPI backbones;
- the Wan2.2 VAE and SAM2.1 checkpoint;
- Qwen caption-tokenizer and Edge processor cache entries; and
- Cosmos-Guardrail1 and Qwen3Guard assets.

## Nano DCP conversion

If the companion root is already complete and the official Cosmos3-Nano base
checkpoint is available only in Hugging Face format, the image also includes
this converter:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o /models/Cosmos3-Nano-dcp \
  --checkpoint-path Cosmos3-Nano
```

Mount `/models` to persistent writable storage. The registered
`Cosmos3-Nano` name resolves the official checkpoint; an already-staged
Hugging Face checkpoint may instead be supplied as an absolute,
container-visible path. The converter creates only the native DCP tree needed
by fine-tuning; it does not install the VAE, tokenizer cache, DINOv2, C-RADIO,
or other companion assets.

## Installed layout

The installed tree includes:

```text
checkpoints/
  Cosmos3-Nano/checkpoint.json
  Cosmos3-Nano/model/.metadata
  Cosmos3-Nano/model/*.distcp
  Cosmos3-Edge/checkpoint.json
  Cosmos3-Edge/model/.metadata
  Cosmos3-Edge/model/*.distcp
  nvidia/Cosmos3-Nano/config.json
  facebook/dinov2-large/
  nvidia/C-RADIO-V3/model.safetensors
  facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt
  wan2pt2/Wan2.2_VAE.pth
  hf/
```

## Known Cosmos3-Edge checksum issue

In a clean CPU run with the public 1.1 image, the Edge Hugging Face source
download and `convert_model_to_dcp` command both completed. The installer then
reported checksum failures for the two generated files:

```text
Cosmos3-Edge/model/__0_0.distcp
Cosmos3-Edge/model/__0_1.distcp
```

All 11 Hub-sourced weight checksums and all six converted Cosmos3-Nano shards
passed. The expected Edge hashes packaged in the image are:

```text
b2f130a51b619ac8da88343454b0510e0734eb2242132276224c3140123f524e  Cosmos3-Edge/model/__0_0.distcp
695b95240a03c0fd4f6236d2c3e41fae268dc95f0b9e90ec1b1339c924d9ed23  Cosmos3-Edge/model/__0_1.distcp
```

Those values also match an earlier preserved Edge DCP conversion, so the
failure is conversion-output drift rather than a failed Edge source download.
Running the converter separately can retain the newly generated DCP instead of
stopping at the installer's final manifest check, but conversion completion
alone does not establish that the new bytes are load-compatible. Validate the
retained DCP with an Edge model-load or generation smoke before using it.

The current Skill Bank AMP, fine-tuning, and generation contracts use
Cosmos3-Nano. For those workflows, an installer failure is non-blocking when
the only reported mismatches are the two Edge files above and all of the
consumer's Nano and companion inputs pass their own preflight. These consumers
do not load the Edge DCP tree. Missing or mismatched Nano shards, Hub-sourced
weights, or required companion assets remain failures.

The upstream `preflight_env_ckpt.sh` checks both model sizes and therefore
continues to report this Edge-only mismatch. Use the selected Nano consumer's
preflight to validate the subset it actually loads.

## Guardrail limitation

Installing the guardrail repositories does not make guardrail-enabled
generation fully network-independent in AnomalyGenNext 1.1. Guardrail
initialization uses an isolated UV invocation that may fetch Python packages
such as `hf-xet`. The current framework preset can also report
`image_guardrail_enforcing=false`: text screening and face blurring remain
active, but no image-safety model is available to block generated images.

`--no-guardrail` generation does not run that guardrail initialization path.
