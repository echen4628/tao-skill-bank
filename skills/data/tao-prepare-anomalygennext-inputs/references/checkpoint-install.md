# AnomalyGenNext 1.1 checkpoint installation

Use the installer shipped in the exact container image declared by
`skill_info.yaml`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

The script is part of the AnomalyGenNext release, owns the version-matched
checkpoint layout, and should remain the single implementation of the full
download and DCP conversion procedure. Do not copy it out of the image or
replace it with a sequence of independently maintained Hugging Face downloads.

## Launch contract

Checkpoint acquisition is side effecting. Resolve the pinned image, complete
the common launch review, and run the script through the selected platform.
The durable checkpoint destination must be new or an explicitly reviewed
existing installation and must be writable during installation:

```bash
export CKPT_DIR=/workspace/paidf-anomalygen/checkpoints
bash /workspace/paidf-anomalygen/scripts/download_checkpoints.sh
```

Bind the durable host destination to `CKPT_DIR`. After successful installation,
all consumers mount that same root read-only at:

```text
/workspace/paidf-anomalygen/checkpoints
```

On SLURM, the script and Python runtime stay inside the container. Keep
`TMPDIR`, Hugging Face caches not published as part of `CKPT_DIR`, UV caches,
and other temporary state on job-specific node-local storage. Checkpoints and
compact logs may be copied to durable shared storage using bounded copy-back.
Do not execute a host checkout or virtual environment from Lustre.

Set `HF_TOKEN` in the job environment only after the required Hugging Face
licenses have been accepted. Never place it in the command, spec, logs, or job
record. Registry authentication may separately require `NGC_KEY`, depending on
the selected platform.

## Full-install scope

The upstream script does not accept workflow profiles. It installs the complete
1.1 asset set, including:

- native DCP trees for Cosmos3-Nano and Cosmos3-Edge;
- a direct Transformers-format Cosmos3-Nano model for AMP grounding;
- DINOv2 and C-RADIO KPI backbones;
- the Wan2.2 VAE and SAM2.1 checkpoint;
- Qwen caption-tokenizer and Edge processor cache entries; and
- Cosmos-Guardrail1 and Qwen3Guard assets.

There is no supported Nano-only, fine-tune-only, or `--no-guardrail` download
mode. Treat it as a one-time full shared installation and report that storage
scope before launch. A consumer may still select `--no-guardrail` at runtime;
that flag changes generation behavior, not what the installer downloads.

## Narrow DCP conversion fallback

The full installer remains the default when the shared root or any companion
assets are missing. If the companion root is already complete and only the
official Cosmos3-Nano base checkpoint is available in Hugging Face format,
convert that checkpoint with the converter baked into the same pinned image:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o /models/Cosmos3-Nano-dcp \
  --checkpoint-path Cosmos3-Nano
```

Mount `/models` to persistent writable storage. The registered
`Cosmos3-Nano` name resolves the official checkpoint; an already-staged
Hugging Face checkpoint may instead be supplied as an absolute,
container-visible path. This fallback creates only the native DCP tree needed
by fine-tuning. It is not a profile mode and does not reconstruct or validate
the VAE, tokenizer cache, DINOv2, C-RADIO, or other companion assets.

## Completion boundary

Accept the installation only when the upstream script exits successfully and
the intended consumer's preflight accepts the root. The expected structure
includes:

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

The image's installer owns its revision selection, checksum manifests, cache
references, and conversion semantics. Consumer preflights remain responsible
for the subset they actually load. Do not mix converted shards or manifests
from a different image build into an accepted root.

## Guardrail limitation

Installing the guardrail repositories does not make guardrail-enabled
generation fully network-independent in the accepted 1.1 runtime. Guardrail
initialization uses an isolated UV invocation that may fetch Python packages
such as `hf-xet`. The current framework preset can also report
`image_guardrail_enforcing=false`: text screening and face blurring remain
active, but no image-safety model is available to block generated images.

`--no-guardrail` generation avoids that initialization path. Treat an offline
checkpoint-tree verification separately from a claim that guardrail-enabled
generation can run in a network-disabled container.
