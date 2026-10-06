# AnomalyGenNext 1.1 container runtime

Read this before submitting generation.

Resolve the image through `references/skill_info.yaml`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

The image contains AnomalyGenNext 1.1 and its runtime at
`/workspace/paidf-anomalygen`. Do not substitute the older 1.0 image and do
not overlay an external checkout or host virtualenv.

Expose the testcase or prepared-input result, task checkpoint and recipe,
Cosmos3-Nano base checkpoint, complete checkpoint root, this skill directory,
and a new durable output directory. Preserve absolute paths or rewrite all
related paths consistently inside the compute frame. The platform owns
writable temporary storage and caches.

The image defaults to the non-root `anomalygen` user with UID 10000. Every
input mount must be readable and traversable by the effective container user;
every output, temporary, and cache mount must be writable. Before starting
Python, use the selected image, mounts, and runtime identity to create and
remove a nested temporary file in each writable mount. Stop on failure and
report the affected host path with its ownership and mode. If the platform
maps the process to the host user instead, that identity must resolve to a
username and all home and cache locations must remain writable. For the local
Docker identity-mapping recipe, see
[Running in Docker](../../tao-generate-anomalies/references/docker.md).

Mount the complete checkpoint root read-only at
`/workspace/paidf-anomalygen/checkpoints`. The action rejects any other mount
location. It always validates `Qwen/Qwen3-VL-8B-Instruct` and
`nvidia/Cosmos3-Edge` in the Hugging Face cache under `hf/`, plus a
Transformers-compatible `facebook/dinov2-large`.

Create a missing root with the pinned image's canonical full installer as
documented in
[`tao-prepare-anomalygennext-inputs`](../../tao-prepare-anomalygennext-inputs/references/checkpoint-install.md).
Do not reconstruct the root with independent Hub downloads; the upstream
script owns the DCP conversion, direct model tree, cache references, and KPI
assets.

The wrapper preserves the native default invocation when guardrails are on and
passes `--no-guardrail` only when they are explicitly disabled. The flag
disables text screening, image content-safety screening, and face blurring
together. When guardrails are enabled, validation also requires
`Qwen/Qwen3Guard-Gen-0.6B` and `nvidia/Cosmos-Guardrail1`; those repositories
are not required with `--no-guardrail`. It then forces Hugging Face and
Transformers offline. Missing assets fail before sampling. Keep registry and
Hugging Face credentials out of specs, commands, logs, and job records.

The accepted 1.1 guardrail path is not fully network-independent merely because
the model repositories are installed. Its isolated UV invocation may fetch a
Python package such as `hf-xet`, and the current framework preset can report
`image_guardrail_enforcing=false`. In that state text screening and face
blurring remain active, but no image-safety model can block a generated image.
Use `--no-guardrail` when a strictly network-disabled generation smoke is
required.

Invoke the command declared in `skill_info.yaml`, for example inside the
container:

```bash
python /opt/tao-generate-od-defects/scripts/generate_od_defects.py \
  --inputs-dir /inputs/prepared \
  --base-checkpoint /models/Cosmos3-Nano \
  --checkpoint-root /workspace/paidf-anomalygen/checkpoints \
  --output-dir /results/generation \
  --no-guardrail \
  --num-gpus 1
```

The exposed GPU count must match `--num-gpus`. The output directory must not
already exist. The base path must contain `checkpoint.json` and the `model/`
checkpoint directory; it is not the shared checkpoint root. Read
`execution-contract.md` for the completion and accounting gates.
