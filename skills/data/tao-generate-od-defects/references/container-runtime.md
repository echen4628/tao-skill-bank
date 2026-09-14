# AnomalyGenNext 1.1 container runtime

Read this before submitting generation.

Resolve the image through `references/skill_info.yaml`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

The image contains AnomalyGenNext 1.1 and its runtime at
`/workspace/paidf-anomalygen`. Do not substitute the older 1.0 image and do
not overlay an external checkout or host virtualenv. Do not mount a scratch
directory over `/workspace`; use a separate target such as `/runwork`, or the
mount will hide the release source that this action executes.

Expose the testcase or prepared-input result, task checkpoint and recipe,
Cosmos3-Nano base checkpoint, optional real-image root, optional Hugging Face
cache, this skill directory, and a new durable output directory. Preserve
absolute paths or rewrite all related paths consistently inside the compute
frame. The platform owns writable temporary storage and caches.

For offline use, preflight the selected Hugging Face cache for the tokenizer
and guardrail assets resolved by the pinned image:
`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3Guard-Gen-0.6B`,
`nvidia/Cosmos-Guardrail1`, and `nvidia/Cosmos3-Edge`. Each cached repository
needs `blobs/` and `snapshots/`; `refs/` is optional. Missing assets can fail
before sampling. Keep registry and Hugging Face credentials out of specs,
commands, logs, and job records.

Invoke the command declared in `skill_info.yaml`, for example inside the
container:

```bash
python /opt/tao-generate-od-defects/scripts/generate_od_defects.py \
  --inputs-dir /inputs/prepared \
  --base-checkpoint /models/Cosmos3-Nano/model \
  --output-dir /results/generation \
  --num-gpus 1
```

The exposed GPU count must match `--num-gpus`. The output directory must not
already exist. Read `execution-contract.md` for the completion and accounting
gates.
