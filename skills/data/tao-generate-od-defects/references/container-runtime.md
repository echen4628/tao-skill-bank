# AnomalyGenNext 1.1 Docker generation

Use this reference when the selected platform is Docker. Other platforms own
their own image import, mounts, staging, cache, and resource conventions; keep
those details out of this data skill.

## Required image

Resolve the image through `references/skill_info.yaml` or its version key:

```text
images.metropolis_sdg.anomalygen_next
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

This official NVIDIA NGC image contains AnomalyGen 1.1, the source tree at
`/workspace/paidf-anomalygen`, and its baked Python environment. The
`nvcr.io/nvidia/paidf-anomalygen:1.0.1` image <!-- versions-key: images.metropolis_sdg.paidf_anomalygen --> is an older
Cosmos-Predict2 release and cannot load Cosmos3-Nano task weights.

If Docker does not already have the image, pull it from NGC after launch
approval:

```bash
docker pull "$AG_IMAGE"
```

The image is publicly resolvable. If the selected platform requires NGC
authentication, check only whether `NGC_KEY` is set and use the platform's
standard `nvcr.io` login flow; never print the value. Registry credentials are
for image import only. Do not pass them into the generation container. Pass
`HF_TOKEN` only when a selected model asset is gated and is not already present
in the mounted cache.

## Path contract

The frozen generation plan contains absolute paths for the testcase,
provenance, task checkpoint, recipe, and real-image root. Every referenced path
must be visible inside the container at the same absolute path. The simplest
layout puts all inputs, checkpoints, and results below one `WORKSPACE` and
bind-mounts it onto itself.

The image supplies its own source code. Do not mount an external checkout or a
host virtualenv over `/workspace/paidf-anomalygen`. Mount only:

- the workspace containing frozen inputs, task weights, recipes, base/evaluator
  checkpoints, real images, and the results parent;
- a writable runtime directory for home, temporary files, and caches; and
- this skill directory read-only so the image can invoke the bank wrapper.

The selected platform decides where those host directories live. If any frozen
path is outside `WORKSPACE`, add a read-only bind mount from that absolute host
path to the identical container path.

## Docker example

Run platform preflight and the launch review first. Replace the example paths;
do not create `OUTPUT` itself because ordinary generation refuses to overwrite
an existing run directory.

```bash
TAO_SKILL_BANK_PATH=/absolute/path/to/tao-skill-bank
CONTROL_PY="$TAO_SKILL_BANK_PATH/.venv/deft/bin/python"
GEN_SKILL="$TAO_SKILL_BANK_PATH/skills/data/tao-generate-od-defects"

AG_IMAGE=$(
  "$CONTROL_PY" "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" \
    images.metropolis_sdg.anomalygen_next
)

WORKSPACE=/absolute/path/containing-inputs-checkpoints-and-results
RUNTIME_ROOT=/absolute/path/to/writable-anomalygen-runtime
INPUTS="$WORKSPACE/prepared-inputs"
BASE_CHECKPOINT="$WORKSPACE/checkpoints/Cosmos3-Nano/model"
OUTPUT="$WORKSPACE/results/anomalygen_next_generation"

mkdir -p "$(dirname "$OUTPUT")" \
  "$RUNTIME_ROOT/home" "$RUNTIME_ROOT/tmp" \
  "$RUNTIME_ROOT/cache/huggingface" "$RUNTIME_ROOT/cache/uv" \
  "$RUNTIME_ROOT/cache/pip" "$RUNTIME_ROOT/cache/torch" \
  "$RUNTIME_ROOT/cache/triton" "$RUNTIME_ROOT/cache/torchinductor" \
  "$RUNTIME_ROOT/cache/matplotlib"

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/runtime/home \
  -e TMPDIR=/runtime/tmp \
  -e XDG_CACHE_HOME=/runtime/cache \
  -e HF_HOME=/runtime/cache/huggingface \
  -e UV_CACHE_DIR=/runtime/cache/uv \
  -e PIP_CACHE_DIR=/runtime/cache/pip \
  -e TORCH_HOME=/runtime/cache/torch \
  -e TRITON_CACHE_DIR=/runtime/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/runtime/cache/torchinductor \
  -e MPLCONFIGDIR=/runtime/cache/matplotlib \
  -v "$RUNTIME_ROOT:/runtime" \
  -v "$WORKSPACE:$WORKSPACE" \
  -v "$GEN_SKILL:/opt/tao-generate-od-defects:ro" \
  -w /workspace/paidf-anomalygen \
  "$AG_IMAGE" \
  bash /opt/tao-generate-od-defects/scripts/generate_od_defects.sh \
    --inputs-dir "$INPUTS" \
    --output-dir "$OUTPUT" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --pipeline-py /opt/tao-generate-od-defects/scripts/generate_od_defects.py \
    --num-gpus 1
```

`--num-gpus` must equal the GPUs exposed to the container. Add
`--datasets id_a,id_b` only to select existing rows from the frozen plan. The
wrapper validates every frozen SHA-256 before GPU work and emits native and
binary COCO files only after the completion gates in `../SKILL.md` pass.
