#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run AnomalyGenNext inference and pseudo-label an OD defect dataset.

set -Eeuo pipefail
set -o pipefail

usage() {
  cat <<'EOF'
Usage: generate_od_defects.sh [options]

Required (flag or same-named environment variable):
  --inputs-dir PATH          Completed tao-prepare-anomalygennext-inputs root (PREPARED_INPUTS_ROOT)
  --output-dir PATH          New generation run directory (RUN_ROOT)

Optional:
  --datasets IDS             Comma-separated frozen dataset ids (GENERATION_DATASETS)
  --num-gpus N               Visible GPUs used by torchrun; default 1 (NUM_GPUS)
  --anomalygen-repo PATH     AnomalyGenNext checkout (ANOMALYGEN_REPO)
  --activate PATH            Environment activation script (ANOMALYGEN_ACTIVATE)
  --python PATH              Python with AnomalyGenNext deps (ANOMALYGEN_PYTHON)
  --base-checkpoint PATH     Cosmos3-Nano base DCP (BASE_CHECKPOINT)
  --uv-bin-dir PATH          Directory containing uv (UV_BIN_DIR)
  --pipeline-py PATH         Generation implementation; defaults beside this script (PIPELINE_PY)
  --hf-cache PATH            AnomalyGenNext HF cache (ANOMALYGEN_HF_CACHE)
  --job-id ID                Audit label; defaults to output basename (JOB_ID)
  --resume-existing-generation
                             Reuse completed raw generation and continue eval/labels
  -h, --help                 Show this help

The script validates every frozen input SHA-256 before generation. Run it
inside a SLURM allocation with at least --num-gpus visible GPUs.
EOF
}

while (($#)); do
  case "$1" in
    --inputs-dir) PREPARED_INPUTS_ROOT=${2:?missing value for --inputs-dir}; shift 2 ;;
    --output-dir) RUN_ROOT=${2:?missing value for --output-dir}; shift 2 ;;
    --datasets) GENERATION_DATASETS=${2:?missing value for --datasets}; shift 2 ;;
    --num-gpus) NUM_GPUS=${2:?missing value for --num-gpus}; shift 2 ;;
    --anomalygen-repo) ANOMALYGEN_REPO=${2:?missing value for --anomalygen-repo}; shift 2 ;;
    --activate) ANOMALYGEN_ACTIVATE=${2:?missing value for --activate}; shift 2 ;;
    --python) ANOMALYGEN_PYTHON=${2:?missing value for --python}; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT=${2:?missing value for --base-checkpoint}; shift 2 ;;
    --uv-bin-dir) UV_BIN_DIR=${2:?missing value for --uv-bin-dir}; shift 2 ;;
    --pipeline-py) PIPELINE_PY=${2:?missing value for --pipeline-py}; shift 2 ;;
    --hf-cache) ANOMALYGEN_HF_CACHE=${2:?missing value for --hf-cache}; shift 2 ;;
    --job-id) JOB_ID=${2:?missing value for --job-id}; shift 2 ;;
    --resume-existing-generation) RESUME_EXISTING_GENERATION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${RUN_ROOT:?RUN_ROOT or --output-dir must be set}
PREPARED_INPUTS_ROOT=${PREPARED_INPUTS_ROOT:?PREPARED_INPUTS_ROOT or --inputs-dir must be set}
ANOMALYGEN_REPO=${ANOMALYGEN_REPO:?ANOMALYGEN_REPO or --anomalygen-repo must be set}
ANOMALYGEN_ACTIVATE=${ANOMALYGEN_ACTIVATE:-$ANOMALYGEN_REPO/.venv/bin/activate}
PIPELINE_PY=${PIPELINE_PY:-$SCRIPT_DIR/generate_od_defects.py}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-$ANOMALYGEN_REPO/checkpoints/Cosmos3-Nano/model}
ANOMALYGEN_HF_CACHE=${ANOMALYGEN_HF_CACHE:-$(dirname "$ANOMALYGEN_REPO")/hf_cache}
NUM_GPUS=${NUM_GPUS:-1}
GENERATION_DATASETS=${GENERATION_DATASETS:-}
JOB_ID=${JOB_ID:-$(basename "$RUN_ROOT")}
RESUME_EXISTING_GENERATION=${RESUME_EXISTING_GENERATION:-0}

case "$NUM_GPUS" in
  ''|*[!0-9]*|0) echo "NUM_GPUS must be a positive integer" >&2; exit 2 ;;
esac

test -d "$PREPARED_INPUTS_ROOT"
test -f "$PIPELINE_PY"
test -d "$ANOMALYGEN_REPO"
test -f "$ANOMALYGEN_ACTIVATE"
test -e "$BASE_CHECKPOINT"

if [ -e "$RUN_ROOT" ]; then
  if [ "$RESUME_EXISTING_GENERATION" -ne 1 ]; then
    echo "refusing to overwrite existing generation directory: $RUN_ROOT" >&2
    exit 2
  fi
  test -d "$RUN_ROOT"
elif [ "$RESUME_EXISTING_GENERATION" -eq 1 ]; then
  echo "--resume-existing-generation requires an existing output directory: $RUN_ROOT" >&2
  exit 2
fi

STATUS_FILE="$RUN_ROOT/status.json"
LOG_DIR="$RUN_ROOT/logs"

write_status() {
  rc=$?
  trap - EXIT
  state=ERROR
  if [ "$rc" -eq 0 ]; then state=COMPLETE; fi
  STATUS_STATE="$state" STATUS_RC="$rc" STATUS_FILE="$STATUS_FILE" \
    "$ANOMALYGEN_PYTHON" -c \
    'import json,os; from datetime import datetime,timezone; from pathlib import Path; Path(os.environ["STATUS_FILE"]).write_text(json.dumps({"state":os.environ["STATUS_STATE"],"exit_code":int(os.environ["STATUS_RC"]),"updated_at":datetime.now(timezone.utc).isoformat(timespec="seconds")},indent=2)+"\n")'
  exit "$rc"
}

mkdir -p "$RUN_ROOT" "$LOG_DIR"
source "$ANOMALYGEN_ACTIVATE"
ANOMALYGEN_PYTHON=${ANOMALYGEN_PYTHON:-$(command -v python)}
test -x "$ANOMALYGEN_PYTHON"
if [ -z "${UV_BIN_DIR:-}" ]; then
  uv_path=$(command -v uv || true)
  if [ -z "$uv_path" ]; then
    echo "uv is not on PATH; pass --uv-bin-dir /directory/containing/uv" >&2
    exit 2
  fi
  UV_BIN_DIR=$(dirname "$uv_path")
fi
test -x "$UV_BIN_DIR/uv" || {
  echo "uv is not executable at $UV_BIN_DIR/uv" >&2
  exit 2
}
export PATH="$UV_BIN_DIR:$PATH"
if [ -z "${ANOMALYGEN_SHARED_SITE:-}" ]; then
  ANOMALYGEN_SHARED_SITE=$(
    "$ANOMALYGEN_PYTHON" -c 'import site; print(site.getsitepackages()[0])'
  )
fi
export PYTHONPATH="$ANOMALYGEN_REPO:$ANOMALYGEN_REPO/assets/lerobot_stub:$ANOMALYGEN_SHARED_SITE"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ANOMALYGEN_HF_CACHE"
trap write_status EXIT

echo "phase=anomalygen_next_generation"
echo "job_id=$JOB_ID"
echo "run_root=$RUN_ROOT"
echo "prepared_inputs_root=$PREPARED_INPUTS_ROOT"
echo "num_gpus=$NUM_GPUS"
echo "started=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

"$ANOMALYGEN_PYTHON" "$PIPELINE_PY" validate-inputs \
  --inputs-dir "$PREPARED_INPUTS_ROOT" 2>&1 | tee "$LOG_DIR/validate_inputs.log"
plan_args=(--inputs-dir "$PREPARED_INPUTS_ROOT")
finalize_args=(--inputs-dir "$PREPARED_INPUTS_ROOT" --run-root "$RUN_ROOT")
if [ -n "$GENERATION_DATASETS" ]; then
  plan_args+=(--datasets "$GENERATION_DATASETS")
  finalize_args+=(--datasets "$GENERATION_DATASETS")
fi
"$ANOMALYGEN_PYTHON" "$PIPELINE_PY" emit-plan \
  "${plan_args[@]}" > "$RUN_ROOT/anomalygen_next_generation_plan.tsv"
test -s "$RUN_ROOT/anomalygen_next_generation_plan.tsv"

while IFS=$'\t' read -r dataset_id anomaly_types_csv testcase provenance checkpoint recipe real_root requested; do
  test -n "$dataset_id"
  IFS=',' read -r -a anomaly_types <<< "$anomaly_types_csv"
  group="$RUN_ROOT/$dataset_id"
  raw="$group/raw"
  searched="$group/searched"
  rounds="$group/rounds"
  mkdir -p "$raw" "$searched" "$rounds"

  echo "generation group=$dataset_id types=$anomaly_types_csv requested=$requested"
  if [ "$RESUME_EXISTING_GENERATION" -eq 1 ]; then
    test -s "$raw/texture_ft_generation_result.csv"
    echo "reuse completed raw generation group=$dataset_id"
  else
    torchrun --nproc_per_node="$NUM_GPUS" "$ANOMALYGEN_REPO/anomalygen/scripts/texture/generate.py" \
      --checkpoint "$checkpoint" --recipe "$recipe" \
      --base_checkpoint "$BASE_CHECKPOINT" \
      --input_data_path "$testcase" --output_dir "$raw" \
      2>&1 | tee "$LOG_DIR/generation_${dataset_id}.log"
  fi

  "$ANOMALYGEN_PYTHON" "$ANOMALYGEN_REPO/anomalygen/scripts/texture/evaluate.py" \
    --gen_root "$raw" --real_root "$real_root" --recipe "$recipe" \
    --anomaly_types "${anomaly_types[@]}" --output_file "$raw/${dataset_id}_kpi.json" \
    2>&1 | tee "$LOG_DIR/evaluation_${dataset_id}.log"

  "$ANOMALYGEN_PYTHON" "$ANOMALYGEN_REPO/anomalygen/scripts/texture/quality_refine.py" select \
    --original "$raw" --original_kpi "$raw/${dataset_id}_kpi.json" \
    --rounds_dir "$rounds" --output "$searched" \
    2>&1 | tee "$LOG_DIR/select_${dataset_id}.log"

  "$ANOMALYGEN_PYTHON" "$ANOMALYGEN_REPO/anomalygen/scripts/texture/evaluate.py" \
    --gen_root "$searched" --real_root "$real_root" --recipe "$recipe" \
    --anomaly_types "${anomaly_types[@]}" --output_file "$searched/${dataset_id}_kpi.json" \
    2>&1 | tee "$LOG_DIR/final_evaluation_${dataset_id}.log"

  "$ANOMALYGEN_PYTHON" "$ANOMALYGEN_REPO/anomalygen/scripts/texture/pseudo_label.py" \
    --gen_root "$searched" --output_dir "$searched/pseudo_labels" --no_caption \
    2>&1 | tee "$LOG_DIR/pseudo_label_${dataset_id}.log"
  test -s "$searched/pseudo_labels/coco_annotations.json"
done < "$RUN_ROOT/anomalygen_next_generation_plan.tsv"

"$ANOMALYGEN_PYTHON" "$PIPELINE_PY" finalize \
  "${finalize_args[@]}" \
  2>&1 | tee "$LOG_DIR/finalize_generation.log"
test -s "$RUN_ROOT/validation_summary.json"
echo "completed=$(date --iso-8601=seconds)"
