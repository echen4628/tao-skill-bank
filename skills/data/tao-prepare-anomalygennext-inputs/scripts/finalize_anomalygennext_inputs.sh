#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Consume completed embeddings, perform pair-preserving retrieval and AMP, and freeze testcase inputs.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: finalize_anomalygennext_inputs.sh --config PATH --output-dir PATH [options]

Required:
  --config PATH             Same filtering YAML used by prepare_anomalygennext_sources.sh
  --output-dir PATH         Prepared directory containing both embedding outputs

Optional:
  --anomalygen-repo PATH    AnomalyGenNext checkout/shared install
  --activate PATH           Environment activation script
  --python PATH             Python with AnomalyGenNext dependencies
  --pipeline-py PATH        Preparation implementation; defaults beside this script
  --hf-cache PATH           AnomalyGenNext HuggingFace cache
  -h, --help                Show this help
EOF
}

while (($#)); do
  case "$1" in
    --config) PIPELINE_CONFIG=${2:?missing value for --config}; shift 2 ;;
    --output-dir) RUN_ROOT=${2:?missing value for --output-dir}; shift 2 ;;
    --anomalygen-repo) ANOMALYGEN_REPO=${2:?missing value for --anomalygen-repo}; shift 2 ;;
    --activate) ANOMALYGEN_ACTIVATE=${2:?missing value for --activate}; shift 2 ;;
    --python) ANOMALYGEN_PYTHON=${2:?missing value for --python}; shift 2 ;;
    --pipeline-py) PIPELINE_PY=${2:?missing value for --pipeline-py}; shift 2 ;;
    --hf-cache) ANOMALYGEN_HF_CACHE=${2:?missing value for --hf-cache}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${RUN_ROOT:?--output-dir must be set}
PIPELINE_CONFIG=${PIPELINE_CONFIG:?--config must be set}
ANOMALYGEN_REPO=${ANOMALYGEN_REPO:?ANOMALYGEN_REPO or --anomalygen-repo must be set}
ANOMALYGEN_ACTIVATE=${ANOMALYGEN_ACTIVATE:-$ANOMALYGEN_REPO/.venv/bin/activate}
ANOMALYGEN_HF_CACHE=${ANOMALYGEN_HF_CACHE:-$RUN_ROOT/cache/hf_anomalygen}
PIPELINE_PY=${PIPELINE_PY:-$SCRIPT_DIR/prepare_anomalygennext_inputs.py}

test -f "$RUN_ROOT/prepared_anomalygennext_inputs/filtering_config.yaml"
cmp -s "$PIPELINE_CONFIG" "$RUN_ROOT/prepared_anomalygennext_inputs/filtering_config.yaml" || {
  echo "config does not match the prepared filtering snapshot" >&2
  exit 2
}
test -s "$RUN_ROOT/embeddings/clean_embeddings.parquet"
test -s "$RUN_ROOT/embeddings/fn_embeddings.parquet"
test -f "$PIPELINE_PY"
test -f "$ANOMALYGEN_ACTIVATE"

source "$ANOMALYGEN_ACTIVATE"
ANOMALYGEN_PYTHON=${ANOMALYGEN_PYTHON:-$(command -v python)}
test -x "$ANOMALYGEN_PYTHON"
mkdir -p "$RUN_ROOT/logs" "$ANOMALYGEN_HF_CACHE"

STATUS_FILE="$RUN_ROOT/status.json"
write_status() {
  rc=$?
  trap - EXIT
  state=ERROR
  if [ "$rc" -eq 0 ]; then state=COMPLETE; fi
  STATUS_STATE="$state" STATUS_RC="$rc" STATUS_FILE="$STATUS_FILE" \
    "$ANOMALYGEN_PYTHON" -c \
    'import json,os; from pathlib import Path; Path(os.environ["STATUS_FILE"]).write_text(json.dumps({"state":os.environ["STATUS_STATE"],"exit_code":int(os.environ["STATUS_RC"])},indent=2)+"\n")'
  exit "$rc"
}
trap write_status EXIT

"$ANOMALYGEN_PYTHON" "$PIPELINE_PY" build-knn-and-amp \
  --config "$RUN_ROOT/prepared_anomalygennext_inputs/filtering_config.yaml" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/logs/build_knn_and_amp.log"

export HF_HOME="$ANOMALYGEN_HF_CACHE"
"$ANOMALYGEN_PYTHON" -m anomalygen.scripts.auto_mask_placement.roi_place \
  --input_pair_path "$RUN_ROOT/amp/amp_samples.json" \
  --defect_desc "$RUN_ROOT/specs/defect_spec.jsonl" \
  --output_dir "$RUN_ROOT/amp" --n_seeds 1 --seed 43 \
  --model_id nvidia/Cosmos3-Nano \
  2>&1 | tee "$RUN_ROOT/logs/amp.log"

test -s "$RUN_ROOT/amp/testcase.jsonl"
"$ANOMALYGEN_PYTHON" "$PIPELINE_PY" finalize-inputs \
  --config "$RUN_ROOT/prepared_anomalygennext_inputs/filtering_config.yaml" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/logs/finalize_inputs.log"
"$ANOMALYGEN_PYTHON" "$PIPELINE_PY" validate-prepared-inputs \
  --prepared-inputs-root "$RUN_ROOT"
