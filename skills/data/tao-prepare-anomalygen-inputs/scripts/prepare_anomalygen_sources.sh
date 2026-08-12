#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Select eligible FN records and emit the two embedding input specifications.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: prepare_anomalygen_sources.sh --config PATH --output-dir PATH [options]

Required:
  --config PATH        Filtering and dataset-routing YAML
  --output-dir PATH    New or matching preparation directory

Optional:
  --python PATH        Host Python with pandas, pyarrow, Pillow, NumPy, and PyYAML
  --pipeline-py PATH   Preparation implementation; defaults beside this script
  -h, --help           Show this help

This command does not compute embeddings. It emits clean_embeddings.yaml and
fn_embeddings.yaml for tao-generate-image-embeddings.
EOF
}

while (($#)); do
  case "$1" in
    --config) PIPELINE_CONFIG=${2:?missing value for --config}; shift 2 ;;
    --output-dir) RUN_ROOT=${2:?missing value for --output-dir}; shift 2 ;;
    --python) PREPARE_PYTHON=${2:?missing value for --python}; shift 2 ;;
    --pipeline-py) PIPELINE_PY=${2:?missing value for --pipeline-py}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${RUN_ROOT:?--output-dir must be set}
PIPELINE_CONFIG=${PIPELINE_CONFIG:?--config must be set}
PIPELINE_PY=${PIPELINE_PY:-$SCRIPT_DIR/prepare_anomalygen_inputs.py}
PREPARE_PYTHON=${PREPARE_PYTHON:-python3}

test -f "$PIPELINE_CONFIG"
test -f "$PIPELINE_PY"
command -v "$PREPARE_PYTHON" >/dev/null 2>&1 || test -x "$PREPARE_PYTHON"

mkdir -p "$RUN_ROOT/phase1" "$RUN_ROOT/logs"
config_snapshot="$RUN_ROOT/phase1/filtering_config.yaml"
if [ -e "$config_snapshot" ]; then
  cmp -s "$PIPELINE_CONFIG" "$config_snapshot" || {
    echo "refusing to reuse $RUN_ROOT with a different filtering config" >&2
    exit 2
  }
else
  cp "$PIPELINE_CONFIG" "$config_snapshot"
fi

"$PREPARE_PYTHON" "$PIPELINE_PY" prepare-phase1 \
  --config "$config_snapshot" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/logs/prepare_sources.log"

STATUS_FILE="$RUN_ROOT/status.json" "$PREPARE_PYTHON" -c \
  'import json,os; from pathlib import Path; Path(os.environ["STATUS_FILE"]).write_text(json.dumps({"state":"PENDING","phase":"embedding","next_action":"run tao-generate-image-embeddings for specs/clean_embeddings.yaml and specs/fn_embeddings.yaml"},indent=2)+"\n")'

echo "prepared=$RUN_ROOT"
echo "clean_embedding_spec=$RUN_ROOT/specs/clean_embeddings.yaml"
echo "fn_embedding_spec=$RUN_ROOT/specs/fn_embeddings.yaml"
