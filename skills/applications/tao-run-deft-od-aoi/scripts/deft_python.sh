#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select an already-provisioned Python without installing or loading secrets.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../../../.." && pwd)

probe='import numpy,pandas,pyarrow,PIL,yaml'
candidates=(
  "${DEFT_PYTHON:-}"
  "$repo_root/.venv/deft/bin/python"
  "$repo_root/.venv/bin/python"
  "${WORKSPACE_DIR:-}/.venv/bin/python"
  "${WORKSPACE:-}/.venv/bin/python"
  "$(command -v python3 2>/dev/null || true)"
)

selected=
for candidate in "${candidates[@]}"; do
  [ -n "$candidate" ] || continue
  [ -x "$candidate" ] || continue
  if "$candidate" -c "$probe" >/dev/null 2>&1; then
    selected=$candidate
    break
  fi
done

if [ -z "$selected" ]; then
  echo "deft_python: no installed Python provides numpy,pandas,pyarrow,PIL,yaml" >&2
  echo "deft_python: provision dependencies outside this workflow" >&2
  exit 2
fi

if [ "$#" -eq 0 ]; then
  printf '%s\n' "$selected"
  exit 0
fi

exec "$selected" "$@"
