#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the newest valid RT-DETR epoch checkpoint from one train directory."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^model_epoch_(\d+)(-EMA)?\.pth$")


def resolve(train_dir: Path, use_ema: bool) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in train_dir.glob("model_epoch_*.pth"):
        match = CHECKPOINT_RE.match(path.name)
        if not match or bool(match.group(2)) != use_ema or path.stat().st_size == 0:
            continue
        candidates.append((int(match.group(1)), path.resolve()))
    if not candidates:
        suffix = "-EMA" if use_ema else ""
        raise FileNotFoundError(
            f"no non-empty model_epoch_<N>{suffix}.pth under {train_dir}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()
    try:
        train_dir = Path(args.train_dir).expanduser().resolve()
        if not train_dir.is_dir():
            raise NotADirectoryError(f"--train-dir is not a directory: {train_dir}")
        print(resolve(train_dir, args.ema))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

