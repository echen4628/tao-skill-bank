#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the single policy-approved RT-DETR epoch extension spec."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--extended-epochs", type=int, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    source = Path(args.input).expanduser().resolve()
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("train spec must be a nested mapping")
    train = spec.get("train")
    dataset = spec.get("dataset")
    if not isinstance(train, dict) or not isinstance(dataset, dict):
        raise ValueError("train and dataset mappings are required")
    if args.expected_epochs < 1:
        raise ValueError("expected_epochs must be positive")
    if train.get("num_epochs") != args.expected_epochs:
        raise ValueError(
            f"expected source num_epochs={args.expected_epochs}, "
            f"got {train.get('num_epochs')}"
        )
    if args.extended_epochs != args.expected_epochs + 12:
        raise ValueError(
            "the DEFT OD AOI policy permits exactly one 12-epoch extension: "
            f"{args.expected_epochs} -> {args.extended_epochs}"
        )
    resume = Path(args.resume_checkpoint).expanduser().resolve()
    expected_resume = f"model_epoch_{args.expected_epochs - 1:03d}.pth"
    if resume.name != expected_resume:
        raise ValueError(
            "extension must resume the terminal checkpoint "
            f"{expected_resume}: {resume}"
        )
    if not resume.is_file():
        raise FileNotFoundError(resume)

    train["num_epochs"] = args.extended_epochs
    train["resume_training_checkpoint_path"] = str(resume)
    dataset["workers"] = 0

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)
    return {
        "source_num_epochs": args.expected_epochs,
        "extended_num_epochs": args.extended_epochs,
        "resume_checkpoint": str(resume),
        "output": str(output),
        "workers": 0,
    }


def main() -> int:
    try:
        report = run(parse_args())
        print(
            "RTDETR_EXTENSION_SPEC_READY "
            f"epochs={report['extended_num_epochs']} workers=0 "
            f"resume={report['resume_checkpoint']} output={report['output']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
