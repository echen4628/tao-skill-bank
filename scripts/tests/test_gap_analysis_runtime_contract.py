# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static regression checks for OD gap-analysis runtime requirements."""

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills/data/tao-analyze-gaps-od-map"


def test_null_gpu_spec_pointer_does_not_advertise_cpu_compatibility() -> None:
    metadata = yaml.safe_load((SKILL / "references/skill_info.yaml").read_text())
    text = (SKILL / "SKILL.md").read_text()

    assert metadata["gpu_spec_key"] is None
    assert "Allocate exactly one NVIDIA GPU" in text
    assert "does not mean this action is CPU-compatible" in text
    assert "nvidia-smi -L" in text
