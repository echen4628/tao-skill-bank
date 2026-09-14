# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static regression checks for AnomalyGenNext AMP runtime requirements."""

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills/data/tao-prepare-anomalygennext-inputs"


def test_amp_declares_gpu_and_external_sam2_runtime() -> None:
    metadata = yaml.safe_load((SKILL / "references/skill_info.yaml").read_text())
    text = (SKILL / "SKILL.md").read_text()
    script = (SKILL / "scripts/run_anomalygennext_amp.py").read_text()

    assert metadata["gpu_spec_key"] is None
    assert "Allocate exactly one NVIDIA GPU for `run_amp`" in text
    assert "Mount node-local working data somewhere else" in text
    assert "scratch over `/workspace` hides" in text
    assert metadata["actions"]["run_amp"]["inputs"]["sam2_checkpoint"]["type"] == "file"
    assert 'parser.add_argument("--sam2-checkpoint", type=Path, required=True)' in script
