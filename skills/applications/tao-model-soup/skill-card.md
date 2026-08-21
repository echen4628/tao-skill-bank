## Description: <br>
Builds a uniform or KPI-selected greedy model soup from compatible TAO RT-DETR checkpoints and emits one checkpoint plus an auditable selection manifest. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers consolidating several compatible RT-DETR fine-tuning runs without an inference-time ensemble. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Checkpoint averaging can silently produce a poor model when inputs have incompatible semantics or do not share a suitable low-error region. <br>
Mitigation: Enforce exact tensor compatibility, require matching architecture and class contracts, prefer a shared initialization, and evaluate the frozen result. <br>

Risk: Greedy selection on the test set leaks test information. <br>
Mitigation: Select ingredients only on a held-out KPI/validation split and evaluate test after the final soup is frozen. <br>

Risk: PyTorch checkpoint loading can execute untrusted pickle payloads. <br>
Mitigation: Accept checkpoints only from trusted sources. <br>

## Skill Output: <br>
**Output Type(s):** [Model checkpoint, JSON manifest, Evaluation logs, Analysis] <br>
**Output Format:** [PTH, JSON, YAML, Text] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>

## Evaluation Tasks: <br>
Evaluated against 3 packaged selection, compatibility, and leakage tasks. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: security, correctness, discoverability, effectiveness, and efficiency. <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility. Users should review checkpoint provenance, validation design, model quality, and deployment requirements with their internal teams. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
