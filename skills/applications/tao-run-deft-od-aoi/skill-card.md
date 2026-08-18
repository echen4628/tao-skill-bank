## Description: <br>
Runs or reproduces the binary RT-DETR DEFT OD AOI loop with loose and strict gap analysis, policy-routed real and clean mining, optional AnomalyGenNext synthesis, admission control, cumulative COCO assembly, and KPI-selected adaptive training. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers and engineers reproducing the reference DEFT OD AOI policy or adapting its explicit FP/FN routing to a new binary RT-DETR defect-detection loop while preserving the canonical Grounding-DINO DEFT workflow separately. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Low-IoU false positives on defect-containing KPI images do not measure all false alarms on completely clean KPI images. <br>
Mitigation: Report the analyzer's GT-present-class limitation and evaluate a dedicated clean set separately when clean-image false-alarm coverage matters. <br>

Risk: Uniform mining or synthetic generation can dominate the data mix. <br>
Mitigation: Freeze the uniform value, cumulative clean cap, cumulative synthetic cap, and generation enablement before launch; review routing and assembly reports every iteration. <br>

Risk: KPI leakage or test-driven model selection can inflate reported quality. <br>
Mitigation: Keep KPI pixels query-only, validate disjoint pools, select checkpoints on KPI only, and keep test report-only. <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, JSON policy, COCO dataset, Model checkpoints, Analysis] <br>
**Output Format:** [Markdown, JSON, YAML, Parquet, COCO JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>

## Evaluation Tasks: <br>
Evaluated against 3 packaged planning and workflow-disambiguation tasks. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: security, correctness, discoverability, effectiveness, and efficiency. <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility. Users should review the selected data, synthetic outputs, thresholds, and deployment requirements with their internal teams. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
