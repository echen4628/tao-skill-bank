## Description: <br>
Projects COCO taxonomies and stages selected COCO training data through TAO Data Services for object-detection workflows. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers preparing multiclass or projected-binary COCO datasets and materializing mined image selections for detector training. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: The pinned Data Services image may predate the `annotations project` and `annotations stage` actions. <br>
Mitigation: Verify both action surfaces after launch approval and stop if either is absent. <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [COCO JSON, JSONL, YAML, text, JSON report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Codex (`codex`) <br>

## Evaluation Tasks: <br>
Evaluated against 2 application-contract tasks. <br>

## Evaluation Metrics Used: <br>
- Correctness <br>
- Discoverability <br>
- Skill execution <br>
- Behavior checks <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility. Review dataset provenance and label policy before using generated training data. <br>
