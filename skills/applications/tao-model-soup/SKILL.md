---
name: tao-model-soup
description: >
  Build a uniform or greedy model soup by averaging compatible TAO
  checkpoints, with automatic KPI evaluation and auditable ingredient
  selection. Use when combining several RT-DETR fine-tuned checkpoints into
  one checkpoint without adding inference-time ensembles, comparing uniform
  and greedy weight averaging, or adding a final consolidation stage to a
  training or DEFT workflow. The packaged implementation currently supports
  RT-DETR PyTorch checkpoints and is dataset-agnostic. Do not use for
  checkpoints with different architectures, tensor layouts, or class heads.
license: Apache-2.0
compatibility: Requires the TAO skill bank, Python 3.10+, PyYAML, a TAO PyTorch container with torch, compatible RT-DETR checkpoints, and a selected execution platform. Greedy soup also requires a held-out KPI evaluation spec.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- application
- model-soup
- checkpoint-averaging
- rtdetr
- object-detection
---

# TAO Model Soup

Create one deployable checkpoint by averaging weights from compatible
fine-tuned models. This skill implements the uniform and greedy procedures from
[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)
(Wortsman et al., ICML 2022).

The implementation is reusable across datasets and workflows. Its first
checkpoint adapter supports TAO RT-DETR `.pth` files; add another adapter only
after its checkpoint and evaluation contracts are known.

## Execution contract

1. Ask which installed platform to use. Do not default among Docker, SLURM,
   Kubernetes, Brev, virtualenv, or an external platform.
2. Read the selected platform skill and run its Preflight. Read
   `references/method-and-runtime.md`, then use `tao-launch-workflow` for the
   launch review and record-before-launch gate.
3. Require at least two trusted checkpoints from the same architecture and
   class-head contract. Prefer checkpoints fine-tuned from the same
   initialization; do not mix architectures, class counts, state-dict keys,
   tensor shapes, or tensor dtypes.
4. For `greedy`, require one held-out KPI/validation evaluation spec. Never use
   a sealed test set to order ingredients, accept a candidate, choose a method,
   or tune `min-improvement`.
5. Run the packaged `scripts/tao_model_soup.py` inside the resolved TAO PyTorch
   image. Submit, monitor, and cancel through the selected platform's four
   verbs. The script itself does not submit jobs.
6. Gate on `soup_manifest.json`, the final checkpoint hash, and a successful
   KPI evaluation record. Evaluate on test only after the final checkpoint and
   soup method are frozen.

TAO checkpoints are pickle-backed. Load only checkpoints from trusted sources.
Never place credentials in commands, specs, or manifests.

## Choose a method

- **Uniform** averages every supplied checkpoint equally. It requires no
  selection evaluation and is the simplest paper baseline.
- **Greedy** evaluates each individual checkpoint on KPI, starts from the best,
  then considers remaining checkpoints in score order. A candidate is accepted
  only when the equal-weight soup of all accepted ingredients plus that
  candidate improves KPI by more than `--min-improvement`.

Greedy is the default when a trustworthy KPI split exists. Uniform is the
default when it does not. Both produce one ordinary RT-DETR checkpoint with no
ensemble-time compute or memory increase.

## Quick Start

Uniform soup:

```bash
python <skill_root>/scripts/tao_model_soup.py \
  --method uniform \
  --checkpoint /inputs/run_a/model_epoch_036.pth \
  --checkpoint /inputs/run_b/model_epoch_036.pth \
  --checkpoint /inputs/run_c/model_epoch_036.pth \
  --output-dir /results/model_soup
```

Greedy RT-DETR soup, using a nested TAO evaluate YAML whose dataset points to
the held-out KPI split:

```bash
python <skill_root>/scripts/tao_model_soup.py \
  --method greedy \
  --checkpoint /inputs/run_a/model_epoch_036.pth \
  --checkpoint /inputs/run_b/model_epoch_036.pth \
  --checkpoint /inputs/run_c/model_epoch_036.pth \
  --eval-spec /inputs/rtdetr_kpi_evaluate.yaml \
  --metric-key mAP50 \
  --output-dir /results/model_soup
```

The default evaluator is `rtdetr evaluate -e {spec}`. For another compatible
evaluator, pass a JSON argv array through `--eval-command-json`; the supported
placeholders are `{checkpoint}`, `{results_dir}`, `{spec}`, and `{label}`.
Shell parsing is never used.

## Required RT-DETR spec

Start from `tao-train-rtdetr/references/spec_template_evaluate.yaml`. Set the
KPI image and COCO annotation paths, class count, class IDs, and all other
model settings to match training. Leave `evaluate.checkpoint` and
`results_dir` as placeholders: the script writes a fresh nested spec for each
evaluation without mutating the template.

By default the metric reader recursively searches evaluation JSON for the
exact `--metric-key`. Use `--metric-file-relative evaluate/status.json` when
the output layout is known. Use a key such as `mAP50`, `val_mAP50`, or
`kpi.mAP50` that the evaluator actually emits.

## Outputs

`output_dir` contains:

```text
model_soup.pth
soup_manifest.json
evaluations/              # greedy only
  individual-*/
  candidate-*/
```

The manifest records the method, direction, metric, checkpoint hashes,
compatibility descriptor, individual and candidate scores, accepted
ingredients, equal final weights, non-floating tensor policy, and final hash.
Floating tensors are averaged on CPU; non-floating tensors are copied from the
first ingredient and their disagreement count is reported.

## Hard gates

- Reject missing, duplicate, or untrusted checkpoint inputs.
- Reject different state-dict locations, keys, shapes, or dtypes before any
  evaluation runs.
- Reject non-finite weights or metrics and evaluator failures.
- Do not compare greedy and uniform on test and then select the winner.
- Do not treat optimizer-state averaging as a resumable training checkpoint;
  the output is for evaluate, inference, export, or a separately justified
  fine-tuning initialization.
- Do not claim an improvement until the frozen final soup has been evaluated.

## References

- `references/method-and-runtime.md` — exact algorithms, RT-DETR adapter,
  platform launch shape, outputs, and failure recovery.
- `scripts/tao_model_soup.py --help` — the complete one-file CLI.
