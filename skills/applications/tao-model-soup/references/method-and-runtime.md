# Model soup method and runtime contract

## Paper algorithms

This skill implements the checkpoint-weight averaging procedures described by
Wortsman et al. in [Model soups: averaging weights of multiple fine-tuned
models improves accuracy without increasing inference
time](https://arxiv.org/abs/2203.05482).

Given compatible model weights `theta_1 ... theta_n`:

- **Uniform soup:** `theta_soup = (theta_1 + ... + theta_n) / n`.
- **Greedy soup:** evaluate every individual model on held-out validation,
  sort by the selection metric, and seed the soup with the best model. In that
  order, propose the equal-weight average of every already accepted original
  model and the next original model. Accept the proposal only if it strictly
  improves the held-out metric. `--min-improvement` raises that strict gate.

The greedy procedure never averages the previous soup as if it were one new
ingredient; each accepted original checkpoint retains equal weight.

Model soups work best when fine-tuned models occupy a connected low-error
region, so a shared initialization and compatible training contract are strong
preconditions. Tensor compatibility alone cannot prove that precondition.

## RT-DETR checkpoint adapter

The packaged CLI recognizes tensor state dictionaries at the checkpoint root
or under `state_dict`, `model_state_dict`, or `model`. Every input must resolve
to the same location and have identical tensor keys, shapes, and dtypes.

Floating and complex tensors are accumulated in higher precision on CPU and
cast back to the reference dtype. Integer and Boolean tensors, including batch
tracking counters, are copied from the first ingredient. The manifest reports
how many such tensors differed across ingredients. Other payload data is
preserved from the first ingredient so TAO can load the result, but optimizer
and trainer state is not averaged and must not be interpreted as resumable
training state.

The script uses `torch.load`, which can execute pickle payloads. Only provide
trusted checkpoints.

## Greedy evaluation adapter

For native RT-DETR evaluation, provide `--eval-spec`. For each individual or
candidate, the CLI:

1. deep-copies the YAML document;
2. sets nested `evaluate.checkpoint` to the candidate path;
3. sets top-level `results_dir` to an isolated evaluation directory;
4. writes `evaluate.yaml`;
5. runs `rtdetr evaluate -e evaluate.yaml` without a shell;
6. extracts the requested finite metric from JSON output.

If the evaluator writes a known relative JSON path, specify it with
`--metric-file-relative`. Otherwise the reader checks `status.json` files
first, followed by other JSON files, and uses the last occurrence of the exact
metric key. A dotted key such as `kpi.mAP50` follows nested mappings.

An alternate evaluator can be supplied as JSON argv, for example:

```bash
--eval-command-json '["python", "/inputs/evaluate.py", "--checkpoint", "{checkpoint}", "--output-dir", "{results_dir}"]'
```

The executable must write the selected metric as JSON beneath `results_dir`.
The CLI does not invoke a shell and does not interpolate environment variables.

## Platform launch shape

Use the platform selected by the user and follow that platform skill's
Preflight plus `tao-launch-workflow`. Resolve `tao_toolkit.pyt` from
`versions.yaml`; the RT-DETR evaluator and PyTorch checkpoint loader must run
in the same container. Mount or stage:

- this skill's `scripts/tao_model_soup.py` read-only;
- all candidate checkpoints read-only;
- the KPI dataset and evaluation spec read-only for greedy mode;
- the job-record `results_dir` read-write as `--output-dir`.

One CPU-only allocation can produce a uniform soup. Greedy RT-DETR evaluation
requires one GPU unless the selected evaluator explicitly supports CPU. The
whole greedy search may run in one allocation; it remains one platform job and
one job-record, while every evaluation receives an isolated result folder.

## Output and recovery

The final checkpoint is written atomically. `soup_manifest.json` is written
only after final hashing succeeds. Candidate checkpoints are temporary unless
`--keep-candidates` is set; evaluation specs, logs, and score records remain
for audit.

On evaluator failure, inspect that candidate's `evaluation.log`, correct the
dataset/spec/runtime problem, and start a new platform job-record. The CLI
refuses to replace an existing final checkpoint unless `--overwrite` was
explicitly passed. Never reinterpret a partial output directory as complete
without a valid manifest and matching final SHA-256.
