# Runtime and launch gates

The runner stages COCO images and YOLO labels into `runtime.scratch_root`.
Clean negatives deliberately receive empty label files. Evaluation preserves
COCO image ids so predictions and KITTI label stems remain aligned with the
shared DEFT gap matcher.

On SLURM, require scratch below `/raid/scratch`. Copy custom scripts and the
hot working set there before execution. Put `TMPDIR`, `XDG_CACHE_HOME`,
`HF_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME`, Triton, TorchInductor, matplotlib,
and compiler caches under the same job-local root. Never execute custom code,
an environment, or a database from Lustre.

The rendered batch script must record allocation/start, staging-complete,
workload-start/end, and copy-back-complete timestamps. Its `EXIT` trap must
preserve the workload exit code and copy back only the allowlisted outputs.
Refuse the launch if `/raid/scratch` is unavailable or too small.

Edge-AI training additionally requires verified OneLogger login material,
application callbacks, and job enablement. `logging.onelogger_enabled: true`
is an assertion made during launch review, not proof by itself. If the callback
module is not packaged and staged, stop before submit and direct the user to
`#aidot-onelogger-e2e-support`.

The initial image is `docker.io/ultralytics/ultralytics:8.4.131`. Confirm the
image and applicable Ultralytics/YOLO license terms during launch review. This
skill and its integration branch stay local until redistribution is approved.
