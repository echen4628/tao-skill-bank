# Runtime and launch gates

The runner stages COCO images and YOLO labels into `runtime.scratch_root`.
Clean negatives deliberately receive empty label files. Evaluation preserves
COCO image ids so predictions and KITTI label stems remain aligned with the
shared DEFT gap matcher.

Use a new writable `runtime.scratch_root` for each action. It must have enough
space for the staged dataset, generated labels, and temporary trainer outputs.

For multi-GPU Ultralytics jobs, also set `YOLO_CONFIG_DIR` beneath the same
job-local root so generated DDP launch files never use the container overlay.
Create a job-private node-local directory with mode `1777` and bind it to
`/dev/shm`; dataloader multiprocessing can otherwise hang during teardown on
the cluster's container-default shared-memory mount.

The runtime image is `docker.io/ultralytics/ultralytics:8.4.131`. Confirm the
exact image during launch review.
