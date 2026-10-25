# Single-checkpoint return to 4090

Open **集群检查点 → 4090** in the training page, refresh once, then select a
step. Cached rows remain selectable while discovery runs in the background.
The API uses the existing Dashboard authentication:

- `GET /api/checkpoints/remote?refresh=1&source=<target>` starts discovery.
- `GET /api/checkpoints/remote` reads the cached inventory.
- `POST /api/checkpoints/sync` with `source`, `target: "local_4090"`,
  `config_name`, `exp_name`, and numeric `step` starts a transfer task.

Discovery uses CPU-only Slurm jobs on Slurm-only targets. Small acknowledged
chunks fit the cluster's Comment length limit. Successful inventories persist
across Dashboard restarts; refreshes coalesce for 30 seconds.

For Slurm sources, a CPU job pins inference files with hard links before
packing, so later checkpoint garbage collection cannot remove files mid-copy.
It excludes optimizer state, exports 256 MiB parts with SHA256 hashes, and
uploads through four SSH lanes by default. The receiver starts downloading
published parts while the remaining parts upload. It verifies every part,
rejects archive links and unsafe paths, validates file sizes and checkpoint
metadata, then installs the checkpoint atomically on 4090. The action contract
must match the destination experiment. Failed downloads can resume partial
parts within the same transfer task.

The default staging directory is `checkpoint_exports` beside the source's
`remote_job_dir`; `checkpoint_ssh_staging_root` can override it. Compute nodes
must be able to SSH to the submit host with the existing account key. Packing
and extraction run on compute/4090; the login host only receives and serves
bytes. This avoids the unreliable WebDAV NAS mount. Staged source bundles are
retained for diagnosis; successful 4090 download caches are removed.

The installed `.dashboard-inference.json` declares
`full_state_available: false`. These copies serve policies and evaluate models;
resume training from the full-state cluster checkpoints. Existing full-
experiment sync remains available when no `step` is supplied.

Validation: `python -m pytest -q tests/test_checkpoint_sync.py
tests/test_dashboard_checkpoint_management.py
tests/test_dashboard_template_null_guard.py` (22 passed, 1 skipped locally).
The real joint 20k transfer on 2026-10-25 verified 6,499,355,046 bytes in
741.67 seconds end to end (8.36 MiB/s), with API submission in 0.079 seconds.
End-to-end time includes Slurm scheduling, export, network, verification and
installation; it is not a raw link-bandwidth measurement. The deployed storage
browser also routes remote OpenPI copy actions to this single-step API.
