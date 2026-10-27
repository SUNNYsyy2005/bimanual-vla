# PPO training monitoring

The simulation dashboard RLinf module reads GRPO, PPO and their Lagrangian
variants. Runs with `protocol.json` and `config.json` identify their algorithm,
task, Slurm job, planned rounds and evaluation schedule automatically. Register
one experiment family root; new task/stage directories under it are discovered.

```bash
python server_4090/configure_ppo_monitor.py \
  --config server_4090/config.simulation.json \
  --root /DATA/disk0/sunny/experiments/low-sr-ppo-20261022-v1/runs
```

The registration is idempotent and preserves other sources and launch profiles.
This is a monitoring adapter. Training/evaluation launch profiles remain specific
to their actual trainer and checkpoint format; adding a monitored PPO run does
not make it compatible with a GRPO launch command or native RLinf weight loader.

## Display contract

- Training round, actual Actor updates, and last evaluated round are separate.
- Table SR/risk refer to the fixed validation cohort when available. Training SR
  is labeled separately; it never fills an absent evaluation result.
- Charts separate fixed validation and extra development cohorts, even when
  round and sampler are identical. Default horizontal axis is training round.
- Numeric `critic` and `ppo_advantage` diagnostics are exposed as metric curves.
- Eligible evaluation episodes and excluded invalid initializations are shown.
- Protocol/completion records provide status; enabled Slurm queries verify live
  jobs. `%A` is used for the underlying numeric job ID of array tasks, matching
  `SLURM_JOB_ID`; `%i` would return the array expression instead.
- Mirror `exported_unix` is a synchronization timestamp, never a result timestamp.
  Result time is `source_updated_unix` or the newest source-file mtime in
  `provenance`. Completed runs may additionally publish `completed_unix`.
  Older mirrors without source mtimes fall back to their source phase timestamp.

## Verification, 2026-10-22

Registered H100 jobs 46500 (TSA basket), 46501 (TSA microwave), 46499 (FSHOA
microwave), each planned for32 rounds. Live API verified both evaluation cohorts,
PPO critic metrics, actual update counts and the three task records.

H200 GRPO-Lag job45438 completed128 training/evaluation rounds, with127 Actor
updates; Slurm ended at2026-10-22 09:33:05 Asia/Shanghai. Its source completion
file timestamp is stable; ongoing mirror synchronization is displayed separately.
The expired mirror job45539 was replaced with CPU-only mirror job46537 to keep
the still-running H200 PPO experiment current.

Checks: Python dashboard/monitor regression suite21 tests and Node UI suite6
tests, including fixed-vs-extra cohort isolation, zero Actor updates, source
timestamp stability across repeated exports, and preservation of explicit stops.

Deployment updated only the RLinf backend, registration helper and RLinf template
sections. Existing live template changes in other modules were preserved. Only
the simulation dashboard service was restarted; training jobs continue unchanged.
