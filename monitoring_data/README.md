# Local monitoring recordings

Each `rtc_client.py` run creates a timestamped directory containing:

- `manifest.json`: command-line arguments, host, PID, and recording format.
- `events.jsonl`: append-only events for Piper connection, camera/policy setup,
  every inference result, every 20 Hz control tick, and blocked/error events.

The control-tick rows include measured `qpos_m`, delivery state, execution
metadata, and the correlated `generation`, `source_index`, `queue_index`, IK
diagnostics, Piper integer commands, next-cycle feedback, and command-following
errors. Images are intentionally not copied into this log; camera device and
capture timestamps are retained.

The default location is `./monitoring_data`. Override it with
`--monitoring-dir /path/to/root` or `BIMANUAL_VLA_MONITORING_DIR`.

`robot_observation_bridge.py` remains a compatibility launcher for legacy deployments.
