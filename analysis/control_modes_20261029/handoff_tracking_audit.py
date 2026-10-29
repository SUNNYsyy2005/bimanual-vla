"""Read-only audit of the user-confirmed 11:05:57 handoff run; no hardware access."""
import json
from pathlib import Path

import numpy as np

from bimanual_vla.data.action_conventions import chunk_origin_deltas_to_absolute_eef_targets
from analysis.control_modes_20261029.postfix_tracking_audit import error

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "deployment_runs/20261029T110557.711109+0800_798395"


def main():
    trajectory, models = ([json.loads(line) for line in (RUN / name).open()]
                          for name in ("trajectory.jsonl", "model_commands.jsonl"))
    t0, end = trajectory[0]["timestamp"], trajectory[-1]["timestamp"]
    active = [r for r in trajectory if r["command_sent"] and not r["command_hold"]
              and r["command_absolute_target"] is not None]
    late = [r for r in active if r["timestamp"] >= end - 10]
    late_all = [r for r in trajectory if r["timestamp"] >= end - 10]
    chunks, future, prefix_errors = {}, [], []
    for record in models:
        with np.load(RUN / record["action_file"]) as data:
            state = data["raw_delivery_state"]
            chunk = chunk_origin_deltas_to_absolute_eef_targets(
                state, data["raw_actions"], arm_count=2)
        chunks[record["generation"]] = chunk
        rtc = record["rtc"]
        if rtc.get("enabled") and rtc["old_generation"] in chunks:
            n, offset = rtc["inference_delay_steps"], rtc["previous_chunk_offset_steps"]
            old = chunks[rtc["old_generation"]][offset:offset+n]
            prefix_errors.extend(error(a[10:20], b[10:20]) for a, b in zip(old, chunk[:n]))
        if record["captured_at"] < end - 10:
            continue
        # Steps >= execution_horizon are beyond the linear RTC guidance window.
        fresh_start = max(rtc.get("execution_horizon", 0), rtc.get("inference_delay_steps", 0))
        distances = np.linalg.norm(chunk[:, 10:13] - state[10:13], axis=1)*1000
        future.append(dict(generation=record["generation"], accepted=record["accepted"],
            t=record["captured_at"]-t0, fresh_start=fresh_start,
            observed_xyz=state[10:13].tolist(), last_xyz=chunk[-1, 10:13].tolist(),
            whole_chunk_max_displacement_mm=float(max(distances)),
            fresh_suffix_max_displacement_mm=float(max(distances[fresh_start:])),
            last_displacement_mm=float(distances[-1]),
            image_age_ms={k:(record["captured_at"]-v)*1000
                          for k,v in record["image_timestamps"].items()}))
    right_errors = [error(r["delivery_state"][10:20], r["command_absolute_target"][10:20]) for r in late]
    samples = []
    for t in (5, 10, 15, 20, 25, 30, 35, end-t0):
        r = min(active, key=lambda r:abs(r["timestamp"]-t0-t))
        samples.append(dict(t=r["timestamp"]-t0, measured_xyz=r["delivery_state"][10:13],
            target_xyz=r["command_absolute_target"][10:13],
            error_mm_deg=error(r["delivery_state"][10:20],r["command_absolute_target"][10:20])))
    transitions, last = [], None
    for r in trajectory:
        key = (r["execution_state"],r["blocked_reason"])
        if key != last:
            transitions.append(dict(t=r["timestamp"]-t0,state=key[0],reason=key[1]))
            last = key
    report = dict(run=RUN.name,duration_s=end-t0,late_window_s=10,
        late_samples=len(late),late_total_rows=len(late_all),
        late_states=sorted(set(r["execution_state"] for r in late_all)),
        late_hold_rows=sum(bool(r["command_hold"]) for r in late_all),
        late_right_error_p50_p95_max_mm_deg=np.percentile(right_errors,[50,95,100],axis=0).tolist(),
        final_right_joints_deg=np.rad2deg(trajectory[-1]["qpos"][7:13]).tolist(),
        max_right_prefix_error_mm_deg=np.max(prefix_errors,axis=0).tolist(),
        samples=samples,late_model_chunks=future,transitions=transitions)
    (OUT / "handoff_tracking_results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("transitions","late_model_chunks")},indent=2))
    print("Last four chunks:",json.dumps(future[-4:],indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True)
    times = [r["timestamp"]-t0 for r in active]
    for axis in range(3):
        line, = axes[0].plot(times,[r["delivery_state"][10+axis]*1000 for r in active],label="XYZ"[axis]+" measured")
        axes[0].plot(times,[r["command_absolute_target"][10+axis]*1000 for r in active],
                     "--",color=line.get_color(),label="XYZ"[axis]+" policy target")
    axes[1].plot(times,[error(r["delivery_state"][10:20],r["command_absolute_target"][10:20])[0] for r in active])
    axes[0].set_ylabel("Right base-frame position (mm)")
    axes[1].set_ylabel("Right position error (mm)")
    axes[1].set_xlabel("Seconds since first control tick")
    axes[0].legend(ncol=3,fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "handoff_tracking.png",dpi=150)


if __name__ == "__main__":
    main()
