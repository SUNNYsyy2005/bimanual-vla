"""Replay recorded selected joint targets; no robot I/O or model inference.

All methods receive identical causal targets on the recorded command timeline.
Feedback is ideal generated position, so this evaluates command trajectories,
not physical stability, torque limits, or closed-loop task success.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from bimanual_vla.deployment.client import JOINT_LIMITS_RAD
from bimanual_vla.deployment.servo import RuckigJointServo, ServoConfig, ServoError

DT = .01
JOINTS = [i for i in range(14) if i % 7 != 6]
LOWER = np.tile(JOINT_LIMITS_RAD[:, 0], 2)
UPPER = np.tile(JOINT_LIMITS_RAD[:, 1], 2)
# Deliberately loose ablations are OFFLINE comparisons, not deployable defaults.
PROFILES = {
    'raw_hold': None,
    'filter_20ms': .020,
    'filter_40ms': .040,
    'ruckig_default': (.30, .80, 4.),
    'ruckig_recorded_caps': (.45, 2.5, 4.),
    'ruckig_default_a_relaxed': (.30, 3., 4.),
    'ruckig_default_j_relaxed': (.30, .80, 60.),
    'ruckig_responsive': (.60, 3., 20.),
    'ruckig_fast': (.90, 6., 60.),
    'ruckig_responsive_zero_v': (.60, 3., 20., 'zero_velocity'),
    'ruckig_no_v': (1000., 3., 20.),
    'ruckig_no_a': (.60, 1000., 20.),
    'ruckig_no_j': (.60, 3., 1e6, 'unbounded_jerk'),
}


def load_segments(root: Path):
    selected = []
    for path in sorted(root.glob('20261008*/trajectory.jsonl')):
        segments, current = [], []
        for line in path.open():
            row = json.loads(line)
            action = row.get('command_action')
            stamp = row.get('command_monotonic_timestamp')
            valid = (row.get('command_sent') and not row.get('command_hold')
                     and isinstance(action, list) and len(action) == 14
                     and np.isfinite(np.asarray(action, dtype=float)).all()
                     and isinstance(stamp, (int, float)))
            gap = current and valid and not 0 < stamp - current[-1]['command_monotonic_timestamp'] <= .125
            if not valid or gap:
                if current:
                    segments.append(current)
                current = []
            if valid:
                current.append(row)
        if current:
            segments.append(current)
        if not segments:
            continue
        rows = max(segments, key=len)
        if len(rows) < 100:
            continue
        times = np.array([r['command_monotonic_timestamp'] for r in rows])
        times -= times[0]
        raw = np.array([r['command_action'] for r in rows])[:, JOINTS]
        # Same mandatory physical-position projection for every method. It is
        # counted, not silently confused with velocity/acceleration smoothing.
        targets = np.clip(raw, LOWER, UPPER)
        initial_raw = np.asarray(rows[0]['qpos'])[JOINTS]
        initial = np.clip(initial_raw, LOWER, UPPER)
        selected.append(dict(run=path.parent.name, times=times, targets=targets,
            initial=initial, generations=np.array([r['command_generation'] for r in rows]),
            projected_rows=int(np.any(raw != targets, axis=1).sum()),
            max_projection_rad=float(np.max(np.abs(raw-targets))),
            initial_projection_rad=float(np.max(np.abs(initial-initial_raw)))))
    return selected


def replay(segment, profile):
    events, targets = segment['times'], segment['targets']
    ticks = np.arange(0., events[-1], DT)
    indexes = np.searchsorted(events, ticks, side='right') - 1
    reference = targets[indexes]
    q = segment['initial'].copy()
    output = np.zeros_like(reference)
    native_v, native_a = [], []
    solver = None
    if isinstance(profile, tuple):
        config = ServoConfig(max_velocity=profile[0], max_acceleration=profile[1], max_jerk=profile[2])
        solver = RuckigJointServo(q, LOWER, UPPER, config)
        if len(profile)>3 and profile[3]=='unbounded_jerk':
            # Ruckig's acceleration-limited second-order solver; true ablation.
            solver.input.max_jerk = [float('inf')]*12
    elif isinstance(profile, float):
        # Continuous critically damped third-order filter; no derivative caps.
        tau = profile
        augmented = np.zeros((4, 4))
        augmented[:3, :3] = [[0, 1, 0], [0, 0, 1], [-1/tau**3, -3/tau**2, -3/tau]]
        augmented[2, 3] = 1/tau**3
        discrete = expm(augmented*DT)
        state = np.zeros((3, 12)); state[0] = q
    previous = -1
    try:
        for k, index in enumerate(indexes):
            if solver is not None:
                if index != previous:
                    velocity = np.zeros(12)
                    if previous >= 0 and not (len(profile)>3 and profile[3]=='zero_velocity'):
                        elapsed = events[index] - events[previous]
                        if 0 < elapsed <= config.target_timeout_s:
                            velocity = np.clip((targets[index]-targets[previous])/elapsed,
                                               -config.max_velocity, config.max_velocity)
                    solver.retarget(targets[index], velocity)
                q = solver.update(solver.position)
                native_v.append(solver.velocity.copy()); native_a.append(solver.acceleration.copy())
            elif isinstance(profile, float):
                state = discrete[:3, :3] @ state + discrete[:3, 3, None] * targets[index]
                q = state[0].copy()
            else:
                q = targets[index].copy()
            output[k] = q
            previous = index
    except (ServoError, ValueError) as exc:
        return {'failure': str(exc), 'failure_time_s': float(ticks[k])}
    if solver is not None:
        v, a = np.asarray(native_v), np.asarray(native_a)
        assert np.max(np.abs(v)) <= profile[0] + 1e-6
        assert np.max(np.abs(a)) <= profile[1] + 1e-6
        if not (len(profile)>3 and profile[3]=='unbounded_jerk'):
            assert np.max(np.abs(np.diff(a, axis=0))) / DT <= profile[2] + 1e-4
    # Identical finite-difference operator for ALL methods. Raw hold derivative
    # spikes are discretized at 100 Hz, not actual motor acceleration.
    velocity = np.diff(output, axis=0)/DT
    acceleration = np.diff(velocity, axis=0)/DT
    jerk = np.diff(acceleration, axis=0)/DT
    warmup = 100  # Exclude first second for every method, including baseline.
    error = output[warmup:] - reference[warmup:]
    active = np.std(reference, axis=0) > .02
    lag = None
    if active.any():
        # Retrospective alignment metric only; never used by the causal replay.
        scores = [np.mean((output[warmup+d:len(output)-100+d, active]
                          - reference[warmup:len(output)-100, active])**2) for d in range(101)]
        lag = int(np.argmin(scores))*10
    samples = dict(error=error.ravel(), peak_error=np.max(np.abs(error),axis=1),
                   velocity=velocity[warmup:].ravel(), acceleration=acceleration[warmup:].ravel(),
                   jerk=jerk[warmup:].ravel())
    return dict(times=ticks+DT, q=output, reference=reference, samples=samples, lag_ms=lag,
                bounds_violations=int(np.any((output<LOWER-1e-9)|(output>UPPER+1e-9),axis=1).sum()))


def metrics(samples):
    return dict(rmse_rad=float(np.sqrt(np.mean(samples['error']**2))),
        worst_joint_error_p95_rad=float(np.percentile(samples['peak_error'],95)),
        velocity_p99_rad_s=float(np.percentile(np.abs(samples['velocity']),99)),
        acceleration_p99_rad_s2=float(np.percentile(np.abs(samples['acceleration']),99)),
        jerk_p99_rad_s3=float(np.percentile(np.abs(samples['jerk']),99)),
        jerk_rms_rad_s3=float(np.sqrt(np.mean(samples['jerk']**2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('deployment_runs'))
    parser.add_argument('--output', type=Path, default=Path('analysis/servo_replay_20261019'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    segments = load_segments(args.root)
    if not segments:
        raise RuntimeError('no usable continuous joint segments')
    summaries, records, plotted = [], [], {}
    # Display a high-jump run and a late run, selected before comparing methods.
    high = max(segments, key=lambda s: np.percentile(np.abs(np.diff(s['targets'],axis=0)),99))['run']
    late = next(s['run'] for s in reversed(segments) if len(s['times'])>300)
    for name, profile in PROFILES.items():
        successful=[]; failures=[]; lags=[]; violations=0
        for segment in segments:
            result = replay(segment, profile)
            if 'failure' in result:
                failures.append(dict(run=segment['run'], **result));continue
            successful.append(result['samples']); violations+=result['bounds_violations']
            if result['lag_ms'] is not None:lags.append(result['lag_ms'])
            records.append(dict(run=segment['run'], profile=name, lag_ms=result['lag_ms'], sample_count=len(result['samples']['error']), **metrics(result['samples'])))
            if segment['run'] in (high,late):
                plotted[(segment['run'],name)] = result
        pooled = {key:np.concatenate([s[key] for s in successful]) for key in successful[0]} if successful else None
        summary=dict(profile=name, parameters=profile, successful_runs=len(successful), failures=failures,
            bounds_violations=violations, median_alignment_lag_ms=float(np.median(lags)) if lags else None,
            **(metrics(pooled) if pooled else {}))
        summaries.append(summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='failures'}),flush=True)
    manifest=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()
               if k not in ('targets','initial','times','generations')} |
              dict(rows=len(s['times']),duration_s=float(s['times'][-1]),
                   chunk_switches=int(np.count_nonzero(np.diff(s['generations'])))) for s in segments]
    common = set.intersection(*[{r['run'] for r in records if r['profile']==name} for name in PROFILES])
    for summary in summaries:
        paired=[r for r in records if r['run'] in common and r['profile']==summary['profile']]
        summary['paired_common_runs']=len(common)
        summary['paired_rmse_rad']=float(np.sqrt(sum(r['rmse_rad']**2*r['sample_count'] for r in paired)/sum(r['sample_count'] for r in paired)))
        summary['paired_median_run_jerk_p99']=float(np.median([r['jerk_p99_rad_s3'] for r in paired]))
        summary['paired_median_alignment_lag_ms']=float(np.median([r['lag_ms'] for r in paired if r['lag_ms'] is not None]))
    (args.output/'results.json').write_text(json.dumps(dict(dt=DT,segments=manifest,profiles=summaries,per_run=records,common_runs=sorted(common)),indent=2))
    with (args.output/'per_run.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Same representative joint for every method in a run, chosen by raw jump.
    colors={'ruckig_default':'#3973ac','ruckig_responsive':'#cf6a26','ruckig_fast':'#319a72','filter_20ms':'#9d58ae'}
    for run in (high,late):
        raw=plotted[(run,'raw_hold')]
        joint=int(np.argmax(np.percentile(np.abs(np.diff(raw['reference'],axis=0)),99,axis=0)))
        center=int(np.argmax(np.abs(np.diff(raw['reference'][:,joint]))))
        left=max(1.,raw['times'][center]-2);right=min(raw['times'][-1],left+6)
        fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
        for name in ['raw_hold',*colors]:
            result=plotted.get((run,name))
            if result is None:continue
            q=result['q'][:,joint];t=result['times'];kw=dict(color=colors.get(name,'#777777'),label=name,alpha=.85)
            axes[0].plot(t,q,**kw)
            axes[1].plot(t[2:],np.diff(q,n=2)/DT**2,**kw)
            axes[2].plot(t[3:],np.diff(q,n=3)/DT**3,**kw)
        axes[0].set_title(f'{run} | joint {joint+1} | causal offline replay')
        for ax,label in zip(axes,['Position (rad)','Acceleration (rad/s²)','Jerk (rad/s³)']):
            ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_xlim(left,right)
        axes[1].set_yscale('symlog',linthresh=1);axes[2].set_yscale('symlog',linthresh=10)
        axes[0].legend(ncol=3,fontsize=9);axes[-1].set_xlabel('Recorded segment time (s)')
        fig.savefig(args.output/f'{run}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    for s in summaries:
        if 'rmse_rad' not in s:continue
        key=s['profile']; x=s['paired_rmse_rad']; y=s['paired_median_run_jerk_p99']
        ax.scatter(x,y,s=50)
        offsets={'ruckig_no_v':(6,24),'ruckig_no_a':(6,-20),'ruckig_recorded_caps':(6,18),
                 'ruckig_default_a_relaxed':(6,-18),'ruckig_responsive_zero_v':(-85,15)}
        ax.annotate(key.replace('ruckig_',''),(x,y),xytext=offsets.get(key,(6,4)),textcoords='offset points',fontsize=8,
                    arrowprops=dict(arrowstyle='-',color='#aaaaaa',lw=.6))
    ax.set_yscale('log');ax.set_xlabel('Tracking RMSE (rad), lower is better');ax.set_ylabel('Median run P99 |jerk| (rad/s³), lower is smoother')
    ax.set_title(f'Recorded joint targets: smoothness versus tracking error\nSame {len(common)} runs for every method; failures reported separately');ax.grid(alpha=.25)
    fig.savefig(args.output/'tradeoff.png',dpi=160)


if __name__=='__main__':
    main()
