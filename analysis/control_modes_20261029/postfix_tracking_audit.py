"""Read-only audit of the first robot run after EEF RTC reanchoring repair."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.data.action_conventions import (
    chunk_origin_deltas_to_absolute_eef_targets, rotation6d_to_matrix,
)
from bimanual_vla.deployment.client import PiperContinuousIK

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'deployment_runs/20261029T104330.359218+0800_794465'


def read(name):
    return [json.loads(line) for line in (RUN / name).open()]


def error(state, target):
    return [float(np.linalg.norm(np.asarray(state[:3])-target[:3])*1000),
            float(np.rad2deg(Rotation.from_matrix(rotation6d_to_matrix(target[3:9]) @
                rotation6d_to_matrix(state[3:9]).T).magnitude()))]


def percentiles(values):
    return np.percentile(values, [50, 95, 100], axis=0).tolist()


def main():
    trajectory, servo, models = (read(name) for name in
        ('trajectory.jsonl', 'servo_commands.jsonl', 'model_commands.jsonl'))
    t0 = trajectory[0]['timestamp']
    solver = PiperContinuousIK()
    chunks = {}
    for record in models:
        with np.load(RUN / record['action_file']) as data:
            chunks[record['generation']] = chunk_origin_deltas_to_absolute_eef_targets(
                data['raw_delivery_state'], data['raw_actions'], arm_count=2)
    prefix_errors = []
    for record in models:
        rtc = record['rtc']
        if rtc['enabled']:
            offset, n = rtc['previous_chunk_offset_steps'], rtc['inference_delay_steps']
            old, new = chunks[rtc['old_generation']][offset:offset+n], chunks[record['generation']][:n]
            prefix_errors.append(dict(generation=record['generation'], reanchored=rtc['reanchored'],
                errors={side: np.max([error(a[k:k+10], b[k:k+10]) for a,b in zip(old,new)],axis=0).tolist()
                        for side,k in [('left',0),('right',10)]}))
    references = {}
    for row in servo:
        context = row.get('context') or {}
        key = (context.get('generation'), context.get('queue_index'))
        if not row['braking'] and key not in references:
            references[key] = row
    samples = []
    for row in trajectory:
        key = (row['command_generation'], row['command_queue_index'])
        if not row['command_sent'] or row['command_hold'] or key not in references:
            continue
        ref = references[key]
        item = dict(t=row['timestamp']-t0, generation=key[0], index=key[1], arms={})
        for arm, side in enumerate(('left','right')):
            state = np.array(row['delivery_state'][arm*10:arm*10+10])
            target = np.array(row['command_absolute_target'][arm*10:arm*10+10])
            joints = np.array(ref['reference'][arm*7:arm*7+6])
            xyz, rot = solver.pose(joints)
            reference_pose = np.r_[xyz, rot[:,0],rot[:,1],0.]
            sent = np.array(ref['command_qpos'][arm*7:arm*7+6])
            sent_xyz, sent_rot = solver.pose(sent)
            _, _, jac = solver._kinematics.evaluate(joints, jacobian=True)
            item['arms'][side] = dict(state=state.tolist(), target=target.tolist(),
                reference=joints.tolist(), measured_joints=row['qpos'][arm*7:arm*7+6],
                policy_error=error(state,target), reference_error=error(state, reference_pose),
                sent_error=error(state,np.r_[sent_xyz,sent_rot[:,0],sent_rot[:,1],0.]),
                reference_to_policy_error=error(reference_pose,target),
                jacobian_min_singular=float(np.linalg.svd(jac,compute_uv=False)[-1]),
                diagnostics=row['pose_diagnostics'][side]['ik_diagnostics'])
        samples.append(item)
    changes=[]
    for a,b in zip(samples,samples[1:]):
        if b['t']-a['t']>.15:
            continue
        item=dict(t=b['t'],generation=b['generation'],index=b['index'],dt=b['t']-a['t'],
                  boundary=a['generation']!=b['generation'],arms={})
        for side in ('left','right'):
            before,after=a['arms'][side],b['arms'][side]
            item['arms'][side]=dict(target_step=error(before['target'],after['target']),
                reference_step_deg=np.rad2deg(np.array(after['reference'])-before['reference']).tolist(),
                reference_before_deg=np.rad2deg(before['reference']).tolist(),
                reference_after_deg=np.rad2deg(after['reference']).tolist())
        changes.append(item)
    report=dict(run=RUN.name, time_origin=t0, prefix_errors=prefix_errors, summary={},
                transitions=[], samples=samples, changes=changes)
    last=None
    for row in trajectory:
        key=(row['execution_state'],row['blocked_reason'])
        if key!=last:
            report['transitions'].append(dict(t=row['timestamp']-t0,state=key[0],reason=key[1]))
            last=key
    for side in ('left','right'):
        values=[s['arms'][side] for s in samples]
        final_target=values[-1]['target']
        k=0 if side=='left' else 10
        report['summary'][side]=dict(samples=len(values),
            relaxed=sum(v['diagnostics']['tolerance_relaxed'] for v in values),
            **{key:percentiles([v[key] for v in values]) for key in
               ('policy_error','sent_error','reference_to_policy_error')},
            reference_step_deg=percentiles([max(abs(np.array(c['arms'][side]['reference_step_deg']))) for c in changes]),
            boundary_step_mm=percentiles([c['arms'][side]['target_step'][0] for c in changes if c['boundary']]),
            final_error_to_last_published_goal=error(trajectory[-1]['delivery_state'][k:k+10],final_target),
            measured_joint_range_deg=np.rad2deg([np.min([v['measured_joints'] for v in values],axis=0),
                                               np.max([v['measured_joints'] for v in values],axis=0)]).tolist())
    (OUT/'postfix_tracking_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=report['summary'],guided_prefixes=len(prefix_errors),
        max_prefix_error={side:np.max([p['errors'][side] for p in prefix_errors],axis=0).tolist() for side in ('left','right')},
        largest_left_changes=sorted(changes,key=lambda c:max(abs(np.array(c['arms']['left']['reference_step_deg']))),reverse=True)[:4]),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for col,side in enumerate(('left','right')):
        t=[s['t'] for s in samples]
        v=[s['arms'][side] for s in samples]
        for axis in range(3):
            axes[0,col].plot(t,[x['state'][axis]*1000 for x in v],label='XYZ'[axis]+' measured')
            axes[0,col].plot(t,[x['target'][axis]*1000 for x in v],ls='--',label='XYZ'[axis]+' policy')
        for key in ('policy_error','sent_error','reference_to_policy_error'):
            axes[1,col].plot(t,[x[key][0] for x in v],label=key)
        for joint in (0,3,5):
            axes[2,col].plot(t,[np.rad2deg(x['reference'][joint]) for x in v],label=f'J{joint+1} reference')
            axes[2,col].plot(t,[np.rad2deg(x['measured_joints'][joint]) for x in v],ls='--',label=f'J{joint+1} measured')
        axes[0,col].set_title(side)
        for row in range(3):
            axes[row,col].legend(fontsize=7,ncol=2)
            axes[row,col].grid(alpha=.25)
        axes[0,col].set_ylabel('XYZ (mm)')
        axes[1,col].set_ylabel('Position error (mm)')
        axes[2,col].set_ylabel('Joints (deg)')
        axes[2,col].set_xlabel('Seconds since first control tick')
    fig.tight_layout()
    fig.savefig(OUT/'postfix_tracking.png',dpi=150)


if __name__=='__main__':
    main()
