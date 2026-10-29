"""Chained IK replay with ideal joint tracking and fixed recorded policy goals.

New solutions become the next feedback AND branch anchor. Recorded hold gaps
start new segments from recorded feedback. A failed segment stops; no failed
target is silently skipped. This is not a robot/policy closed-loop simulation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.client import PiperContinuousIK, project_delivery_target_step, rotation_from_state
from bimanual_vla.deployment.ik_workers import IKUnreachableError
from .postfix_tracking_audit import OUT


def main():
    recorded=json.loads((OUT/'postfix_tracking_results.json').read_text())['samples']
    segments=[]
    for row in recorded:
        if not segments or row['t']-segments[-1][-1]['t']>.15:
            segments.append([])
        segments[-1].append(row)
    solver=PiperContinuousIK()
    reports={}
    for name,tolerance,minimum in [('baseline',.0015,False),('tolerance_only',.003,False),('minimum_motion',.003,True)]:
        arms={}
        for side in ('left','right'):
            rows=[];failures=[]
            for segment_index,segment in enumerate(segments):
                q=np.array(segment[0]['arms'][side]['measured_joints']); previous=None
                for row in segment:
                    original=np.array(row['arms'][side]['target'])
                    xyz,rot=solver.pose(q)
                    current=np.r_[xyz,rot[:,0],rot[:,1],0.]
                    goal,_=project_delivery_target_step(current,original,max_translation_step_m=.05,max_rotation_step_rad=.18)
                    try:
                        result=solver.solve_with_diagnostics(q,goal[:3],Rotation.from_matrix(rotation_from_state(goal.copy())).as_euler('xyz',degrees=True),
                            initial_joints_rad=previous,max_joint_step_rad=.3,search_joint_radius_rad=.3,
                            position_tolerance_m=tolerance,rotation_tolerance_rad=.02,max_nfev=100,minimum_motion=minimum)
                    except IKUnreachableError as exc:
                        failures.append(dict(segment=segment_index,t=row['t'],error=str(exc)))
                        break
                    solved=result.solution_joints_rad
                    px,pr=solver.pose(solved)
                    rows.append(dict(t=row['t'],segment=segment_index,generation=row['generation'],index=row['index'],
                        joints=solved.tolist(),step_deg=float(np.rad2deg(np.max(abs(solved-q)))),
                        position_error_mm=float(np.linalg.norm(px-original[:3])*1000),
                        rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(rotation_from_state(original.copy())@pr.T).magnitude())),
                        relaxed=result.tolerance_relaxed,method=result.selection_method,
                        elapsed_ms=result.solve_elapsed_s*1000,timed_out=result.minimum_motion_timed_out))
                    q=solved.copy();previous=q.copy()
            arms[side]=dict(rows=rows,failures=failures,
                summary=dict(accepted=len(rows),failures=failures,relaxed=sum(x['relaxed'] for x in rows),
                    minimum_motion=sum(x['method']=='minimum_motion' for x in rows),timed_out=sum(x['timed_out'] for x in rows),
                    **{k:np.percentile([x[k] for x in rows],[50,95,100]).tolist() for k in
                       ('step_deg','position_error_mm','rotation_error_deg','elapsed_ms')}))
        reports[name]=arms
    result=dict(profiles=reports,
        limitation='Fixed recorded targets, ideal tracking of newly solved joints. No camera/policy feedback, actuator dynamics, collisions or new model observations simulated.')
    (OUT/'continuity_chain_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{side:arm['summary'] for side,arm in arms.items()} for name,arms in reports.items()},indent=2))


if __name__=='__main__':
    main()
