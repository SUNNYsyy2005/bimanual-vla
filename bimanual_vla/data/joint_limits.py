"""Read-only Piper joint-limit diagnostics. Never clamp or reinterpret EEF values."""
import json
import re
from pathlib import Path
import numpy as np

def driver_limit_diagnostics(state, action, names, *, robot_type, action_is_absolute=True):
    state=np.asarray(state); action=np.asarray(action)
    profile=json.loads(Path(__file__).with_name('piper_driver_limits.json').read_text())
    bounds=np.full((state.shape[1],2),np.nan)
    state_excess=np.zeros(state.shape);target_excess=np.zeros(state.shape)
    checked=str(robot_type or '').lower().replace('_','-').startswith('piper') and state.shape[1] in (7,14)
    if checked:
        for col,name in enumerate(names):
            match=re.fullmatch(r'(?:left_|right_)?joint_([1-6])(?:_rad)?',str(name))
            if match:bounds[col]=np.deg2rad(profile['joint_limits'][int(match[1])-1])
        checked=bool(np.isfinite(bounds).any())
    if checked:
        for values,out in ((state,state_excess),(action,target_excess)):
            if out is target_excess and not action_is_absolute:continue
            for col,(lo,hi) in enumerate(bounds):
                if np.isfinite(lo) and col<values.shape[1]:
                    excess=np.maximum(lo-values[:,col],values[:,col]-hi)
                    out[:,col]=np.where(excess>profile['numerical_tolerance_rad'],excess,0)
    mask=np.any(state_excess>0,axis=1)|np.any(target_excess>0,axis=1)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    runs=[dict(start_frame=int(a),end_frame=int(b-1),frame_count=int(b-a)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    return dict(checked=checked,reason=None if checked else 'Requires Piper joint-angle observations; EEF-only or other robot data cannot be checked against this profile',profile=profile,bounds_rad=bounds,state_excess_rad=state_excess,target_excess_rad=target_excess,mask=mask,runs=runs,target_checked=checked and action_is_absolute)

def limit_summary(report):
    return dict(joint_limits_checked=report['checked'],joint_limits_reason=report['reason'],joint_limit_profile=report['profile'],joint_targets_checked=report['target_checked'],joint_limit_frames=int(report['mask'].sum()),joint_state_limit_frames=int(np.any(report['state_excess_rad']>0,axis=1).sum()),joint_target_limit_frames=int(np.any(report['target_excess_rad']>0,axis=1).sum()),joint_limit_runs=report['runs'])

def limit_frame(report,index,col):
    lo,hi=report['bounds_rad'][col];s=float(report['state_excess_rad'][index,col]);a=float(report['target_excess_rad'][index,col])
    return dict(joint_limits_deg=np.rad2deg([lo,hi]).tolist() if np.isfinite(lo) else None,state_limit_violation=s>0,target_limit_violation=a>0,state_limit_excess_deg=float(np.rad2deg(s)),target_limit_excess_deg=float(np.rad2deg(a)))
