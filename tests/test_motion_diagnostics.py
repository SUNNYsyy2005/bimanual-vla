import numpy as np
from scipy.spatial.transform import Rotation
from bimanual_vla.data.action_conventions import matrix_to_rotation6d
from bimanual_vla.data.episode_analysis import analyze_episode, analysis_payload, frame_payload, detect_motion_anomalies


def test_right_arm_spike_and_gripper_exclusion():
    q = np.zeros((100,14)); t=np.arange(100)/30
    q[50,10]=.3
    a=analyze_episode(q,timestamps=t)
    groups=analysis_payload(a)['motion_diagnostics']['groups']
    assert groups[0]['abrupt_frames']==0
    assert groups[1]['abrupt_frames']>0
    assert a.abrupt_change[48:53].any()
    q[:]=0; q[::2,6]=1; q[::2,13]=1
    a=analyze_episode(q,timestamps=t)
    assert not a.abrupt_change.any()
    assert not a.velocity_reversal.any()
    assert not a.jitter.any()


def eef_data():
    q=np.zeros((100,20))
    for s in (0,10):
        q[:,s+3:s+9]=matrix_to_rotation6d(np.eye(3))
    names=[f'{side}_{v}' for side in ('left','right') for v in ('eef_x','eef_y','eef_z','r0','r1','r2','r3','r4','r5','gripper')]
    return q,names


def test_eef_separate_translation_rotation_and_wrap():
    q,names=eef_data(); t=np.arange(100)/30
    for i,angle in enumerate(np.linspace(170,190,100)):
        q[i,13:19]=matrix_to_rotation6d(Rotation.from_euler('z',angle,degrees=True).as_matrix())
    a=analyze_episode(q,timestamps=t,state_names=names)
    assert not a.abrupt_change.any()
    assert not a.jitter.any()
    q[50,10]=.1
    a=analyze_episode(q,timestamps=t,state_names=names)
    groups=analysis_payload(a)['motion_diagnostics']['groups']
    assert [(g['kind'],g['unit']) for g in groups]==[('translation','m'),('rotation','rad')]*2
    assert groups[2]['abrupt_frames']>0
    assert groups[0]['abrupt_frames']==groups[1]['abrupt_frames']==groups[3]['abrupt_frames']==0
    assert frame_payload(a,50)['motion_diagnostics']['groups'][2]['acceleration_norm']>0


def test_seventh_joint_and_unknown_schema():
    q=np.zeros((100,16));q[50,14]=.3
    assert analyze_episode(q,fps=30).abrupt_change.any()
    a=analyze_episode(np.zeros((100,20)),fps=30)
    assert not analysis_payload(a)['motion_diagnostics']['checked']


def test_low_level_detector_checks_all_dimensions():
    q=np.zeros((100,7));q[50,6]=.3
    assert detect_motion_anomalies(q,fps=30)['abrupt_change'].any()
