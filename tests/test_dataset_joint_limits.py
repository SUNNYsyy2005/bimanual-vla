import unittest
import numpy as np
from bimanual_vla.data.episode_analysis import analyze_episode, analysis_payload, frame_payload

class JointLimitsTest(unittest.TestCase):
    def test_initial_negative_j2_and_target_only_violation(self):
        q=np.zeros((100,14));a=q.copy();q[0,1]=np.deg2rad(-.02);a[53,11]=np.deg2rad(70.5)
        names=[f'{side}_joint_{j}_rad' if j<7 else f'{side}_gripper_opening_fraction' for side in ('left','right') for j in range(1,8)]
        analysis=analyze_episode(q,a,fps=30,robot_type='piper_bimanual',arm_base_offset=(.8,0,0),state_names=names)
        report=analysis_payload(analysis,max_points=2)
        self.assertEqual(report['joint_limit_frames'],2)
        self.assertEqual(report['joint_state_limit_frames'],1)
        self.assertEqual(report['joint_target_limit_frames'],1)
        self.assertEqual([r['start_frame'] for r in report['joint_limit_runs']],[0,53])
        frame=frame_payload(analysis,0)
        self.assertIn('joint_state_limit',frame['anomaly_types'])
        self.assertAlmostEqual(frame['joints'][1]['state_limit_excess_deg'],.02)
        self.assertAlmostEqual(frame_payload(analysis,53)['joints'][11]['target_limit_excess_deg'],.5)

    def test_limits_include_boundaries_and_ignore_gripper(self):
        q=np.array([[-150,0,-170,-100,-70,-180,0],[150,180,0,100,70,180,0]],dtype=float)
        q[:,:6]=np.deg2rad(q[:,:6]);q[:,6]=99
        report=analysis_payload(analyze_episode(q,q,robot_type='piper',fps=30))
        self.assertEqual(report['joint_limit_frames'],0)
        self.assertTrue(report['joint_limits_checked'])

    def test_other_robot_and_eef_are_not_joint_angles(self):
        for robot,width in [('franka',7),('piper',20)]:
            report=analysis_payload(analyze_episode(np.zeros((3,width)),fps=30,robot_type=robot))
            self.assertFalse(report['joint_limits_checked'])

    def test_delta_action_is_not_checked_as_absolute_target(self):
        q=np.zeros((3,7));a=np.full((3,7),10.)
        report=analysis_payload(analyze_episode(q,a,robot_type='piper',fps=30,action_is_absolute=False))
        self.assertEqual(report['joint_limit_frames'],0)
        self.assertFalse(report['joint_targets_checked'])
