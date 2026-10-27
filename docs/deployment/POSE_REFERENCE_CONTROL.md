# End-effector online reference shaping

The new `--pose-trajectory` option smooths **delivery policy execution targets**
before continuous IK. It is opt-in (CLI and GUI default off). In the GUI, open
**Action continuity settings → End-effector reference** and enable **Smooth
end-effector targets before IK** with the Ruckig joint planner selected.
The Joint reference planner tab still controls the downstream physical joint
speed/acceleration/jerk bounds. Neither setting starts or enables the robot.

## What questVR actually does

Source reviewed: [agilexrobotics/questVR_ws](https://github.com/agilexrobotics/questVR_ws),
commit `393a32b26dfe2b9f4151f3cdfcfe9ba4d56d3a78` (master on 2026-10-20).
Paths below are relative to that repository:

| Stage | Source | Observed behavior |
| --- | --- | --- |
| VR input | `src/oculus_reader/scripts/oculus_reader.py:51,190` | Background ADB logcat reader stores the latest sample. No temporal smoothing was found in this Python path; headset/APK filtering is unknown. |
| Pose publication | `src/oculus_reader/scripts/pub_pose.py:13–18,131–144` | ROS publishers use queue size 1 and a 100 Hz loop, sampling the latest VR pose. This is a configured publication rate, not proof of 100 Hz fresh sensor samples or IK solutions. |
| Coordinate mapping | `src/oculus_reader/scripts/teleop_single_piper.py:63–70,304,322–342` | Queue size 1; maps the pose relative to the A-button origin into an absolute robot pose using SE(3), rather than summing successive deltas. |
| IK | `src/oculus_reader/scripts/teleop_single_piper.py:178–243` | CasADi/Pinocchio SE(3) residual, position weight 1 and orientation weight 0.1 inside the squared residual; objective `20 * task_cost + 0.01 * sum(q²)`. Joint bounds and previous-solution warm start. The optional last-solution smoothness objective is commented out. |
| Joint output | `src/oculus_reader/scripts/piper_control.py:158–170` | Publishes each IK result immediately. The 50 Hz linear interpolation over 0.5 s in `init_pose` is for homing, not ordinary teleoperation. |
| Driver | `src/Piper_ros/src/piper/scripts/piper_ctrl_single_node.py:108,350–375` | Initial `MotionCtrl_2(0x01,0x01,30,0xad)`; ordinary empty-velocity JointState callback uses `MotionCtrl_2(0x01,0x01,50,0xad)` followed by `JointCtrl`. |
| Feedback | Same driver, line 129 | 200 Hz is the ROS feedback publication loop. It is not the command/IK publication rate. |

The plausible contributors to smooth behavior are fresh, naturally continuous
human pose inputs, bounded/warm-started IK and driver tracking settings. There
is no evidence in these files of a complete online Cartesian trajectory servo.
The upstream code's 30-degree jump handling resets the **next IK initial guess**
to zero; that is not a bounded motion or stop mechanism. Its collision-return
handling also does not establish a command gate equivalent to this project's
existing safety checks. Those behaviors are not copied.

This project uses an existing guarded 100 Hz Ruckig joint writer and a default
Piper speed percentage of 10, while questVR's usual callback requests 50 and
uses a different fourth motion-control argument (`0xad` versus `0x00`). The
exact firmware/API meaning and effects require hardware-specific verification.
This change does not alter either value or claim that changing them is safe.
Increasing publication rate alone does not remove a VLA model's low-frequency
chunk discontinuities or camera/control latency.

## Implementation and limits

`pose_trajectory.py` maintains translation plus base-frame linear/angular
velocity across new policy targets. It uses the shortest SO(3) rotation error
and integrates rotations on SO(3), avoiding Euler wrap and invalid rotation6d
interpolation. Euclidean vector speed and acceleration bounds apply before IK.
Defaults are 0.20 m/s, 1.20 m/s², 0.75 rad/s, 8.0 rad/s², and a 0.06 s response
time. The original continuous IK retains measured-joint regularization, a
previous-solution warm start, joint bounds and residual checks. Ruckig owns
joint jerk bounds and the sole CAN writer; the pose layer has no actuator thread.

This is a **causal online reference shaper at the client reference rate**, not
whole-chunk Cartesian planning or a Cartesian jerk guarantee. Braking after a
target reversal can overshoot that target; every shaped position is checked
against the configured workspace before IK. This is not collision avoidance.
Feedback initializes a new reference and subsequently checks tracking (defaults
0.10 m / 0.80 rad); it does not reset the trajectory state each tick. Clock gaps
over 0.15 s fail closed. Holds, faults, protocol changes and shutdown discard
the pose state; resumed motion initializes from fresh feedback and remains
subject to the existing writer's safe-stop/ownership rules.

Both arms finish target checks and IK before a writer submission. If either arm
fails, the shared failure path discards both pose states and holds the writer.
Pose shaping is only called in the delivery branch. The joint RTC planner-prefix
gate remains joint-only, and delivery action/state dimensions and chunk-origin
delta conventions are unchanged.

Shaping adds execution delay. It does not make the policy's RTC prefix reflect
the shaped motion. Increasing response time or tightening limits may worsen
tracking and wrist-camera visibility; the defaults are starting limits, not
validated task-optimal settings. Logs record the original absolute policy
target, shaped target, translation/rotation lag, velocities, accelerations,
tracking deviation, limiting flags, and the existing IK residuals in each arm's
`command_pipeline.pose_trajectory` / `ik_diagnostics` data. Use these together
with measured feedback, timestamps and images when assessing grasp accuracy.
When deployment recording is active, `trajectory.jsonl` also persists these
under each arm's `pose_diagnostics` independently of optional monitoring.

### Parameter selection and when defaults take effect

The 2026-10-20 preset was selected by staged offline replay of 25 `fix`
demonstrations (12067 frames, 20 Hz) and independent seeded-noise, reversal and
single-frame pulse tests. This preserves one shared response time for
translation/rotation and all existing tracking/workspace limits.

| Reference deviation | Previous preset | Selected preset |
| --- | --- | --- |
| Position P95 / P99 / max | 63.25 / 117.12 / 168.37 mm | 1.99 / 5.94 / 35.41 mm |
| Rotation P95 / P99 / max | 2.14 / 3.27 / 13.85 deg | 0.50 / 0.91 / 10.61 deg |

These are ideal-feedback FK reference deviations, not actual grasp errors,
time delays or closed-loop success rates. No IK, workspace rejection, robot or
camera is simulated. The remaining worst cases (35.41 mm / 10.61 deg) matter.
The 0.05 s response alternative slightly reduces median/P95 reference error but
amplifies the seeded 20 Hz white-noise RMS to 1.178 / 1.124 times the input;
the selected 0.06 s response yields 0.917 / 0.906. A +/-50 mm / +/-0.3 rad step
reversal still overshoots by 9.50 mm / 1.03 deg. A one-frame 0.15 m / 1 rad
pulse passes approximately 3.00 mm / 1.15 deg into the reference. This is a
measured compromise, not a universal stability/collision guarantee.

The preset takes effect in a **new CLI process** when its individual pose
arguments are not explicitly overridden, and in a **new GUI process** for
preference keys that are absent. Existing saved GUI values are deliberately
preserved; enter the values listed above in the End-effector reference tab to
use this preset there. Running processes do not hot-reload these defaults.
The feature still requires explicit enablement and the Ruckig backend and only
affects delivery policies. It does not change today's joint-policy execution,
Piper speed percentage, or the joint planner's own speed/acceleration/jerk
limits. No hardware or GUI restart is performed by this parameter change.

The existing shared delivery workspace defaults are X=(-0.05,0.30),
Y=(0.01,0.50), Z=(0.14,0.52) metres. These are not automatically suitable for
both arms: after-17:00 joint recordings include right-arm SDK FK Y below zero.
Enabling delivery can therefore legitimately reject those right-arm poses.
Check each arm's actual base coordinate convention and configured safe workspace
before a real delivery run. This patch does not widen bounds or infer a mirror
transform. FK geometry replay only counts these workspace rejections; it does
not bypass them in the real controller and is not full safety-path acceptance.

## Offline verification

Run with the robot Conda environment, without CAN access:

```bash
PYTHONPATH=.:tests /home/user/miniconda3/envs/dual_arm/bin/python -m unittest \
  test_pose_trajectory test_servo test_robot_observation_bridge \
  test_collect_gui_dataset_tools -q
```

Coverage includes axis-changing rotations near pi, +/-179 degree wrap,
target reversals, variable update intervals, invalid/stale clocks, workspace
and feedback guards, restart/hold behavior, dual-arm IK failure atomicity,
protocol changes, diagnostics and GUI command construction, plus existing
writer/IK/GUI regression tests. Tests use simulated publishers, never CAN.
Today's selected recordings after 17:00 are joint-mode runs; deriving poses
using forward kinematics is only a geometric stress replay, not a real
delivery-policy evaluation or a closed-loop success-rate measurement.
