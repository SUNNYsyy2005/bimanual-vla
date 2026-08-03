# Piper Data Contract

This file defines the boundary between collection clients, dataset conversion,
π0.5 training, and real-robot inference. UI implementations must call the
shared backend instead of constructing actions or NPZ payloads themselves.

## UI Mapping

| UI event | Backend operation |
|---|---|
| Connect | `CollectionSession.connect()` |
| Start episode | `start_episode(task_name, instruction)` |
| Capture tick | `capture_once()` |
| Stop episode | `stop_episode()` |
| Save success/failure | `save_episode(success=...)` |
| Discard | `discard_episode()` |
| Disconnect | `disconnect()` |

## Supported Contracts

| arm mode | schema | state | action | cameras |
|---|---|---:|---:|---|
| single | `joint` | 7D | 7D | `cam_high` + one wrist camera |
| bimanual | `joint` | 14D | 14D | `cam_high` + `cam_left_wrist` + `cam_right_wrist` |
| single | `delivery` | 10D | 7D | `cam_high` + one wrist camera |
| bimanual | `delivery` | 20D | 14D | `cam_high` + `cam_left_wrist` + `cam_right_wrist` |

All bimanual vectors use fixed order:

```text
left + right
```

One joint arm is:

```text
[joint_1_rad, ..., joint_6_rad, gripper_opening_m]
```

One delivery state arm is:

```text
[eef_xyz_base_m (3), rotation6d_base_eef (6), gripper_closed_fraction (1)]
```

`gripper_closed_fraction` uses `0=open, 1=closed`.

One delivery action arm is:

```text
[delta_xyz_base_m (3), left_delta_rotvec_base_rad (3), gripper_target_closed_fraction (1)]
```

## Action Source And Alignment

The action source must be explicit; measured feedback must not be mislabeled as
an operator command.

### Output-only collection (`collect_output_arm.py`, `collect_gui.py`)

The collector cannot read the master arm command:

```text
joint:    action_source=next_measured_qpos, action_alignment=next_observation, action_offset=1
delivery: action_source=next_measured_eef,  action_alignment=next_observation, action_offset=1
```

For joint data, `action[t] = measured_qpos[t+1]`. For delivery data,
`action[t]` is derived from `state[t] -> state[t+1]`. These episodes include a
repeated terminal observation and terminal hold action.

### Master/slave teleoperation (`teleop_single.py`, `teleop.py`)

This is the preferred π0.5 joint training source because the command is known:

```text
state = slave measured qpos
action = master qpos command
action_source = master_joint_feedback
action_alignment = same_step_command
action_offset = 0
```

Single-arm vectors are 7D. Bimanual vectors are 14D in `left + right` order.

## Raw NPZ Fields

Every new episode contains:

```text
state              float32 (T,state_dim)
actions            float32 (T,action_dim)
timestamps         float64 (T,)
instruction        Unicode scalar, natural-language model prompt
success            bool scalar
joint_qpos         float32 (T,7 or 14), optional diagnostics
task / task_name   Unicode scalar, internal task ID (optional)
```

Camera arrays are RGB HWC `uint8`. New joint and bimanual contracts use:

```text
images_cam_high
images_cam_left_wrist / images_cam_right_wrist
image_timestamps_cam_high
image_timestamps_cam_left_wrist / image_timestamps_cam_right_wrist
```

The legacy-compatible single-arm delivery contract keeps:

```text
image
wrist_image
image_timestamps_cam_high
image_timestamps_cam_wrist
```

Machine-readable metadata includes:

```text
contract_version, schema, arm_mode, arm_side, robot_type
state_dim, action_dim, camera_keys
action_semantics, action_source, action_alignment, action_offset
terminal_padding
```

## LeRobot / OpenPI Export

Canonical LeRobot fields are:

```text
observation.state
observation.images.cam_high
observation.images.cam_<side>_wrist                    # single arm
observation.images.cam_left_wrist / cam_right_wrist    # bimanual
action
task
```

`instruction` from NPZ is written as the LeRobot task/prompt. Internal `task`
or `task_name` identifiers are not substituted for the natural-language prompt.

The source of truth is `piper_data_contract.py`. Every saved episode must pass
`validate_piper_data.py`; exported datasets must pass `check_pi05_dataset.py`
and the OpenPI loader before training.
