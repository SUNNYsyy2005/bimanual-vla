# Piper Data Contract

This file defines the boundary between any collection UI and the training
pipeline. UI implementations may change without changing this contract.

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

The UI must not construct NPZ dictionaries or actions itself.

## Fixed Model Data

```text
state       float32 (T,10)
actions     float32 (T,7)
image       uint8   (T,256,256,3), RGB HWC
wrist_image uint8   (T,256,256,3), RGB HWC
```

`state` is base-frame EEF xyz in metres, the first two columns of
`R_base_eef`, and gripper closed fraction where `0=open` and `1=closed`.

`actions[t]` is base-frame translation delta, the rotation vector from
`R[t+1] @ R[t].T`, and the absolute next-frame gripper target.

Every episode includes one repeated terminal observation. Its motion action is
zero and its gripper action holds the final gripper state.

## Fixed Episode Metadata

```text
timestamps                    float64 (T,)
task                          Unicode scalar, internal ID
instruction                   Unicode scalar, natural-language model prompt
success                       bool scalar
joint_qpos                    float32 (T,7), optional diagnostics
image_timestamps_cam_high     float64 (T,)
image_timestamps_cam_wrist    float64 (T,)
```

LeRobot export must pass `instruction` as the LeRobot task. `task` is never the
model prompt.

The source of truth is `piper_data_contract.py`. Every saved episode must pass
`validate_piper_data.py` before it is accepted by the UI.
