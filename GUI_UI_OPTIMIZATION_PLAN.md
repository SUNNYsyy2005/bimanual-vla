# Piper Collection GUI UI Optimization

Date: 2026-08-12

## Scope

This change simplifies the collection GUI around the operator's main job:
select a dataset, activate/connect devices, watch three cameras, inspect the
two arm states, and start/stop episodes.

The raw NPZ contract, validation, conversion, upload, CAN safety checks, and
camera capture implementation remain unchanged unless an interaction fix
requires a small supporting change.

## Current Problems

1. Status and robot telemetry repeat too much technical information. The same
   state is shown in several places, including a large joint table that is not
   useful during routine collection.
2. Device fields occupy most of the left panel even though CAN names, camera
   paths, and rates rarely change.
3. Initial-pose tolerance controls and the live pose check add visual noise and
   are no longer collection blockers.
4. Reset to Home is exposed in the routine collection controls despite being a
   potentially disruptive robot command.
5. Camera role swapping is disabled in bimanual mode and cannot be toggled off.
6. The Space shortcut is lost when a combobox or button retains focus.
7. Camera previews use square tiles for wide camera streams, wasting space and
   making the useful image content appear too small.
8. Dataset selection only supports typing a name. Existing dataset directories
   are not offered as choices, and the dataset cannot be switched after robot
   connection.

## Target Interface

### Task And Dataset

- Keep arm mode, schema, dataset, task, and instruction on the main screen.
- Dataset is an editable combobox populated from existing directories under
  the configured dataset parent directory.
- Typing a new valid name remains supported.
- Dataset selection remains enabled after devices connect, but is locked while
  an episode is actively recording or awaiting save/discard review.
- A `Device settings` button opens a small modal containing CAN names, camera
  selectors, collection rate, and camera rate.

### Collection Controls

- Control order starts with `Activate CAN`, then `Connect devices`.
- Keep Start, Stop, Refresh, Replay, dataset editing/tools, and Exit.
- Remove Reset to Home.
- Replace the old swap action with a functional left/right wrist-camera toggle.
  The toggle may be enabled or disabled while disconnected. Changing it while
  connected requires camera reconnection, so the UI will disconnect/reconnect
  cameras through the normal device connection workflow rather than mutate an
  active capture silently.

### Status

- Status contains one short operational line only: disconnected, activating,
  connecting, ready, recording, saved, or error.
- Episode progress and dataset counts remain concise.
- Remove initial-pose status from the visible UI.

### Arm Data

- Remove `Live robot pose` and its per-joint tolerance table.
- Show exactly two compact rows in bimanual mode, one for left and one for
  right.
- Delivery schema row: `x y z rx ry rz gripper`, where rotation is displayed as
  Euler XYZ radians derived from the stored rotation6D state.
- Joint schema row: `j1 j2 j3 j4 j5 j6 gripper`.
- Single-arm mode shows one row using the selected arm side.

### Camera Layout

- Bimanual mode displays three active cameras without a reserved fourth tile.
- Use wide 16:9 preview containers that match the camera content better.
- Desktop layout: overhead spans the top width; left and right wrist views sit
  side by side below it.
- Preview dimensions use responsive constraints and preserve source aspect
  ratio without stretching.

### Keyboard Control

- Space starts/stops an episode regardless of retained focus on a combobox or
  ordinary button.
- Space is ignored only while typing in text/entry widgets, while a modal is
  open, or when another top-level window owns focus.
- The handler returns `break` when it performs the collection action so the
  focused widget cannot also activate.

## Files

- `collect_gui.py`: primary UI and interaction refactor.
- `test_collect_gui_dataset_tools.py`: pure helper and dataset discovery tests.
- `test_piper_data_contract.py`: telemetry formatting tests if needed.
- `GUI_OPERATION_GUIDE.md`: update operator-facing controls after behavior is
  verified.

## Verification

1. Static compile and `git diff --check`.
2. Existing collection/data-contract unit tests.
3. New tests for dataset discovery, telemetry formatting, and keyboard action
   routing where practical without a display.
4. Tk construction check under the active display.
5. Hardware-safe connection/read test only; no Reset/Home or motion commands.
6. Restart GUI and inspect desktop layout.
