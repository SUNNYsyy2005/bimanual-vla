# Hardware identity configuration

The collection and inference GUIs share `~/.config/bimanual-vla/hardware.json`
(override with `BIMANUAL_VLA_HARDWARE_CONFIG` or `--hardware-config`). The file
binds physical devices to roles; it does not select `single`/`bimanual` mode.
Runtime mode and explicit CLI selectors remain command line options.

Camera records must contain a stable serial and may contain a model. The
runtime resolves the serial to the physical USB camera, then chooses its RGB
V4L2 stream. `/dev/videoN`, by-path names, and `auto` are not identities and
must not be written to the file.

Example (replace the values with the identity wizard's results):

```json
{
  "version": 1,
  "arms": {
    "left": {"can": {"selector": "serial:LEFT_USB_CAN_SERIAL"}},
    "right": {"can": {"selector": "serial:RIGHT_USB_CAN_SERIAL"}}
  },
  "cameras": {
    "overhead": {"serial": "OVERHEAD_SERIAL", "model": "D435i"},
    "left_wrist": {"serial": "LEFT_WRIST_SERIAL", "model": "D405"},
    "right_wrist": {"serial": "RIGHT_WRIST_SERIAL", "model": "D405"}
  },
  "quest": {"adb_serial": "QUEST_ADB_SERIAL", "model_dir": ""},
  "resolver": {"path": ""}
}
```

Open **Device settings → Configure hardware identity**. For each role, scan
the baseline, insert one device, detect it, and save the identity. The wizard
enumerates devices only and does not enable CAN or send robot commands.
