# RLSOK v1.5.8 → bimanual-vla: pre-open checkpoints and return paths

`start_gui.sh` is a launcher, not a hardware-opening function. Its path is:

```text
start_gui.sh
  → bin/bimanual-vla collect-gui
  → bimanual_vla.collection.gui.main()
  → CollectorGUI (Tk event loop)
```

The selected arm mode, side and camera/CAN fields are available only after
the GUI has started. Put the authoritative check at the event that opens the
devices, rather than at shell startup. `start_gui.sh` merely passes through
`BIMANUAL_VLA_RLSOK_RESOLVER`; its existing sysfs CAN warnings do not decide
device identity.

| Path | Exact checkpoint | First later hardware open | On resolver refusal |
| --- | --- | --- | --- |
| Collection | `CollectorGUI.toggle_connection()` constructs `CollectionConfig`; `CollectionSession.connect()` calls `_resolve_devices()` first | `collection.output.connect()` → Piper `CreateCanBus()`/`ConnectPort()`, then `CameraCapture.open()` | `toggle_connection()` catches the error, cleans the unconnected session, leaves `piper=None`, enables Device settings and Connect, and shows the reason code. Return to the collection tab and Device settings; no episode begins. |
| Wrist-camera swap during collection | `CollectionSession.reconnect_cameras()` calls `_resolve_devices()` before closing old cameras | New `CameraCapture.open()` | `CollectorGUI.swap_camera_roles()` restores prior role fields/config and keeps or reopens old cameras. Disconnect before changing Device settings. |
| GUI real execution | `CollectorGUI.start_inference()` calls `check_devices()` before `subprocess.Popen()` | Child `run_rtc_client()` → `connect_piper()` → `CreateCanBus()`/`ConnectPort()`, then `CameraCapture.open()` | Do not create the child. Keep inference idle and Device settings enabled; show the reason code. |
| Direct `bin/bimanual-vla rtc-client` | `run_rtc_client()` calls `check_devices()` at its first executable step | `connect_piper()` | Exit 2 before CAN, camera, policy connection or control loop. |

The GUI execution child checks again immediately before its own first CAN
open. That second check catches a role change between GUI precheck and child
startup. It consumes the same resolver contract in
`DEVICE_RESOLVER_GUI_GLUE.md`. Checks run only on connect/start/reconnect, not
on the 20 Hz servo loop.

The resolver returns concrete camera locators to `CollectionSession` or
`run_rtc_client()`. In RLSOK mode, `CameraCapture` uses strict selectors: if a
reviewed locator disappears before open, it errors rather than falling back
to model-based discovery. If the resolver path is configured but missing,
times out, exits nonzero, denies or returns malformed/stale evidence, every
entry above refuses to open devices. An empty resolver path keeps the existing
workflow until the project-side glue is installed.

Discovery and hardware open are separate operations, so a device could change
between them. The child recheck and strict camera selectors narrow that gap;
they do not prove the identity of a unit replaced at the same USB port.

This integration does not start a robot or run a policy during preflight.
No RLSOK saved-setup approval is treated as authorization to move the arm.
