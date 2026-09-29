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
`DEVICE_RESOLVER_GUI_GLUE.md`. Pre-open checks run at connect/start/reconnect;
the deployment child also repeats the review asynchronously during execution.

During real execution, `run_rtc_client()` also checks the background camera
stream on every control tick, using only capture-thread status and monotonic
frame times. A failed read or frame age over `max(0.5 s, 3 / camera_fps)`
ends that execution session. Piper CAN feedback that is missing or stale,
or an abnormal live Piper status, has the same result. The client discards
the old action chunk, stops launching inference, reports the fault to the GUI
through the child exit, and skips automatic return-to-initial motion. A new
session requires a fresh pre-open check and operator execution authorization.
The checks detect loss of data; they do not send CAN probes or motion frames.

When the resolver is configured, `RuntimeDeviceGuard` additionally repeats
the read-only saved-setup review with fresh inventory every 5 seconds in a
daemon thread. `NEEDS_MATERIAL`, `REVIEW_REQUIRED`, timeout, malformed reply,
or a locator change from the session's active endpoints latches a fault,
observed by the control loop before its next command. A locator may change
while RLSOK still reports `UNCHANGED` for the reviewed identity; an active
hardware handle must still be reopened in a new session.
This subprocess never runs on the 20 Hz control thread. A still-open V4L2/CAN
handle can outlive a changed USB locator, so live stream checks and fresh
inventory checks serve different purposes. Recovery is disconnect and a new
start; there is no automatic hardware remap or continuation of the old chunk.

The resolver returns concrete camera locators to `CollectionSession` or
`run_rtc_client()`. In RLSOK mode, `CameraCapture` uses strict selectors: if a
reviewed locator disappears before open, it errors rather than falling back
to model-based discovery. If the resolver path is configured but missing,
times out, exits nonzero, returns `NEEDS_MATERIAL` or `REVIEW_REQUIRED`, or
returns malformed/stale evidence, every
entry above refuses to open devices. An empty resolver path keeps the existing
workflow until the project-side glue is installed.

Discovery and hardware open are separate operations, so a device could change
between them. The child recheck and strict camera selectors narrow that gap;
they do not prove the identity of a unit replaced at the same USB port. The
collection capture loop already stops on camera read or Piper feedback errors;
its episode remains for operator review instead of silently accepting missing
frames. Collection has no continuous motion policy to keep running.

This integration does not start a robot or run a policy during preflight.
No RLSOK saved-setup approval is treated as authorization to move the arm.
