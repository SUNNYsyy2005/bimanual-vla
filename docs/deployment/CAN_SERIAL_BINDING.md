# USB-CAN serial binding

Device settings accepts `serial:<USB serial number>` in the left/right CAN
fields. The dropdown discovers connected adapters through Linux sysfs. Interface
names such as `can0` remain accepted for legacy scripts, but serial selectors
are the persistent choice when enumeration order or USB ports change.

Example:

```text
--left-can serial:003D00395246570620323934
--right-can serial:003C00365246570620323934
```

These particular serials describe this workstation's configured mapping, not
universal Piper identifiers. They identify USB-CAN adapters: moving an adapter's
CAN cable to the other robot changes which robot that identity controls.

Resolution happens before connecting or activating the interface. Missing or
ambiguous serials fail explicitly without falling back to `can0`/`can1`. Two
selectors resolving to the same interface are rejected before either arm is
connected. Enumeration uses the nearest USB device's serial, never its hub's.
No udev rules, privileged rename, or automatic motion are needed for resolution.
The existing GUI activation action remains responsible for bringing CAN up.

GUI preferences store the selectors. A separate user-only
`~/.config/bimanual-vla/can_bindings.json` preserves hardware bindings if an older
GUI process rewrites its preferences at shutdown. Restart the GUI to load new
code and bindings. Inference, feedback collection and manual return through the
deployment client use the same resolver. Standalone master/slave teleoperation
scripts are not migrated by this change.

Saved return poses compare their device bindings. The workstation's existing
pose was migrated from the last recorded `left=can1/right=can0` assignment after
checking those adapters' current serials; joint coordinates were retained.
Do not automatically migrate an unknown old interface assignment after a swap.

Validation uses simulated sysfs devices: swapped netdev names, changed USB port,
missing/duplicate serial, mixed selectors for the same interface, and a hub whose
serial must not be mistaken for its adapter's. Real sysfs identities were read
without activating CAN or sending robot commands.
