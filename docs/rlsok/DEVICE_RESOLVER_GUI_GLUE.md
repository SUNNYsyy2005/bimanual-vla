# RLSOK v1.5.8 → bimanual-vla: device resolver / GUI contract

This is the contract for the **project-side glue executable**. RLSOK v1.5.8
provides read-only discovery and saved-setup review commands; it does not
provide this runtime GUI resolver. The glue implements the comparison between
the current GUI selection, a preserved operator-reviewed role baseline and a
new inventory. `bimanual_vla.device_guard.check_devices()` is the consumer.

## Invocation

- Configure the absolute executable path in GUI **Device settings → RLSOK
  resolver**, in `BIMANUAL_VLA_RLSOK_RESOLVER`, or with the direct client's
  `--rlsok-resolver` flag. No shell is used.
- The client starts it once per check, sends one UTF-8 JSON object on stdin,
  and expects one JSON object on stdout. Exit 0 is required for either an
  explicit allow or an explicit deny. Nonzero exit, timeout (8 seconds), bad
  JSON and incomplete fields are failures and deny device opening.
- The `request_id` is unique per call. Echo it unchanged; a reply for another
  call is rejected. Capture new evidence during this invocation.

Single-arm request (all identifiers here are fictional):

```json
{
  "schema_version": 1,
  "request_id": "f0e1d2c3b4a5968778695a4b3c2d1e0f",
  "purpose": "collection",
  "arm_mode": "single",
  "arm_side": "right",
  "requested": {
    "arms": {"right": "auto"},
    "cameras": {"overhead": "auto", "right_wrist": "auto"}
  }
}
```

`purpose` is `collection` or `deployment`. Bimanual requests use
`arm_side: "both"`, arm roles `left` and `right`, and camera roles
`overhead`, `left_wrist`, `right_wrist`. The `requested` values are the current
GUI/CLI selectors. `auto` means “find the reviewed identity”; it never means
“choose the first camera of this model.” Current bimanual GUI configuration
requires distinct, explicit left/right CAN names. If an explicit CAN name
has moved, refuse the check and update the reviewed configuration and GUI
selector before retrying; the bridge will not silently remap it.

## Minimum response to the GUI bridge

Allow (fictional and redacted locators):

```json
{
  "schema_version": 1,
  "rlsok_version": "1.5.8",
  "request_id": "f0e1d2c3b4a5968778695a4b3c2d1e0f",
  "decision": "allow",
  "inventory_method": "linux-sysfs-udev",
  "observed_at": "2026-09-29T08:00:00Z",
  "resolved": {
    "arms": {"right": "can1"},
    "cameras": {
      "overhead": "/dev/v4l/by-id/EXAMPLE-REDACTED-video-index0",
      "right_wrist": "/dev/v4l/by-path/EXAMPLE-REDACTED-video-index0"
    }
  }
}
```

The wire fields are exactly what the bridge needs: version and nonce for
validation; decision; evidence method/time; and one concrete locator per
active role. The GUI shows the check result. CollectionSession uses its
resolved locators; the inference child performs a fresh check and uses that
result. The GUI does not need raw serials, the full
inventory, baseline contents or RLSOK report text. A locator may itself
contain an identifier, so do not print the response or resolver stderr into
the GUI log. Extra JSON fields are rejected. The example above is redacted;
it is not a real device mapping.

Allowed camera locators are `/dev/videoN`, `/dev/v4l/by-id/<name>` and
`/dev/v4l/by-path/<name>`; CAN locators are interface names. Use a stable
camera symlink when it uniquely identifies the reviewed stream. Every active
role must resolve to exactly one endpoint, and endpoints must be distinct.

Denial (same fictional request):

```json
{
  "schema_version": 1,
  "rlsok_version": "1.5.8",
  "request_id": "f0e1d2c3b4a5968778695a4b3c2d1e0f",
  "decision": "deny",
  "reason_code": "AMBIGUOUS_CAMERA"
}
```

`reason_code` is 1–64 uppercase ASCII letters, digits or underscores. It is
the only resolver-provided error text surfaced to the GUI. A malformed code
becomes `DEVICE_CHECK_REJECTED`. Keep details in a private local report if
needed. The bridge requires `linux-sysfs-udev` evidence captured for this
invocation and at most 30 seconds old; stale/future timestamps deny.

## What the glue must verify

1. Run `rlsok profile discover-setup-devices --output <new-private-file>` to
   capture **fresh**, read-only sysfs/udev evidence. Never reuse the
   `operator-inventory.json` produced by `prepare-piper-setup`; it is a
   selected-role import, not a new hardware observation.
2. Compare current requested selectors and fresh endpoints with the preserved
   operator-reviewed identities. Refuse missing/ambiguous matches, role swaps,
   wrong explicit selectors and duplicate endpoints. Do not silently change
   identity basis or follow a changed USB path.
3. Prefer camera unit serial. If a unit has no readable serial, accept only an
   explicitly reviewed `usb_path` **together with** `interface` and
   `videoIndex`. A USB path identifies a port, not the physical camera unit;
   replacement at the same port still needs manual review.
4. A CAN adapter serial identifies the adapter, not the Piper arm behind it.
   This check resolves endpoint roles; it does not authorize motion. Keep the
   existing execution authorization and Piper safety checks independent.

RLSOK `resolve-setup` edits copies of saved configuration, and `review-setup`
reports saved-configuration differences. Neither command is itself a runtime
motion permit. The glue must make the final current-selector decision and
return the small response above.
