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
  and expects one JSON object on stdout. The glue exits 0 after returning a
  valid review result. A nonzero **glue** exit, timeout (8 seconds), bad JSON
  or incomplete fields prevents device opening. RLSOK's own `review-setup`
  exits 1 for both `NEEDS_MATERIAL` and `REVIEW_REQUIRED`: the glue must read
  `report.json`, not discard either result as an undifferentiated CLI error.
- The `request_id` is unique per call. Echo it unchanged; a reply for another
  call is rejected. Capture new evidence during this invocation.
- During deployment, the same request is issued every 5 seconds from a
  background thread. Keep discovery read-only and bounded; compare against
  the same reviewed identity baseline on every call. A review requiring
  intervention or a changed active endpoint terminates the execution session.
  The existing command
  loop independently checks live camera frames and CAN feedback each tick.

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

`UNCHANGED` (fictional and redacted locators):

```json
{
  "schema_version": 1,
  "rlsok_version": "1.5.8",
  "request_id": "f0e1d2c3b4a5968778695a4b3c2d1e0f",
  "review_decision": "UNCHANGED",
  "review_scope": "saved-configuration-only",
  "hardware_dispatch": false,
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
validation; the RLSOK review decision and its limited scope; fresh inventory
method/time; and one concrete locator per active role. The GUI shows the
check result. CollectionSession uses its
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

`NEEDS_MATERIAL` (same fictional request):

```json
{
  "schema_version": 1,
  "rlsok_version": "1.5.8",
  "request_id": "f0e1d2c3b4a5968778695a4b3c2d1e0f",
  "review_decision": "NEEDS_MATERIAL",
  "review_scope": "saved-configuration-only",
  "hardware_dispatch": false,
  "reason_code": "AMBIGUOUS_CAMERA"
}
```

`REVIEW_REQUIRED` uses the same short response with
`"review_decision": "REVIEW_REQUIRED"` and a suitable reason code. Neither
status opens hardware. `reason_code` is a project-glue summary, **not** a
native RLSOK report field; it is 1–64 uppercase ASCII letters, digits or
underscores. It is the only resolver-provided error text surfaced to the GUI.
A malformed code becomes `DEVICE_CHECK_REJECTED`. Keep detailed issues in a
private local report. An `UNCHANGED` response requires `linux-sysfs-udev`
evidence captured for this invocation and at most 30 seconds old;
stale/future timestamps prevent opening.

The native `capture-setup` observation has `status: READY_FOR_REVIEW` or
`NEEDS_MATERIAL`. After comparing it with the preserved approved baseline,
`review-setup` writes `report.json` with `decision: UNCHANGED`,
`NEEDS_MATERIAL` or `REVIEW_REQUIRED`, plus
`scope: saved-configuration-only` and `hardwareDispatch: false`. Map those exact
decision names to `review_decision` above. `UNCHANGED` means the selected
saved configuration has no semantic change; it never authorizes motion.

| Native review decision | GUI bridge action |
| --- | --- |
| `UNCHANGED` | Validate fresh inventory and current selectors, then return resolved endpoints for the existing camera/CAN open path. Existing Dashboard motion authorization remains required. |
| `NEEDS_MATERIAL` | Refuse connect/start and ask the operator to supply or correct the missing inventory, binding or locator material. Do not reapprove the baseline automatically. |
| `REVIEW_REQUIRED` | Refuse connect/start and return to Device settings for a deliberate review of changed roles, schema, source or other selected configuration. Preserve the old baseline until explicitly approved. |

## What the glue must verify

1. Run `rlsok profile discover-setup-devices --output <new-private-file>` to
   capture **fresh**, read-only sysfs/udev evidence. Never reuse the
   `operator-inventory.json` produced by `prepare-piper-setup`; it is a
   selected-role import, not a new hardware observation.
2. Use the preserved `approve-setup` baseline and a new `capture-setup` /
   `review-setup` report. Return `NEEDS_MATERIAL` for missing/ambiguous
   identity or a stale/wrong explicit locator; return `REVIEW_REQUIRED` for
   semantic changes such as a role swap or identity-basis change. Do not
   silently change identity basis or follow a changed USB path.
3. An unchanged physical identity may acquire a new `/dev/videoN` or CAN
   locator. RLSOK reports `UNCHANGED` **only after** `resolve-setup` explicitly
   creates new configuration copies and those copies are captured and
   compared. `resolve-setup` never edits live GUI settings. Require the
   operator-reviewed replacement selector before starting with a new explicit
   locator. During an active session, even an identity-preserving endpoint
   change stops the session so handles can be reopened after review.
4. Prefer camera unit serial. If a unit has no readable serial, accept only an
   explicitly reviewed `usb_path` **together with** `interface` and
   `videoIndex`. A USB path identifies a port, not the physical camera unit;
   replacement at the same port still needs manual review.
5. A CAN adapter serial identifies the adapter, not the Piper arm behind it.
   This check resolves endpoint roles; it does not authorize motion. Keep the
   existing execution authorization and Piper safety checks independent.

The glue must make the final current-selector and fresh-endpoint check, then
return the small response above. The client separately requires its Dashboard
execution authorization and live camera/CAN safety checks. See RLSOK v1.5.8
`docs/saved-setup-review.md`, `docs/piper-confirmed-roles.md` and
`packages/composable-shadow/saved-setup.ts` for the native review semantics.
