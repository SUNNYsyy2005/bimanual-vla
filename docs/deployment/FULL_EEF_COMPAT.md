Full EEF compatibility bridge
=============================

The bridge accepts the existing delivery v3 state: left then right, each with
base XYZ metres, the first two rotation-matrix columns, and gripper opening fraction.
It produces the Full model's per-arm `[x,y,z,roll,pitch,yaw,opening_m]`
input. Euler angles use SciPy extrinsic `xyz` radians. The confirmed maximum
gripper opening is 0.07 m.

The user confirmed the output layout on 2026-11-05: left XYZ, left roll/pitch/yaw,
left gripper, right XYZ, right roll/pitch/yaw, right gripper; position and gripper
units are metres and angles are radians. The user subsequently confirmed that
both state and action use the same native Piper SDK definition: absolute pose
in each arm's own base frame, native SDK tool reference, SDK RPY convention,
and gripper 0 m closed / 0.07 m fully open.

Absolute model output is converted back to delivery v3 targets, then to the
existing chunk-origin XYZ / left rotvec / opening-fraction wire actions.
Every output in a chunk uses the same request state as its anchor. Thus the
client's existing local IK and command admission path can decode the result.
Nonfinite values, malformed arrays and gripper values outside the physical
range are rejected. Model RTC prefixes are unsupported.

`scripts/models/full_eef_bridge.json` records the confirmed contract:
identity base/tool transforms and the input/output layout. The confirmed output
arm order is left then right. For each physical arm:

    model_pose = model_base_from_client_base @ client_pose @ client_tool_from_model_tool

The SDK FK used by the existing local IK was checked against 10 recorded
joint/EEF pairs from `monitoring_data/20261105_095313_19bba18f/events.jsonl`:
maximum position difference 0.000767 m, maximum rotation difference 0.002534 rad.
Original checkpoint provenance remains unchanged under `source_provenance`.
The adapter advertises `deployment_ready=true` only after validating the
confirmed SDK contract, identity transforms, arm order and recorded FK check.
Changing `validated` alone is insufficient. Unvalidated configurations retain
the shadow-only client and server gates.

This validates representation compatibility and recorded kinematics consistency.
No live model-controlled robot motion or task success test has been performed.

On 4x4090 the two services are `bimanual-foldbox-compat@abc20k` (8220)
and `bimanual-foldbox-compat@pi05_base` (8221). Defaults are GPU 2 for ABC20K
and GPU 1 for Pi0.5 Base. The memory preflight requires at least 13 GB free
before loading. Select a different available GPU with `FULL_EEF_GPU`. Start script:

    bash scripts/models/start_4090_full_eef_compat.sh pi05_base

Use the ordinary deployment client with `--port 8221` and `--no-rtc-enabled`.
Without `--allow-execution`, the client performs shadow evaluation. With the
confirmed contract, `--allow-execution` is accepted; existing local IK, workspace,
joint limits, feedback freshness and trajectory checks still control every command.
Old 8213/8214 services can continue running. Review and stop a specific service
with `systemctl --user stop bimanual-foldbox-compat@pi05_base` on 4x4090.

Validation on 2026-11-05: the bridge and existing client tests passed. Both real checkpoints
returned finite `(50,14)` converted actions from synthetic images and their own
normalization mean state. Decoding them recovered the raw absolute model targets
with maximum component error `1.1920929e-7`. Before source contract confirmation,
the local client and real server rejected execution requests. After confirmation,
execution-mode handshakes and synthetic inference requests were checked without
opening CAN or sending robot commands.
