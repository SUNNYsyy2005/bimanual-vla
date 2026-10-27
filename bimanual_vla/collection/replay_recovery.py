"""Bounded startup recovery; never reset, enable, or restart a moving replay."""
import time

import numpy as np
from bimanual_vla.collection.replay_errors import ReplayBlocked, ReplayCancelled, ReplayFeedbackPending


def eligible_recovery_joints(arm, drivers, max_age):
    """Allow only cool, disabled, latched thermal faults (deployment policy)."""
    if not drivers.get('faulted'):
        return ()
    msg = arm.GetArmLowSpdInfoMsgs()
    received, enabled = drivers.get('received'), drivers.get('enabled')
    faults = drivers.get('faults')
    if (received is None or len(received) != 6 or not all(received)
            or enabled is None or len(enabled) != 6 or any(enabled)
            or faults is None or len(faults) != 6
            or not 0 <= time.time() - msg.time_stamp <= min(.2, max_age)):
        raise ReplayBlocked('Automatic recovery requires fresh six-motor feedback and all motors disabled')
    candidates = []
    for joint, errors in enumerate(faults, 1):
        if not errors:
            continue
        motor = getattr(msg, f'motor_{joint}')
        if (motor.foc_status_code not in (0x02, 0x22)
                or motor.foc_status.driver_enable_status
                or not 0 <= motor.motor_temp <= 45 or not 0 <= motor.foc_temp <= 45
                or set(errors) - {'motor_overheating', 'driver_error_status'}):
            raise ReplayBlocked(f'Joint {joint}: fault is not eligible for automatic recovery; '
                                f'raw={motor.foc_status_code:#x}; operator intervention required')
        candidates.append(joint)
    if not candidates:
        raise ReplayBlocked('Unknown driver fault; operator intervention required')
    return tuple(candidates)


def recover_startup_faults(hardware, stop, progress):
    """Clear each eligible joint once, with stationary checks before and after.

    The caller has acquired exclusive interfaces. A stopped replay is never
    resumed here: arm-level emergency/fault status is rejected by read().
    """
    from bimanual_vla.deployment.client import driver_enable_status_dict

    def sample(*, require_disabled=False):
        if stop.is_set():
            raise ReplayCancelled()
        hardware.check_ownership()
        data = hardware.read(require_ready=False, allow_recoverable_faults=True)
        statuses = {s: driver_enable_status_dict(a) for s, a in hardware.arms.items()}
        plan = {s: eligible_recovery_joints(a, statuses[s], hardware.config.feedback_max_age)
                for s, a in hardware.arms.items()}
        plan = {s: joints for s, joints in plan.items() if joints}
        if (plan or require_disabled) and any(any(v['enabled']) for v in statuses.values()):
            raise ReplayBlocked('Automatic fault recovery requires both arms disabled')
        return data, plan

    deadline = time.monotonic() + hardware.config.feedback_ready_timeout
    progress({'state': 'recovery_checking'})
    while True:
        try:
            baseline, plan = sample()
            break
        except ReplayFeedbackPending as exc:
            if time.monotonic() >= deadline:
                raise ReplayBlocked(f'Automatic recovery feedback timeout: {exc}') from exc
            stop.wait(.02)
    if not plan:
        progress({'state': 'recovery_not_needed'})
        return

    def stationary(data):
        for side, state in data.items():
            delta = abs(state['joint'] - baseline[side]['joint'])
            if np.max(delta[:6]) > .003 or delta[6] > .001:
                raise ReplayBlocked(f'{side}: arm moved during automatic recovery; refusing to continue')

    progress({'state': 'recovery_waiting_stationary', 'joints': plan})
    for _ in range(20):
        stop.wait(.05)
        data, current = sample(require_disabled=True)
        stationary(data)
        if current != plan:
            raise ReplayBlocked('Fault state changed before recovery; no automatic clear sent')

    attempted = hardware.recovery_attempted
    if any((side, joint) in attempted for side, joints in plan.items() for joint in joints):
        raise ReplayBlocked('Recovery was already attempted; refusing repeated fault clearing')
    for side, joints in plan.items():
        for joint in joints:
            data, current = sample(require_disabled=True)
            stationary(data)
            # Prior clears may already have taken effect, but no new faults
            # may appear while the remaining commands are being published.
            if any(set(js) - set(plan.get(s, ())) for s, js in current.items()):
                raise ReplayBlocked('New fault during recovery; refusing further clears')
            if joint not in current.get(side, ()):
                continue
            progress({'state': 'recovering_driver_fault', 'side': side, 'joint': joint})
            data, current = sample(require_disabled=True)
            stationary(data)
            if joint not in current.get(side, ()):
                continue
            if any(set(js) - set(plan.get(s, ())) for s, js in current.items()):
                raise ReplayBlocked('New fault before recovery publication; refusing clear')
            attempted.add((side, joint))
            hardware.arms[side].JointConfig(joint_num=joint, set_zero=0,
                acc_param_is_effective=0, max_joint_acc=500, clear_err=0xAE)

    cleared = set()
    for _ in range(20):
        stop.wait(.05)
        data, remaining = sample(require_disabled=True)
        stationary(data)
        active = {(s, j) for s, js in remaining.items() for j in js}
        if active & cleared or active - attempted:
            raise ReplayBlocked('Fault recurred/changed after recovery; no repeated clear')
        cleared |= attempted - active
    if remaining:
        raise ReplayBlocked('Fault persisted after one recovery attempt; operator intervention required')
    hardware.read(require_ready=False)
    progress({'state': 'recovery_completed', 'joints': plan})
