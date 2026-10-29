"""Single-use LEFT J2-to-zero diagnostic. Default only reads/query limits.

Operator must first recover the arm safely, disconnect the leader and clear
the movement area. No reset/recovery, homing, calibration or limit changes.
The only planned joint displacement is J2 from its current small negative
angle to zero, using firmware MOVE_J at 1% speed. Gripper is not commanded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import threading
import time

import numpy as np

from analysis.collection_check_20261025.probe_negative_j2 import SELECTOR, check_anchor
from bimanual_vla.collection.can_identity import resolve_can_selector
from bimanual_vla.collection.replay import (
    PiperReplayHardware, ReplayBlocked, ReplayCancelled, ReplayConfig, wait_for_replay_feedback,
)
from bimanual_vla.deployment.client import arm_status_dict, driver_enable_status_dict


def zero_target(anchor):
    target = np.asarray(anchor, dtype=float).copy()
    target[1] = 0.
    return target


def check_feedback(q, anchor):
    q, anchor = np.asarray(q), np.asarray(anchor)
    other = [0, 2, 3, 4, 5]
    if (not np.isfinite(q).all()
            or np.max(abs(q[other] - anchor[other])) > math.radians(.3)
            or abs(q[6] - anchor[6]) > .001
            or q[1] < anchor[1] - math.radians(.15)
            or q[1] > math.radians(.15)):
        raise ReplayBlocked('Feedback escaped the reviewed J2-to-zero envelope: '
            f'joint_delta_deg={np.rad2deg(q[:6] - anchor[:6]).round(4).tolist()}, '
            f'gripper_delta_mm={(q[6] - anchor[6])*1000:.3f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('j2_zero_plan.json'))
    args = parser.parse_args(argv)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    hardware = PiperReplayHardware(('left',), {'left': SELECTOR}, ReplayConfig(speed_pct=1))
    started = time.monotonic()
    rows, result = [], {'success': False}
    log_path = Path(__file__).with_name(f'j2_zero_{"execute" if args.execute else "preflight"}_{time.time_ns()}.json')
    interfaces = {'left': resolve_can_selector(SELECTOR),
                  'right': resolve_can_selector('serial:003C00365246570620323934')}
    def tx_counts():
        return {s: int((Path('/sys/class/net') / n / 'statistics/tx_packets').read_text())
                for s, n in interfaces.items()}
    before = tx_counts()
    def emit(event):
        rows.append({'time': time.time(), **event})
        if event.get('state') != 'sample':
            print(json.dumps(rows[-1]), flush=True)
    def check_control():
        if stop.is_set():
            raise ReplayCancelled('Stop requested')
        if time.monotonic() - started > 20.:
            raise ReplayBlocked('Total diagnostic deadline exceeded')
        hardware.check_ownership()
    try:
        reviewed = None
        if args.execute:
            reviewed = json.loads(args.plan.read_text())
            if (reviewed.get('execution_attempted_at') is not None
                    or reviewed['selector'] != SELECTOR
                    or not 0 <= time.time() - reviewed['created_at'] <= 600):
                raise ReplayBlocked('Reviewed plan is wrong, expired or already consumed')
        check_control()
        hardware.open()
        initial = wait_for_replay_feedback(hardware, stop, 3., allow_disabled=True)
        anchor = initial['left']['joint'].copy()
        limits = hardware.limits()['left']
        check_anchor(anchor, limits)
        target = zero_target(anchor)
        if np.any(target[:6] < limits[:, 0]) or np.any(target[:6] > limits[:, 1]):
            raise ReplayBlocked('Zero target is outside queried driver/software limits')
        for _ in range(20):
            check_control()
            q = hardware.read(require_ready=False)['left']['joint']
            if np.max(abs(q[:6] - anchor[:6])) > math.radians(.1) or abs(q[6] - anchor[6]) > .001:
                raise ReplayBlocked('Arm moving during passive preflight')
            stop.wait(.05)
        arm = hardware.arms['left']
        enabled = driver_enable_status_dict(arm)['enabled']
        if any(enabled) and not all(enabled):
            raise ReplayBlocked('Partially enabled motors; no automatic recovery')
        if all(enabled):
            status = arm_status_dict(arm)
            if status['ctrl_mode'] != 1 or status['mode_feed'] != 1:
                raise ReplayBlocked('Enabled arm is not already in CAN joint mode')
        plan = {'created_at': time.time(), 'selector': SELECTOR, 'anchor_rad_m': anchor.tolist(),
                'initial_joint_deg': np.rad2deg(anchor[:6]).tolist(),
                'target_joint_deg': np.rad2deg(target[:6]).tolist(),
                'firmware_speed_pct': 1, 'other_joint_drift_limit_deg': .3,
                'gripper_commanded': False, 'sides': ['left']}
        if not args.execute:
            args.plan.write_text(json.dumps(plan, indent=2) + '\n')
            result['success'] = True
            emit({'state': 'passive_plan_ready', **plan})
            return
        saved_q = np.asarray(reviewed['anchor_rad_m'])
        if np.max(abs(saved_q[:6] - anchor[:6])) > math.radians(.1) or abs(saved_q[6] - anchor[6]) > .001:
            raise ReplayBlocked('Pose changed since reviewed plan')
        reviewed['execution_attempted_at'] = time.time()
        args.plan.write_text(json.dumps(reviewed, indent=2) + '\n')
        raw_target = np.rint(np.rad2deg(target[:6]) * 1000).astype(int).tolist()
        def publish():
            check_control()
            if time.time() > hardware.feedback_deadline:
                raise ReplayBlocked('Feedback expired before publication')
            hardware.commanded.add('left')
            # The complete zero-J2 target is staged before selecting MOVE_J.
            arm.JointCtrl(*raw_target)
            arm.ModeCtrl(0x01, 0x01, 1, 0x00)
            arm.JointCtrl(*raw_target)
        motion_start = time.monotonic()
        previous_time, previous_q = motion_start, anchor.copy()
        stable_since, last_stamp, ready = None, None, False
        emit({'state': 'moving_j2_to_zero', **plan})
        while True:
            check_control()
            data = hardware.read(require_ready=False)
            now = time.monotonic()
            q = data['left']['joint']
            status, drivers = arm_status_dict(arm), driver_enable_status_dict(arm)
            emit({'state': 'sample', 'measured_rad_m': q.tolist(), 'target_rad_m': target.tolist(),
                  'status': status, 'drivers': drivers})
            check_feedback(q, anchor)
            dt = now - previous_time
            if dt > .15:
                raise ReplayBlocked('Control clock stalled')
            if dt >= .005 and abs(q[1] - previous_q[1]) / dt > math.radians(3.):
                raise ReplayBlocked('Measured J2 velocity exceeded 3 degrees/s')
            previous_time, previous_q = now, q.copy()
            elapsed = now - motion_start
            ready = drivers['ready'] and status['ctrl_mode'] == 1 and status['mode_feed'] == 1
            if not ready and elapsed > 3.:
                raise ReplayBlocked('Enable/CAN joint mode not ready within 3 s')
            publish()
            if not drivers['ready']:
                # Only normal, healthy feedback reaches here. No reset or fault clear.
                check_control()
                if time.time() > hardware.feedback_deadline:
                    raise ReplayBlocked('Feedback expired before enable')
                # EnablePiper uses the all-motor selector 7. In the first
                # hardware attempt the uncommanded gripper changed by 1.33 mm
                # during that request. Address ONLY the six arm joints here.
                for joint, is_enabled in enumerate(drivers['enabled'], start=1):
                    if not is_enabled:
                        check_control()
                        if time.time() > hardware.feedback_deadline:
                            raise ReplayBlocked('Feedback expired during joint enable')
                        arm.EnableArm(joint)
                publish()
            stamp = status['feedback_timestamp']
            if ready and abs(q[1]) <= math.radians(.1) and stamp != last_stamp:
                stable_since = stable_since or now
                if now - stable_since >= .5:
                    result.update(success=True, final_joint_deg=np.rad2deg(q[:6]).tolist())
                    emit({'state': 'completed', **result})
                    return
            elif not ready or abs(q[1]) > math.radians(.1):
                stable_since = None
            last_stamp = stamp
            if elapsed > 10.:
                raise ReplayBlocked('J2 did not converge to zero within 10 s')
            stop.wait(.02)
    except BaseException as exc:
        result.update(error=str(exc), error_type=type(exc).__name__)
        emit({'state': 'failed', **result, 'stop_errors': hardware.emergency_stop()})
        raise
    finally:
        result.update(commanded_sides=sorted(hardware.commanded), motion_commands_sent=bool(hardware.commanded))
        hardware.close()
        after = tx_counts()
        result['tx_delta'] = {s: after[s] - before[s] for s in before}
        log_path.write_text(json.dumps({'result': result, 'events': rows}, indent=2) + '\n')
        print(json.dumps({'log': str(log_path), **result}), flush=True)


if __name__ == '__main__':
    main()
