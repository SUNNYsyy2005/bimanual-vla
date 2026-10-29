"""Bounded LEFT follower diagnostic, separate from production replay limits.

Default: read-only state and limit queries, save a reviewable test plan.
Execution needs --execute --plan PATH after operator clearance confirmation.
No home/reset, calibration, persistent limit changes, or right-arm commands.
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

from bimanual_vla.collection.replay import (
    PiperReplayHardware, ReplayBlocked, ReplayCancelled, ReplayConfig,
    wait_for_replay_feedback,
)
from bimanual_vla.collection.can_identity import resolve_can_selector
from bimanual_vla.deployment.client import driver_enable_status_dict

SELECTOR = 'serial:003D00395246570620323934'
EXCURSION_RAD = math.radians(.3)
MAX_DRIFT_RAD = math.radians(.3)
RATE = 50.


def check_anchor(q, limits):
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ReplayBlocked('Expected finite measured 7D pose')
    # This experiment is only for an already observed small negative J2.
    if not -3. <= math.degrees(q[1]) <= -.5:
        raise ReplayBlocked('Diagnostic only allows current J2 in [-3, -0.5] degrees')
    if not np.allclose(limits[1], np.deg2rad([0., 180.]), atol=.002):
        raise ReplayBlocked('Driver/software J2 limits differ from reviewed configuration')
    other = np.array([0, 2, 3, 4, 5])
    if np.any(q[other] < limits[other, 0]) or np.any(q[other] > limits[other, 1]):
        raise ReplayBlocked('Another joint is outside its normal limits')


def target_at(anchor, elapsed, returning=False):
    # 2 s rest-to-rest quintic segment; peak speed 0.28125 deg/s.
    u = float(np.clip(elapsed / 2., 0., 1.))
    blend = u**3 * (10 - 15*u + 6*u*u)
    target = np.asarray(anchor, dtype=float).copy()
    target[1] += EXCURSION_RAD * (1 - blend if returning else blend)
    return target


def validate_feedback(q, anchor, target):
    q, anchor = np.asarray(q), np.asarray(anchor)
    other = [0, 2, 3, 4, 5]
    if (np.max(abs(q[other] - anchor[other])) > MAX_DRIFT_RAD
            or abs(q[6] - anchor[6]) > .001
            or q[1] < anchor[1] - math.radians(.15)
            or q[1] > anchor[1] + math.radians(.45)
            or abs(q[1] - target[1]) > MAX_DRIFT_RAD):
        raise ReplayBlocked('Measured feedback escaped the small diagnostic envelope')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('negative_j2_plan.json'))
    args = parser.parse_args(argv)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    hardware = PiperReplayHardware(('left',), {'left': SELECTOR}, ReplayConfig(speed_pct=1))
    entries = []
    suffix = 'execute' if args.execute else 'preflight'
    log_path = Path(__file__).with_name(f'negative_j2_{suffix}_{time.time_ns()}.json')
    interfaces = {s: resolve_can_selector(v) for s, v in {
        'left': SELECTOR, 'right': 'serial:003C00365246570620323934'}.items()}
    def tx_counts():
        return {s: int((Path('/sys/class/net') / n / 'statistics/tx_packets').read_text())
                for s, n in interfaces.items()}
    before = tx_counts()
    started = time.monotonic()
    def emit(event):
        entries.append({'time': time.time(), **event})
        if event.get('state') != 'sample':
            print(json.dumps(entries[-1]), flush=True)
    def cancelled():
        if stop.is_set():
            raise ReplayCancelled('Operator/process requested stop')
        if time.monotonic() - started > 20.:
            raise ReplayBlocked('Diagnostic exceeded 20 s total deadline')
        hardware.check_ownership()
    result = {'success': False, 'motion_commands_sent': False}
    try:
        saved = None
        if args.execute:
            saved = json.loads(args.plan.read_text())
            if saved.get('execution_attempted_at') is not None:
                raise ReplayBlocked('This reviewed plan has already been consumed; no automatic retry')
            if saved['selector'] != SELECTOR or not 0 <= time.time() - saved['created_at'] <= 600:
                raise ReplayBlocked('Wrong or expired reviewed plan; repeat passive preflight')
        cancelled()
        hardware.open()
        arm = hardware.arms['left']
        if getattr(arm, '_C_PiperInterface_V2__start_sdk_joint_limit', True):
            raise ReplayBlocked('SDK would clamp negative targets; leaving its setting unchanged')
        baseline = wait_for_replay_feedback(hardware, stop, 3., allow_disabled=True)
        anchor = baseline['left']['joint'].copy()
        limits = hardware.limits()['left']
        check_anchor(anchor, limits)
        for _ in range(20):
            cancelled()
            q = hardware.read(require_ready=False)['left']['joint']
            if np.max(abs(q[:6] - anchor[:6])) > math.radians(.1) or abs(q[6] - anchor[6]) > .001:
                raise ReplayBlocked('Arm is not stationary before diagnostic')
            stop.wait(.05)
        plan = {'created_at': time.time(), 'selector': SELECTOR,
                'anchor_rad_m': anchor.tolist(), 'anchor_deg': np.rad2deg(anchor[:6]).tolist(),
                'outward_target_deg': np.rad2deg(target_at(anchor, 2.)[:6]).tolist(),
                'driver_and_software_limits_deg': np.rad2deg(limits).tolist(),
                'excursion_deg': .3, 'segment_duration_s': 2., 'firmware_speed_pct': 1,
                'sides': ['left'], 'return_to_anchor': True}
        if not args.execute:
            args.plan.write_text(json.dumps(plan, indent=2) + '\n')
            result['success'] = True
            emit({'state': 'passive_plan_ready', **plan})
            return
        if np.max(abs(anchor[:6] - np.asarray(saved['anchor_rad_m'])[:6])) > math.radians(.1):
            raise ReplayBlocked('Pose changed since reviewed plan; repeat passive preflight')
        if abs(anchor[6] - saved['anchor_rad_m'][6]) > .001:
            raise ReplayBlocked('Gripper changed since reviewed plan')
        raw_read = hardware.read
        def guarded_read(*, require_ready=True):
            cancelled()
            data = raw_read(require_ready=require_ready)
            emit({'state': 'sample', 'phase': 'enable_hold',
                  'target_rad_m': anchor.tolist(),
                  'measured_rad_m': data['left']['joint'].tolist()})
            validate_feedback(data['left']['joint'], anchor, anchor)
            return data
        # Apply a tighter measured envelope during initialization than replay's default.
        hardware.read = guarded_read
        saved['execution_attempted_at'] = time.time()
        args.plan.write_text(json.dumps(saved, indent=2) + '\n')
        hardware.enable_holding(baseline, stop, emit)
        hardware.read = raw_read
        extrema = []
        for returning in (False, True):
            phase_start, stable_since = time.monotonic(), None
            phase_target = target_at(anchor, 2., returning)
            emit({'state': 'return_negative_pose' if returning else 'small_positive_excursion',
                  'target_deg': np.rad2deg(phase_target[:6]).tolist()})
            while True:
                cancelled()
                data = hardware.read()
                q = data['left']['joint']
                elapsed = time.monotonic() - phase_start
                target = target_at(anchor, elapsed, returning)
                emit({'state': 'sample', 'returning': returning, 'elapsed_s': elapsed,
                      'target_rad_m': target.tolist(), 'measured_rad_m': q.tolist()})
                validate_feedback(q, anchor, target)
                hardware.send({'left': target})
                extrema.append(float(q[1]))
                if elapsed >= 2. and abs(q[1] - phase_target[1]) <= math.radians(.08):
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= .3:
                        break
                else:
                    stable_since = None
                if elapsed > 4.:
                    raise ReplayBlocked('Small target did not converge within 4 s')
                stop.wait(1 / RATE)
        span = math.degrees(max(extrema) - min(extrema))
        if span < .15:
            raise ReplayBlocked(f'Measured motion {span:.3f} deg cannot prove target tracking')
        final = hardware.read()['left']['joint']
        result.update(success=True, motion_commands_sent=bool(hardware.commanded), measured_j2_span_deg=span,
                      final_joint_deg=np.rad2deg(final[:6]).tolist(),
                      enabled=driver_enable_status_dict(arm)['enabled'])
        emit({'state': 'completed', **result})
    except BaseException as exc:
        result.update(error=str(exc), error_type=type(exc).__name__, motion_commands_sent=bool(hardware.commanded))
        emit({'state': 'failed', **result, 'stop_errors': hardware.emergency_stop()})
        raise
    finally:
        result['motion_commands_sent'] = bool(hardware.commanded)
        result['commanded_sides'] = sorted(hardware.commanded)
        hardware.close()
        after = tx_counts()
        result['tx_delta'] = {s: after[s] - before[s] for s in before}
        log_path.write_text(json.dumps({'result': result, 'events': entries}, indent=2) + '\n')
        print(json.dumps({'log': str(log_path), **result}), flush=True)


if __name__ == '__main__':
    main()
