"""Read driver limits and recover only cool, disabled, latched thermal faults."""
import logging
import re
import time

import numpy as np

# Deployment choice confirmed by the operator: assume V1.5-8+ units when
# firmware cannot be read. This decodes raw 300 as 3 rad/s; it does not set
# the commanded speed or write the driver's persistent configuration.
ASSUMED_JOINT_SPEED_UNIT = .01

# Exact fallback for offline calculations; live execution queries each driver.
NOMINAL_JOINT_LIMITS_RAD = np.deg2rad([
    [-150., 150.], [0., 180.], [-170., 0.], [-100., 100.], [-70., 70.], [-180., 180.],
])


def install_can_send_guard(piper, *, error_type=RuntimeError):
    """Latch SDK transmit failures instead of treating logged errors as success.

    This confirms host-side submission only, never motor execution. A failed
    connection is not retried automatically, including during cleanup.
    """
    bus = piper.GetCanBus()
    if getattr(piper, '_vla_guarded_can_bus', None) is bus:
        return
    original_send = bus.SendCanMessage
    success = bus.CAN_STATUS.SEND_MESSAGE_SUCCESS
    name = piper.GetCanName()
    piper._vla_can_send_fault = None

    def checked_send(arbitration_id, *args, **kwargs):
        if piper._vla_can_send_fault is not None:
            raise error_type(piper._vla_can_send_fault)
        try:
            result = original_send(arbitration_id, *args, **kwargs)
        except Exception as exc:
            piper._vla_can_send_fault = (
                f'{name} CAN transmit failed at 0x{arbitration_id:03x}: {exc}; connection latched')
            raise error_type(piper._vla_can_send_fault) from exc
        if result != success:
            piper._vla_can_send_fault = (
                f'{name} CAN transmit failed at 0x{arbitration_id:03x}: {result}; connection latched')
            raise error_type(piper._vla_can_send_fault)
        return result

    bus.SendCanMessage = checked_send
    piper._vla_guarded_can_bus = bus


def require_can_send_healthy(pipers, *, error_type=RuntimeError):
    for side, piper in pipers.items():
        fault = getattr(piper, '_vla_can_send_fault', None)
        if isinstance(fault, str) and fault:
            raise error_type(f'{side}: {fault}')


def joint_speed_unit(firmware):
    """0x473/0x474 changed units at firmware V1.5-8 (inclusive).

    SDK 0.6.1's docstrings omit this version distinction. Return None for
    unknown versions so the deployment assumption is logged explicitly.
    """
    match = re.fullmatch(r"(?:S-)?V(\d+)\.(\d+)-(\d+)", str(firmware).strip())
    if match is None:
        return None
    version = tuple(map(int, match.groups()))
    return .01 if version >= (1, 5, 8) else .001


def query_limits(piper, timeout=2.0):
    started = time.time()
    firmware_reader = getattr(piper, 'GetPiperFirmwareVersion', None)
    firmware_query = getattr(piper, 'SearchPiperFirmwareVersion', None)
    if callable(firmware_query):
        firmware_query()
    piper.SearchAllMotorMaxAngleSpd()
    deadline = time.monotonic() + timeout
    previous = None
    diagnostic = 'no fresh complete reply'
    next_query = time.monotonic() + .2
    while time.monotonic() < deadline:
        msg = piper.GetAllMotorAngleLimitMaxSpd()
        motors = msg.all_motor_angle_limit_max_spd.motor
        # The SDK exposes a shared aggregate, updated one motor at a time.
        # Its timestamp and prefilled motor numbers do not prove all six
        # replies have arrived. Copy values and wait for stable valid data.
        snapshot = tuple((motors[i].motor_num, motors[i].min_angle_limit,
                          motors[i].max_angle_limit, motors[i].max_joint_spd)
                         for i in range(1, 7))
        if msg.time_stamp >= started and all(row[0] == i for i, row in enumerate(snapshot, 1)):
            firmware = firmware_reader() if callable(firmware_reader) else None
            unit = joint_speed_unit(firmware)
            if unit is None and callable(firmware_reader) and time.monotonic() + .06 < deadline:
                time.sleep(.02)
                continue
            verified_unit = unit is not None
            if not verified_unit:
                unit = ASSUMED_JOINT_SPEED_UNIT
            limits = np.deg2rad([[row[1] / 10, row[2] / 10] for row in snapshot])
            raw_speeds = np.array([row[3] for row in snapshot])
            speeds = raw_speeds * unit
            invalid = (~np.isfinite(limits).all(axis=1) | (limits[:, 0] >= limits[:, 1])
                       | (np.abs(limits) > 2 * np.pi).any(axis=1) | ~np.isfinite(speeds)
                       | (speeds <= 0) | (speeds > 3.0))
            if invalid.any():
                diagnostic = (f'invalid driver joint limits: '
                              f'raw (joint,min,max,speed)={[snapshot[i] for i in np.flatnonzero(invalid)]}; '
                              f'firmware={firmware!r} unit={unit}')
                previous = None
            elif previous != snapshot:
                previous = snapshot
                diagnostic = 'driver joint limits not yet stable'
            else:
                if not verified_unit:
                    logging.warning('Piper firmware version unavailable/unrecognized (%r); '
                                    'using operator-selected V1.5-8+ assumption: '
                                    '0.01 rad/s/raw (300 = 3.0 rad/s); firmware unit is unverified.',
                                    firmware)
                piper._vla_joint_limits = limits
                piper._vla_joint_speeds = speeds
                piper._vla_speed_unit_verified = verified_unit
                piper._vla_firmware_version = firmware
                logging.info('Driver limits deg=%s configured speed rad/s=%s '
                             'raw=%s firmware=%r unit=%s verified=%s',
                             np.rad2deg(limits).tolist(), speeds.tolist(), raw_speeds.tolist(),
                             firmware, unit, verified_unit)
                return limits, speeds
        else:
            previous = None
        if time.monotonic() >= next_query:
            piper.SearchAllMotorMaxAngleSpd()
            next_query = time.monotonic() + .2
        time.sleep(.02)
    raise RuntimeError(f'driver joint limit query incomplete: {diagnostic}; refusing motion')


def effective_limits(piper, software):
    hardware = getattr(piper, '_vla_joint_limits', None)
    if hardware is None:
        return software.copy()
    # A validated driver is authoritative, including wider ranges than old
    # SDK defaults. Never round radians or intersect with stale constants.
    if (not isinstance(hardware, np.ndarray) or hardware.shape != (6, 2)
            or not np.isfinite(hardware).all()
            or np.any(hardware[:, 0] >= hardware[:, 1])):
        raise RuntimeError('invalid cached driver joint limits')
    return hardware.copy()


def effective_speed(pipers, requested, *, speed_pct=None):
    """Bound references by driver speed and the active MOVE_J percentage.

    Reserve 20% of the percentage-scaled ceiling for actuator tracking.
    Omitting speed_pct retains the configured-driver-limit-only API.
    """
    factor = 1.
    if speed_pct is not None:
        if not np.isfinite(speed_pct) or not 0 < speed_pct <= 100:
            raise ValueError('Firmware speed percentage must be within (0, 100]')
        factor = float(speed_pct) / 100 * .8
    limit = float(requested)
    for piper in pipers.values():
        speeds = getattr(piper, '_vla_joint_speeds', None)
        if isinstance(speeds, np.ndarray):
            limit = min(limit, float(np.min(speeds)) * factor)
    return limit


def recover_return_faults(pipers, status_reader):
    """One attempt per joint per request; caller first verifies standstill.

    45 C is a conservative recovery gate, not a motor rated temperature.
    Other faults (including unexplained driver errors) require intervention.
    """
    candidates = []
    for side, piper in pipers.items():
        status = status_reader(piper)
        if not status.get('faulted'):
            continue
        msg = piper.GetArmLowSpdInfoMsgs()
        for i, faults in enumerate(status.get('faults') or [], start=1):
            if not faults:
                continue
            m = getattr(msg, f'motor_{i}')
            raw = m.foc_status_code
            if (raw not in (0x02, 0x22) or m.foc_status.driver_enable_status
                    or not (0 <= m.motor_temp <= 45 and 0 <= m.foc_temp <= 45)
                    or not 0 <= time.time() - msg.time_stamp <= .2):
                raise RuntimeError(f'{side} joint{i}: fault is not eligible for automatic recovery; '
                                   f'raw={raw:#x}, motor={m.motor_temp}C driver={m.foc_temp}C')
            candidates.append((side, piper, i))
    for side, piper, i in candidates:
        logging.warning('Manual return: clearing cool latched fault once: %s joint%d', side, i)
        piper.JointConfig(joint_num=i, set_zero=0, acc_param_is_effective=0,
                          max_joint_acc=500, clear_err=0xAE)
    if candidates:
        # Observe fresh replies for a full second; never repeatedly clear.
        for _ in range(10):
            time.sleep(.1)
            for side, piper in pipers.items():
                status = status_reader(piper)
                if not status.get('healthy'):
                    raise RuntimeError(f'{side}: fault persisted/recurred after one clear; return aborted: {status}')
