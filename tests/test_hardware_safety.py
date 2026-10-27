import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
from bimanual_vla.deployment.hardware_safety import (
    query_limits, effective_limits, effective_speed, recover_return_faults, joint_speed_unit,
)


class HardwareSafetyTest(unittest.TestCase):
    def test_move_j_percentage_limits_reference_speed_with_tracking_reserve(self):
        p = NS(_vla_joint_speeds=np.full(6, 3.))
        self.assertAlmostEqual(effective_speed({'left': p}, .6, speed_pct=10), .24)
        self.assertAlmostEqual(effective_speed({'left': p}, .6, speed_pct=5), .12)
        self.assertAlmostEqual(effective_speed({'left': p}, .6, speed_pct=100), .6)
        self.assertAlmostEqual(effective_speed({'left': p}, .08, speed_pct=10), .08)
        slower = NS(_vla_joint_speeds=np.array([3., 2., 3., 3., 3., 3.]))
        self.assertAlmostEqual(effective_speed({'left': p, 'right': slower}, .6, speed_pct=10), .16)
        for pct in (0, -1, 101, float('nan')):
            with self.subTest(pct=pct), self.assertRaises(ValueError):
                effective_speed({'left': p}, .6, speed_pct=pct)

    def test_firmware_speed_unit_boundary(self):
        for version, unit in [('S-V1.5-7', .001), ('S-V1.5-8', .01),
                              ('S-V1.6-3', .01), ('S-V1.8-8', .01),
                              ('V1.5-8', .01), (None, None), (-0x4AF, None),
                              ('unknown', None), ('S-V1.5-', None)]:
            with self.subTest(version=version):
                self.assertEqual(joint_speed_unit(version), unit)

    def test_new_firmware_300_is_three_rad_per_second(self):
        motors = [None] + [NS(motor_num=i, min_angle_limit=-1000,
                             max_angle_limit=1000, max_joint_spd=300) for i in range(1, 7)]
        p = NS(SearchAllMotorMaxAngleSpd=Mock(), SearchPiperFirmwareVersion=Mock(),
               GetPiperFirmwareVersion=lambda: 'S-V1.5-8', GetAllMotorAngleLimitMaxSpd=lambda:
               NS(time_stamp=time.time(), all_motor_angle_limit_max_spd=NS(motor=motors)))
        _, speeds = query_limits(p)
        np.testing.assert_allclose(speeds, 3.)
        self.assertEqual(effective_speed({'left': p}, .6), .6)
        self.assertEqual(effective_speed({'left': p}, .3), .3)
        self.assertTrue(p._vla_speed_unit_verified)
        p.SearchPiperFirmwareVersion.assert_called_once()
        motors[1].max_joint_spd = 0x7FFF
        with self.assertRaisesRegex(RuntimeError, '32767.*firmware=.*unit=0.01'):
            query_limits(p, timeout=.06)

    def test_partial_aggregate_waits_for_all_valid_motor_limits(self):
        motors = [None] + [NS(motor_num=i, min_angle_limit=0,
                             max_angle_limit=0, max_joint_spd=0) for i in range(1, 7)]
        reads = 0

        def read():
            nonlocal reads
            reads += 1
            # Aggregate timestamp and motor IDs were ready from the first
            # reply, but the remaining motor values arrive asynchronously.
            for i in range(1, min(reads, 6) + 1):
                motors[i].min_angle_limit = -1000
                motors[i].max_angle_limit = 1000
                motors[i].max_joint_spd = 300
            return NS(time_stamp=time.time(), all_motor_angle_limit_max_spd=NS(motor=motors))

        p = NS(SearchAllMotorMaxAngleSpd=Mock(), GetAllMotorAngleLimitMaxSpd=read,
               GetPiperFirmwareVersion=lambda: 'S-V1.8-2')
        _, speeds = query_limits(p, timeout=.5)
        self.assertGreaterEqual(reads, 7)
        np.testing.assert_allclose(speeds, 3.)

    def test_persistent_partial_aggregate_never_installs_limits(self):
        motors = [None] + [NS(motor_num=i, min_angle_limit=0,
                             max_angle_limit=0, max_joint_spd=0) for i in range(1, 7)]
        p = NS(SearchAllMotorMaxAngleSpd=Mock(), GetPiperFirmwareVersion=lambda: 'S-V1.8-2',
               GetAllMotorAngleLimitMaxSpd=lambda:
               NS(time_stamp=time.time(), all_motor_angle_limit_max_spd=NS(motor=motors)))
        with self.assertRaisesRegex(RuntimeError, 'invalid driver joint limits.*refusing motion'):
            query_limits(p, timeout=.06)
        self.assertFalse(hasattr(p, '_vla_joint_limits'))

    def test_query_units_and_intersection(self):
        motors = [None] + [NS(motor_num=i, min_angle_limit=-1000,
                             max_angle_limit=1000, max_joint_spd=300) for i in range(1, 7)]
        p = NS(SearchAllMotorMaxAngleSpd=Mock(), GetAllMotorAngleLimitMaxSpd=lambda:
               NS(time_stamp=time.time(), all_motor_angle_limit_max_spd=NS(motor=motors)),
               GetPiperFirmwareVersion=lambda: 'S-V1.5-7')
        limits, speeds = query_limits(p)
        np.testing.assert_allclose(limits, np.tile(np.deg2rad([-100, 100]), (6, 1)))
        np.testing.assert_allclose(speeds, .3)
        np.testing.assert_allclose(effective_limits(p, np.tile([-1., 1.], (6, 1))), np.tile([-1., 1.], (6, 1)))
        self.assertEqual(effective_speed({'left': p}, .6), .3)
        self.assertEqual(effective_speed({'left': p}, .2), .2)

    def test_unknown_firmware_uses_operator_assumption_without_raising_gui_limit(self):
        motors = [None] + [NS(motor_num=i, min_angle_limit=-1000,
                             max_angle_limit=1000, max_joint_spd=300) for i in range(1, 7)]
        p = NS(SearchAllMotorMaxAngleSpd=Mock(), GetAllMotorAngleLimitMaxSpd=lambda:
               NS(time_stamp=time.time(), all_motor_angle_limit_max_spd=NS(motor=motors)))
        with self.assertLogs(level='WARNING') as logs:
            _, speeds = query_limits(p)
        np.testing.assert_allclose(speeds, 3.)
        self.assertFalse(p._vla_speed_unit_verified)
        self.assertIn('operator-selected', logs.output[0])
        self.assertEqual(effective_speed({'left': p}, .3), .3)
        self.assertEqual(effective_speed({'left': p}, .6), .6)
        self.assertEqual(effective_speed({'left': p}, 4.), 3.)
        p.GetPiperFirmwareVersion = lambda: 'unavailable'
        with self.assertLogs(level='WARNING'):
            _, speeds = query_limits(p, timeout=.1)
        np.testing.assert_allclose(speeds, 3.)

    def driver(self, raw=0x22, temperature=35):
        m = NS(foc_status_code=raw, motor_temp=temperature, foc_temp=39,
               foc_status=NS(driver_enable_status=False))
        return NS(GetArmLowSpdInfoMsgs=lambda: NS(time_stamp=time.time(), motor_5=m),
                  JointConfig=Mock())

    def faulty(self):
        return {'faulted': True, 'healthy': False,
                'faults': [[], [], [], [], ['motor_overheating', 'driver_error_status'], []]}

    def test_cool_latched_fault_cleared_once_without_enabling_or_rezeroing(self):
        p = self.driver()
        status = Mock(side_effect=[self.faulty()] + [{'healthy': True, 'faulted': False}] * 10)
        with patch('bimanual_vla.deployment.hardware_safety.time.sleep'):
            recover_return_faults({'left': p}, status)
        p.JointConfig.assert_called_once_with(joint_num=5, set_zero=0,
            acc_param_is_effective=0, max_joint_acc=500, clear_err=0xAE)

    def test_hot_or_nonthermal_fault_is_never_cleared(self):
        for raw, temperature in [(0x22, 65), (0x20, 35), (0x24, 35), (0x62, 35)]:
            p = self.driver(raw, temperature)
            with self.subTest(raw=raw, temperature=temperature), self.assertRaises(RuntimeError):
                recover_return_faults({'left': p}, lambda _: self.faulty())
            p.JointConfig.assert_not_called()

    def test_recurrence_aborts_without_repeated_clears(self):
        p = self.driver()
        with patch('bimanual_vla.deployment.hardware_safety.time.sleep'), self.assertRaises(RuntimeError):
            recover_return_faults({'left': p}, lambda _: self.faulty())
        self.assertEqual(p.JointConfig.call_count, 1)
