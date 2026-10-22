import unittest

from bimanual_vla.deployment.client import PeriodicSchedule


class RequestDeadlineTest(unittest.TestCase):
    def test_request_wakes_between_control_ticks_without_consuming_slot(self):
        schedule = PeriodicSchedule(2.5, next_at=1.421)
        self.assertEqual(schedule.wakeup_deadline(1.45, 1.402, in_flight=False), 1.421)
        self.assertEqual(schedule.next_at, 1.421)
        self.assertTrue(schedule.due(1.421))
        schedule.mark_launched(1.421)
        self.assertAlmostEqual(schedule.next_at, 1.821)

    def test_inflight_and_overdue_retry_keep_control_deadline(self):
        schedule = PeriodicSchedule(4, next_at=1.42)
        self.assertEqual(schedule.wakeup_deadline(1.45, 1.4, in_flight=True), 1.45)
        self.assertEqual(schedule.wakeup_deadline(1.45, 1.43, in_flight=False), 1.45)
        self.assertEqual(schedule.wakeup_deadline(1.45, 1.42, in_flight=False), 1.45)


if __name__ == '__main__':
    unittest.main()
