import threading
import time
import unittest

import numpy as np

from bimanual_vla.deployment.client import AsyncPolicyInference, InferenceLaunch, InferenceWorkerResult


def launch(generation=1):
    return InferenceLaunch(generation, time.time(), time.monotonic(), time.time(),
                           time.monotonic(), np.zeros(0), np.zeros(7), {})


class InferenceWakeupTest(unittest.TestCase):
    def test_reference_refresh_wakes_owner_before_control_deadline(self):
        worker = AsyncPolicyInference()
        self.addCleanup(worker.shutdown)
        worker.launch_callable(lambda: {"actions": np.zeros((50, 7))}, launch())
        deadline = time.monotonic() + 0.2
        completions = []
        def accept(result):
            completions.append(result)
            return True
        self.assertTrue(worker.wait_until(deadline, accept))
        self.assertLess(time.monotonic(), deadline - 0.05)
        self.assertEqual(len(completions), 1)
        self.assertFalse(worker.in_flight)

    def test_completion_is_admitted_early_on_owner_thread_without_advancing_tick(self):
        worker = AsyncPolicyInference()
        self.addCleanup(worker.shutdown)
        release = threading.Event()
        received = []
        owner = threading.get_ident()

        def task():
            release.wait(1)
            return InferenceWorkerResult({"actions": np.zeros((50, 7))}, {},
                                         {"response_received_monotonic": time.monotonic()})

        worker.launch_callable(task, launch())
        deadline = time.monotonic() + 0.1
        timer = threading.Timer(0.005, release.set)
        timer.start()
        self.addCleanup(timer.join)
        self.addCleanup(release.set)
        worker.wait_until(deadline, lambda result: received.append((time.monotonic(), threading.get_ident(), result)))
        self.assertEqual(len(received), 1)
        self.assertLess(received[0][0], deadline - 0.02)
        self.assertEqual(received[0][1], owner)
        self.assertGreaterEqual(time.monotonic(), deadline)
        self.assertLess(received[0][2].result["_client_transport_timing"]["response_to_admission_ms"], 30)
        self.assertFalse(worker.in_flight)
        self.assertIsNone(worker.poll())

    def test_immediate_completion_and_errors_wake_without_losing_notifications(self):
        worker = AsyncPolicyInference()
        self.addCleanup(worker.shutdown)
        results = []
        for generation in range(6):
            def task():
                if generation % 2:
                    raise ValueError("synthetic failure")
                return {"actions": np.zeros((50, 7))}
            self.assertTrue(worker.launch_callable(task, launch(generation)))
            self.assertFalse(worker.launch_callable(task, launch(generation)))
            worker.wait_until(time.monotonic() + 0.01, results.append)
            self.assertEqual(len(results), generation + 1)
            self.assertEqual(results[-1].launch.generation, generation)
            self.assertEqual(isinstance(results[-1].error, ValueError), bool(generation % 2))

    def test_deadline_does_not_wait_for_incomplete_inference(self):
        worker = AsyncPolicyInference()
        self.addCleanup(worker.shutdown)
        release = threading.Event()
        self.addCleanup(release.set)
        worker.launch_callable(lambda: release.wait(1), launch())
        started = time.monotonic()
        worker.wait_until(started + 0.01, lambda result: self.fail("inference not ready"))
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertTrue(worker.in_flight)
