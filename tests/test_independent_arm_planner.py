import unittest
import numpy as np
from bimanual_vla.deployment.servo import IndependentArmJointPlanner, RuckigJointPlanner, ServoConfig, ServoError


class IndependentArmPlannerTest(unittest.TestCase):
    def make_planner(self):
        return IndependentArmJointPlanner(np.zeros(12), -np.ones(12), np.ones(12), ServoConfig())

    def test_left_reversals_do_not_retime_right_arm(self):
        planner = self.make_planner()
        right = RuckigJointPlanner(np.zeros(6), -np.ones(6), np.ones(6), ServoConfig())
        right.retarget(np.full(6, .1))
        last_velocity = np.zeros(12)
        last_acceleration = np.zeros(12)
        for tick in range(140):
            if tick % 10 == 0:
                planner.retarget(np.r_[np.full(6, .4 if (tick//10)%2 else -.4), np.full(6, .1)])
            result = planner.advance()
            np.testing.assert_allclose(result[6:], right.advance(), atol=1e-12)
            self.assertLessEqual(np.max(abs(planner.velocity)), 1.2+1e-7)
            self.assertLessEqual(np.max(abs(planner.velocity-last_velocity))/.01, 8.+1e-7)
            self.assertLessEqual(np.max(abs(planner.acceleration-last_acceleration))/.01, 60.+1e-6)
            last_velocity, last_acceleration = planner.velocity, planner.acceleration
        np.testing.assert_allclose(planner.position[6:], .1, atol=1e-10)

    def test_retarget_and_brake_preserve_state(self):
        planner = self.make_planner()
        planner.retarget(np.full(12, .3))
        for _ in range(15):
            planner.advance()
        state = [planner.position, planner.velocity, planner.acceleration]
        planner.retarget(np.full(12, .4))
        for a,b in zip(state,[planner.position,planner.velocity,planner.acceleration]):
            np.testing.assert_array_equal(a,b)
        planner.brake()
        for a,b in zip(state,[planner.position,planner.velocity,planner.acceleration]):
            np.testing.assert_array_equal(a,b)
        for _ in range(100):
            planner.advance()
        np.testing.assert_allclose(planner.velocity, 0., atol=1e-8)
        self.assertTrue(planner.braking)

    def test_rejected_arm_never_advances_either_state(self):
        planner = self.make_planner()
        with self.assertRaises(ServoError):
            planner.retarget(np.r_[np.full(6, .2), np.full(6, 2.)])
        np.testing.assert_array_equal(planner.position, np.zeros(12))
        np.testing.assert_array_equal(planner.velocity, np.zeros(12))


if __name__ == '__main__':
    unittest.main()
