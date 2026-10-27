import unittest
from server_4090.rlinf_dashboard import _curve_points

class PPOV3DashboardTest(unittest.TestCase):
    def test_separate_round_update_counts_and_finite_diagnostics(self):
        points=_curve_points([dict(round=18,actor_updates=25,actor_optimizer_steps_this_round=8,
            ppo_update=dict(approx_kl=.01,clip_fraction=.25,optimizer_steps=8,details=[1,2]),
            ppo_advantage=dict(prefit_cost_mc_mse=.1,prefit_risk_last5_underestimate=None))])
        self.assertEqual(points[0]['round'],18)
        self.assertEqual(points[0]['actor_updates'],25)
        self.assertEqual(points[0]['ppo_update.optimizer_steps'],8)
        self.assertEqual(points[0]['ppo_update.clip_fraction'],.25)
        self.assertNotIn('ppo_update.details',points[0])
        self.assertNotIn('ppo_advantage.prefit_risk_last5_underestimate',points[0])

if __name__=='__main__':unittest.main()
