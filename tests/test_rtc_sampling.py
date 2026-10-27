import unittest
import importlib.util
import numpy as np
from bimanual_vla.training.settings import FinetuningSettings
from bimanual_vla.training.rtc_sampling import sample_rtc_delays

class RTCSamplingTest(unittest.TestCase):
    def test_settings_serialization_and_resume(self):
        import json
        s=FinetuningSettings(rtc_training=True,rtc_training_max_delay=2,rtc_training_delay_weights=[1,2,3])
        restored=FinetuningSettings.resolve({},json.loads(json.dumps(s.to_dict())),resume=True)
        self.assertEqual(s,restored)
        self.assertIn('--rtc-training-delay-weights',s.command_args())
        with self.assertRaises(ValueError):
            FinetuningSettings.resolve({'rtc_training_delay_weights':[2,2,2]},s.to_dict(),resume=True)
        for weights in ([1,2], [0,0,0], [-1,2,3], [float('nan'),1,1]):
            with self.assertRaises(ValueError):
                FinetuningSettings(rtc_training_max_delay=2,rtc_training_delay_weights=weights)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires the JAX training environment")
    def test_empirical_distribution_and_zero_weight(self):
        import jax
        values=np.asarray(jax.jit(lambda key:sample_rtc_delays(key,(20000,),2,(0,1,3)))(jax.random.key(1)))
        self.assertEqual(int(np.sum(values==0)),0)
        self.assertAlmostEqual(float(np.mean(values==2)),.75,delta=.015)
        uniform=np.asarray(sample_rtc_delays(jax.random.key(2),(20000,),2))
        self.assertTrue(np.all(np.abs(np.bincount(uniform)/len(uniform)-1/3)<.015))
