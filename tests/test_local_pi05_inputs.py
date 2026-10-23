import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/pi05_local'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('local_pi05_serve', SCRIPTS/'serve.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)

class LocalPi05InputsTest(unittest.TestCase):
    def test_right_arm_training_camera_layout(self):
        high=np.full((224,224,3),10,np.uint8); wrist=np.full_like(high,20)
        result=serve.PiperInputs(7,'right')({'state':np.zeros(7),'images':{'cam_high':high,'cam_right_wrist':wrist.transpose(2,0,1)}})
        self.assertEqual(list(result['image']),list(serve.CAMERAS.values()))
        np.testing.assert_array_equal(result['image']['right_wrist_0_rgb'],wrist)
        self.assertFalse(result['image_mask']['left_wrist_0_rgb'])
        self.assertTrue(result['image_mask']['right_wrist_0_rgb'])
        self.assertEqual(np.count_nonzero(result['image']['left_wrist_0_rgb']),0)
        self.assertEqual(result['state'].shape,(7,))
    def test_rejects_bimanual_state_on_single_policy(self):
        with self.assertRaises(ValueError):
            serve.PiperInputs(7,'right')({'state':np.zeros(14),'images':{}})
    def test_bimanual_still_uses_three_cameras(self):
        result=serve.PiperInputs()({'state':np.zeros(14),'images':{k:np.zeros((224,224,3),np.uint8) for k in serve.CAMERAS}})
        self.assertTrue(all(result['image_mask'].values()))
        self.assertEqual(result['state'].shape,(14,))
    def test_physical_output_slicing(self):
        a=np.arange(50*32,dtype=np.float32).reshape(50,32)
        np.testing.assert_array_equal(serve.PiperOutputs(7)({'actions':a})['actions'],a[:,:7])
        a[0,0]=np.nan
        with self.assertRaises(ValueError):serve.PiperOutputs(7)({'actions':a})

if __name__=='__main__':unittest.main()
