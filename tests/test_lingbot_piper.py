import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from bimanual_vla.deployment.lingbot_piper import PiperPolicy, CAMERAS, DEFAULT_TASK, metadata, prepare_image
from bimanual_vla.deployment.client import validate_policy_metadata


class PiperAdapterTest(unittest.TestCase):
    def observation(self):
        return {'state': np.zeros(14, np.float32),
                'images': {k: np.zeros((3, 256, 256), np.uint8) for k in CAMERAS}}

    def test_metadata_matches_hardware_contract(self):
        protocol = validate_policy_metadata(metadata(), 'both', 'bimanual', 'joint')
        self.assertEqual(protocol.action_dim, 14)
        self.assertFalse(metadata()['rtc_supported'])

    def test_exact_training_crop(self):
        frame = np.random.default_rng(4).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        expected = cv2.resize(frame[55:200], (256, 256), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(prepare_image(frame.transpose(2, 0, 1)), expected)
        with self.assertRaises(ValueError):
            prepare_image(np.zeros((3, 224, 224), np.uint8))

    def test_heldout_images_are_not_cropped_twice(self):
        frame = np.arange(3 * 256 * 256, dtype=np.uint8).reshape(3, 256, 256)
        np.testing.assert_array_equal(prepare_image(frame, 'piper_training_rgb256'), frame.transpose(1, 2, 0))
        with self.assertRaises(ValueError): prepare_image(frame, 'unknown')

    def test_legacy_config_conversion_preserves_values(self):
        import ast
        from types import SimpleNamespace
        from bimanual_vla.deployment.lingbot_piper import normalize_legacy_data_config
        config = SimpleNamespace(joints=[{'arm.position': 14}], norm_type=[{'arm.position': 'bounds_99_woclip'}])
        normalize_legacy_data_config(config)
        self.assertEqual(ast.literal_eval(config.joints[0]), {'arm.position': 14})
        self.assertEqual(ast.literal_eval(config.norm_type[0]), {'arm.position': 'bounds_99_woclip'})
        before = config.joints.copy()
        normalize_legacy_data_config(config)
        self.assertEqual(config.joints, before)

    def test_two_requests_forward_fresh_observations(self):
        model = Mock()
        model.infer.return_value = {'action': np.zeros((50, 14))}
        adapter = PiperPolicy(model)
        obs = self.observation()
        obs['prompt'] = DEFAULT_TASK
        adapter.infer(obs)
        obs.pop('prompt')
        obs['state'][0] = 0.2
        result = adapter.infer(obs)
        self.assertEqual(model.infer.call_count, 2)
        self.assertEqual(result['actions'].shape, (50, 14))
        self.assertAlmostEqual(float(model.infer.call_args.args[0]['observation.state'][0]), 0.2)

    def test_invalid_requests_and_outputs_fail_closed(self):
        model = Mock()
        adapter = PiperPolicy(model)
        obs = self.observation()
        obs['client_metadata'] = {'rtc': {'enabled': True}}
        with self.assertRaises(ValueError): adapter.infer(obs)
        model.infer.assert_not_called()
        obs.pop('client_metadata')
        obs['prompt'] = 'wrong task'
        with self.assertRaises(ValueError): adapter.infer(obs)
        obs.pop('prompt')
        model.infer.return_value = {'action': np.zeros((50, 7))}
        with self.assertRaises(ValueError): adapter.infer(obs)
        model.infer.return_value = {'action': np.full((50, 14), np.nan)}
        with self.assertRaises(ValueError): adapter.infer(obs)

class SyncLauncherTest(unittest.TestCase):
    def test_launcher_locks_sync_and_preserves_explicit_devices(self):
        import sys
        from unittest.mock import patch
        from bimanual_vla.deployment import lingbot_sync_client as launcher
        from bimanual_vla.deployment import client
        argv = ['lingbot', '--left-can', 'test-left', '--right-can', 'test-right',
                '--cam-left-wrist-device', '/test/left', '--cam-right-wrist-device', '/test/right']
        with patch.object(sys, 'argv', argv), patch.object(client, 'main') as main:
            launcher.main()
            main.assert_called_once()
            self.assertIn('--no-rtc-enabled', sys.argv)
            self.assertIn('--no-async-inference', sys.argv)
            self.assertIn('--no-preresize-policy-images', sys.argv)
            self.assertNotIn('--allow-execution', sys.argv)

    def test_launcher_rejects_async_override(self):
        import sys
        from unittest.mock import patch
        from bimanual_vla.deployment import lingbot_sync_client as launcher
        with patch.object(sys, 'argv', ['lingbot', '--async-inference']):
            with self.assertRaises(SystemExit): launcher.main()

    def test_can_serial_survives_interface_renumbering_and_rejects_duplicates(self):
        import tempfile
        from pathlib import Path
        from bimanual_vla.deployment.lingbot_sync_client import resolve_can
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interfaces = root / 'net'
            interfaces.mkdir()
            for interface, serial in [('can1', 'left-serial'), ('can0', 'right-serial')]:
                device = root / 'devices' / interface
                (device / 'net' / interface).mkdir(parents=True)
                (device / 'serial').write_text(serial)
                (interfaces / interface).symlink_to(device / 'net' / interface)
            self.assertEqual(resolve_can('left-serial', interfaces), 'can1')
            self.assertEqual(resolve_can('right-serial', interfaces), 'can0')
            (interfaces / 'can2').symlink_to(root / 'devices/can1/net/can1')
            with self.assertRaises(RuntimeError): resolve_can('left-serial', interfaces)
            with self.assertRaises(RuntimeError): resolve_can('missing', interfaces)
