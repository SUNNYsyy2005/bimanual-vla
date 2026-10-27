"""Piper LingBot adapter for the existing synchronous safety client."""
from __future__ import annotations

import numpy as np

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
DEFAULT_TASK = 'Place the bottles on the black pad in the center of the table.'


def metadata(horizon=50, source=None):
    return dict(transport='openpi_websocket_v1', schema='joint', arm_mode='bimanual',
                arm_side='both', state_dim=14, action_dim=14, contract_version=3,
                action_semantics='absolute_joint_position_opening_fraction',
                gripper_semantics='absolute_opening_fraction_0_closed_1_open',
                state_gripper_semantics='absolute_opening_fraction_0_closed_1_open',
                camera_keys=list(CAMERAS), action_hz=20, action_horizon=horizon,
                rtc_supported=False, image_resize_mode='pad',
                policy_family='lingbot_piper', checkpoint_source=source,
                image_preprocessing='padded256_crop_0_55_256_200_cv2_linear256')


def prepare_image(value, image_format="padded256"):
    import cv2
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.shape != (3, 256, 256):
        raise ValueError(f'Expected padded RGB uint8 CHW 256 image, got {image.shape}/{image.dtype}')
    if image_format == 'piper_training_rgb256':
        return image.transpose(1, 2, 0).copy()
    if image_format != 'padded256':
        raise ValueError(f'Unknown input image format: {image_format}')
    return cv2.resize(image.transpose(1, 2, 0)[55:200, :], (256, 256),
                      interpolation=cv2.INTER_LINEAR)


def normalize_legacy_data_config(config):
    """Older LingBot FeatureConfig expects literal strings instead of YAML maps."""
    for key in ('joints', 'norm_type'):
        values = getattr(config, key, None)
        if isinstance(values, list):
            setattr(config, key, [repr(v) if isinstance(v, dict) else v for v in values])


class PiperPolicy:
    def __init__(self, policy, task=DEFAULT_TASK, horizon=50):
        self.policy, self.task, self.horizon = policy, task, horizon

    def infer(self, observation):
        rtc = observation.get('client_metadata', {}).get('rtc', {})
        if rtc.get('enabled') or 'action_prefix' in observation:
            raise ValueError('LingBot Piper accepts synchronous inference only; disable RTC')
        # Fixed task per service avoids prompt leakage between websocket clients.
        if observation.get('prompt', self.task) != self.task:
            raise ValueError('Prompt differs from the task configured on this service')
        state = np.asarray(observation['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 14D Piper state')
        inputs = {'observation.state': state, 'task': self.task}
        for key in CAMERAS:
            inputs[f'observation.images.{key}'] = prepare_image(observation['images'][key], observation.get('image_format', 'padded256'))
        actions = np.asarray(self.policy.infer(inputs)['action'], dtype=np.float32)
        if actions.shape != (self.horizon, 14) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid Piper action chunk: {actions.shape}')
        return {'actions': actions}
