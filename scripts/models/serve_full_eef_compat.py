"""Serve Full EEF checkpoints through an explicitly configured delivery bridge."""
import argparse
import json
import logging
from pathlib import Path

from bimanual_vla.deployment.full_eef_compat import (
    EEFBridge, FullEEFPolicy, compatibility_metadata, execution_ready_from_config,
)


def main():
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize
    from openpi.training import config
    from vla_server.config import register_configs

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--bridge-config', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    provenance = json.loads((checkpoint / 'vla_provenance.json').read_text())
    if (provenance.get('config_name') != 'piper_pi05_full'
            or provenance.get('robot_type') != 'piper_bimanual_eef'
            or provenance.get('vla_model_gripper_semantics') != 'meters'
            or provenance.get('deployment_ready') is not False):
        raise ValueError('unexpected source checkpoint contract')
    settings = json.loads(args.bridge_config.read_text())
    if (settings.get('model_representation') != 'xyz_rpy_xyz_rad_opening_m'
            or settings.get('gripper_max_m') != 0.07):
        raise ValueError('this server requires the XYZ/RPY/metre bridge and 0.07m gripper')
    execution_ready = execution_ready_from_config(settings)
    bridge = EEFBridge(**settings['transforms'])
    stats = normalize.load(checkpoint / 'assets')
    if any(len(stats[name].mean) != 14 for name in ('state', 'actions')):
        raise ValueError('expected 14D normalization statistics')
    register_configs()
    raw = policy_config.create_trained_policy(config.get_config('piper_pi05_full'), checkpoint, norm_stats=stats)
    policy = FullEEFPolicy(raw, bridge, execution_ready=execution_ready)
    metadata = compatibility_metadata(str(checkpoint), execution_ready=execution_ready)
    metadata['source_provenance'] = provenance
    metadata['bridge_config'] = settings
    logging.info('Full EEF bridge listening on %s; execution contract ready=%s', args.port, execution_ready)
    WebsocketPolicyServer(policy, host='0.0.0.0', port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
