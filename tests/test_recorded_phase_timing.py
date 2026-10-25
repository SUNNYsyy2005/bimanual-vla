import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from bimanual_vla.deployment.recording import DeploymentRunRecorder


class RecordedPhaseTimingTest(unittest.TestCase):
    def test_nested_server_phases_survive_recording(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp)
            run = recorder.start({'control_hz': 20})
            launch = SimpleNamespace(generation=1, captured_at=100., captured_monotonic=20.,
                                     launched_at=100.01, launched_monotonic=20.01,
                                     raw_delivery_state=np.zeros(0), qpos_m=np.zeros(7), image_timestamps={})
            phases = {'sampler_wall_ms': 301., 'openpi_input_transform_steps_ms': {'resize': .1}}
            recorder.record_model_result(launch=launch, result={'actions': np.zeros((3,7)),
                                         'model_phase_timing': phases}, arrived_at=100.4,
                                         arrived_monotonic=20.4, protocol=SimpleNamespace(), accepted=True)
            recorder.stop(reason='offline test')
            row = json.loads((run/'model_commands.jsonl').read_text().splitlines()[0])
            self.assertEqual(row['model_phase_timing'], phases)


if __name__ == '__main__':
    unittest.main()
