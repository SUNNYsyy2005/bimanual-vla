import subprocess
import unittest
from unittest.mock import Mock, patch

from server_4090.slurm_job_runner import StepLogStream, build_script


class StepLogStreamTest(unittest.TestCase):
    def test_independent_node_validates_base_before_workloads(self):
        script = build_script({
            'partition': 'h200', 'workdir': '/project', 'dataset_root': '/datasets',
            'access_mode': 'slurm_only', 'base_checkpoint': '/models/pi05 base',
        }, 'test', [['/conda/bin/python', 'train.py']], ['train'])
        check = "test -f '/models/pi05 base/params/_METADATA'"
        self.assertIn(check, script)
        self.assertLess(script.index(check), script.index('srun '))

    @patch('server_4090.slurm_job_runner.subprocess.Popen')
    @patch('server_4090.slurm_job_runner.run_ssh')
    def test_attach_only_own_numeric_steps_and_reconnect(self, ssh, popen):
        ssh.return_value = subprocess.CompletedProcess([], 0, '123.batch\n123.0\n456.0\ninvalid\n')
        process = Mock()
        process.poll.return_value = None
        popen.return_value = process
        stream = StepLogStream('login-server', '123')
        stream.poll()
        stream.poll()
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(popen.call_args.args[0][-1], 'sattach --quiet 123.0')
        process.poll.return_value = 1
        stream.poll()
        self.assertEqual(popen.call_count, 2)
        process.poll.return_value = None
        stream.close()
        process.terminate.assert_called_once()

    @patch('server_4090.slurm_job_runner.subprocess.Popen')
    @patch('server_4090.slurm_job_runner.run_ssh')
    def test_transport_failure_does_not_attach(self, ssh, popen):
        ssh.return_value = subprocess.CompletedProcess([], 255, 'ssh unavailable')
        StepLogStream('login-server', '123').poll()
        popen.assert_not_called()
