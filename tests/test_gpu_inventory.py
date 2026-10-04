import subprocess
import unittest
from unittest import mock

from server_4090.app import gpu_inventory


class GpuInventoryTests(unittest.TestCase):
    @mock.patch('server_4090.app.subprocess.check_output')
    def test_process_timeout_preserves_gpu_records_and_blocks_compute(self, query):
        query.side_effect = [
            '0, GPU-test, NVIDIA GeForce RTX 4090, 24564, 123\n',
            subprocess.TimeoutExpired('nvidia-smi', 10),
        ]
        rows = gpu_inventory(strict=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['memory_used_mib'], 123)
        self.assertFalse(rows[0]['compute_available'])
        self.assertIn('GPU process query failed', rows[0]['health_issue'])

    @mock.patch('server_4090.app.subprocess.check_output')
    def test_strict_query_reports_failure_instead_of_an_empty_list(self, query):
        query.side_effect = subprocess.TimeoutExpired('nvidia-smi', 10)
        with self.assertRaisesRegex(RuntimeError, 'GPU inventory query failed'):
            gpu_inventory(strict=True)

    @mock.patch('server_4090.app.subprocess.check_output')
    def test_successful_idle_gpu_remains_available(self, query):
        query.side_effect = ['0, GPU-test, NVIDIA GeForce RTX 4090, 24564, 16\n', '']
        rows = gpu_inventory()
        self.assertTrue(rows[0]['compute_available'])
        self.assertEqual(rows[0]['processes'], [])


if __name__ == '__main__':
    unittest.main()
