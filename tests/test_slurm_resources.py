import subprocess
import unittest

from server_4090.slurm_resources import render_snapshot, snapshot_command


TARGETS = [{'node': 'gpu-01', 'partition': 'gpu'}]
SNAPSHOT = '''__BEGIN_META__
sunny
2026-10-20T14:00:00+08:00
__RC_META__=0
__BEGIN_NODES__
NodeName=gpu-01 State=MIXED+DRAIN CPUAlloc=16 CPUTot=32 RealMemory=64000 AllocMem=32000 CfgTRES=cpu=32,gres/gpu=8,gres/gpu:h100=8 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h100=2
__RC_NODES__=0
__BEGIN_JOBS__
77_2|gpu|a-long-job-name-that-must-not-be-truncated|other|RUNNING|1:00|gpu-01|gres/gpu:h100:2
88|gpu|my-pending-job|sunny|PENDING|0:00|(Priority)|gres/gpu:h100:2
89|gpu|someone-elses-pending-job|other|PENDING|0:00|(Resources)|gres/gpu:h100:1
__RC_JOBS__=0
__BEGIN_DETAILS__
JobId=79 ArrayJobId=77 ArrayTaskId=2 AllocTRES=cpu=16,gres/gpu=2 Nodes=gpu-01 CPU_IDs=0-15 GRES=gpu:h100:2(IDX:0,3)
__RC_DETAILS__=0
__BEGIN_QUOTA__
GPU remaining: 500 hours
__RC_QUOTA__=0
'''


class SlurmResourceTest(unittest.TestCase):
    def test_other_users_gpu_jobs_and_own_unassigned_jobs_are_visible(self):
        output, failed = render_snapshot(SNAPSHOT, TARGETS)
        self.assertFalse(failed)
        self.assertRegex(output, r'gpu-01\s+MIXED\+DRAIN\s+2/8\s+6')
        self.assertRegex(output, r'77_2\s+other\s+RUNNING')
        self.assertIn('0,3', output)
        self.assertIn('a-long-job-name-that-must-not-be-truncated', output)
        self.assertRegex(output, r'88\s+sunny\s+PENDING')
        self.assertIn('(Priority)', output)
        self.assertNotIn('someone-elses-pending-job', output)
        self.assertIn('someone-elses-pending-job', render_snapshot(SNAPSHOT, TARGETS, True)[0])

    def test_failure_is_not_hidden_by_successful_quota(self):
        output, failed = render_snapshot(SNAPSHOT.replace('__RC_JOBS__=0', '__RC_JOBS__=1'), TARGETS)
        self.assertTrue(failed)
        self.assertIn('JOBS 查询失败', output)
        self.assertIn('GPU remaining', output)

    def test_missing_node_or_sections_never_report_success(self):
        self.assertTrue(render_snapshot('', TARGETS)[1])
        output, failed = render_snapshot(SNAPSHOT.replace('NodeName=gpu-01', 'NodeName=other'), TARGETS)
        self.assertTrue(failed)
        self.assertIn('节点 gpu-01 未返回数据', output)

    def test_multi_node_allocation_keeps_indices_for_each_node(self):
        raw = SNAPSHOT.replace('GRES=gpu:h100:2(IDX:0,3)',
                               'GRES=gpu:h100:2(IDX:0,3) Nodes=gpu-02 CPU_IDs=1 GRES=gpu:h100:1(IDX:7)')
        output, failed = render_snapshot(raw, TARGETS)
        self.assertFalse(failed)
        self.assertIn('0,3, gpu-02:7', output)

    def test_missing_allocations_are_unknown_and_typed_gpus_not_double_counted(self):
        raw = SNAPSHOT.replace('AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h100=2', '')
        self.assertRegex(render_snapshot(raw, TARGETS)[0], r'\?/8\s+\?')
        raw = SNAPSHOT.replace('gres/gpu=8,', '').replace('gres/gpu=2,', '')
        self.assertRegex(render_snapshot(raw, TARGETS)[0], r'2/8\s+6')

    def test_remote_shell_queries_partition_without_node_or_user_filter(self):
        command = snapshot_command(TARGETS, native=True)
        subprocess.run(['bash', '-n'], input=command, text=True, check=True)
        self.assertIn('squeue -p gpu -h', command)
        self.assertNotIn('squeue -w', command)
        self.assertNotIn(' -u ', command)
        self.assertIn('scontrol show job -dd -o', command)
        self.assertIn('section NATIVE resources', command)

    def test_empty_slurm_alloc_tres_means_zero_not_missing(self):
        raw = SNAPSHOT.replace('AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h100=2', 'AllocTRES=')
        self.assertRegex(render_snapshot(raw, TARGETS)[0], r'0/8\s+8')

    def test_remote_command_preserves_failure_after_successful_quota(self):
        functions = '''scontrol() { echo "permission denied"; return 1; }
squeue() { return 0; }
myquota() { echo "quota available"; }
'''
        result = subprocess.run(['bash'], input=functions + snapshot_command(TARGETS),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('__RC_NODES__=1', result.stdout)
        self.assertIn('__RC_QUOTA__=0', result.stdout)
        self.assertTrue(render_snapshot(result.stdout, TARGETS)[1])


if __name__ == '__main__':
    unittest.main()
