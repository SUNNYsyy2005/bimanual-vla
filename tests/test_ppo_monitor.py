import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from server_4090.rlinf_dashboard import scan_runs, metrics_for_run, _scan_root, _local_metadata
from server_4090.configure_ppo_monitor import configure_ppo_monitor


class PPOMonitorTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.stage=self.root/'task'/'train-123';self.stage.mkdir(parents=True)
        self.cfg={'rlinf_dashboard':{'sources':[dict(id='ppo',root=str(self.root),max_depth=2)]}}
        self.put('protocol.json',dict(status='RUNNING',job_id='123',algorithm='PPO-Lag',config=dict(
            task_name='fshoa_microwave',algorithm='ppo-lag',rounds=32,evaluation_rounds=[0,8,16,24,32],initialization_risk_failure_scored=False)))
        self.put('training_curve.json',[dict(round=1,actor_updates=0,SR=0,critic={'critic_cost_mse':.2}),dict(round=2,actor_updates=1,SR=.25,ppo_advantage={'cost_gae_mean':.1})])
        self.put('evaluation_curve.json',[dict(round=0,actor_updates=0,sampler='sde',cohort='fixed_validation',SR=.05,eligible_episodes=20,excluded_initialization_risk=12),dict(round=0,actor_updates=0,sampler='sde',cohort='extra_development',SR=.4167)])
        (self.stage.parent/'monitor_phase.json').write_text(json.dumps(dict(name='actor_backward',round=3,output=str(self.stage))))

    def put(self,name,value):
        (self.stage/name).write_text(json.dumps(value))

    def test_ppo_metadata_and_cohorts_do_not_conflate_rounds_updates_or_sr(self):
        run=scan_runs(self.cfg)[0]
        self.assertEqual((run['algorithm'],run['completed_training_round'],run['actor_updates']),('ppo-lag',2,1))
        self.assertEqual((run['planned_rounds'],run['next_evaluation_round'],run['phase']),(32,8,'actor_backward'))
        self.assertEqual(run['latest']['eval.SR@sde'],.05)
        self.assertEqual(run['latest']['eval.excluded_initialization_risk@sde'],12)
        self.assertEqual(run['latest']['train.critic.critic_cost_mse'],.2)
        self.assertEqual(run['latest']['train.ppo_advantage.cost_gae_mean'],.1)
        self.assertEqual(len(metrics_for_run([run],run['id'])['evaluation_points']),2)

    def test_zero_points_still_show_startup_and_completion_wins(self):
        self.put('training_curve.json',[]);self.put('evaluation_curve.json',[])
        self.assertEqual(scan_runs(self.cfg)[0]['state'],'running')
        self.put('completion.json',dict(status='FAILED',error='numeric check failed'))
        run=scan_runs(self.cfg)[0]
        self.assertEqual(run['state'],'failed');self.assertIn('numeric check failed',run['status_reason'])

    def test_remote_scan_and_local_metadata_match(self):
        scan=_scan_root({'python':sys.executable},str(self.root),2)
        self.assertEqual(scan['metadata'],_local_metadata(self.root,2))

    def test_slurm_absence_does_not_claim_running(self):
        cfg=copy.deepcopy(self.cfg);cfg['rlinf_dashboard']['sources'][0].update(python=sys.executable,query_slurm=True)
        scan=_scan_root({'python':sys.executable},str(self.root),2)
        scan['job_states']={'123':'NOT_IN_QUEUE'}
        with patch('server_4090.rlinf_dashboard._scan_root',return_value=scan):
            self.assertEqual(scan_runs(cfg)[0]['state'],'observed')
        scan['job_states']={'123':'RUNNING'}
        with patch('server_4090.rlinf_dashboard._scan_root',return_value=scan):
            self.assertEqual(scan_runs(cfg)[0]['state'],'running')

    def test_explicit_stop_survives_stale_protocol(self):
        self.cfg['rlinf_dashboard']['sources'][0]['state']='stopped'
        self.assertEqual(scan_runs(self.cfg)[0]['state'],'stopped')

    def test_registration_is_idempotent_and_preserves_other_settings(self):
        config=dict(secret='fixture',rlinf_dashboard=dict(sources=[{'id':'existing'}],targets={'h100':{'profiles':{'grpo':{}}}}))
        new=configure_ppo_monitor(config,root='/runs')
        self.assertEqual(configure_ppo_monitor(new,root='/runs'),new)
        self.assertEqual(new['rlinf_dashboard']['targets'],config['rlinf_dashboard']['targets'])
        self.assertEqual(new['rlinf_dashboard']['sources'][0],{'id':'existing'})

    def test_completed_mirror_timestamp_ignores_repeated_exports(self):
        source=self.cfg['rlinf_dashboard']['sources'][0]
        source.update(root=str(self.root/'mirror'),python=sys.executable,status_from_process=True)
        mirror=Path(source['root']);mirror.mkdir()
        value=dict(status='COMPLETE',job_id='45438',exported_unix=200,provenance=[dict(mtime=100)],
                   training_curve=[dict(round=128,actor_updates=127,SR=.75)],evaluation_curve=[])
        (mirror/'process.json').write_text(json.dumps(value))
        first=scan_runs(self.cfg)[0]
        value['exported_unix']=300
        (mirror/'process.json').write_text(json.dumps(value))
        second=scan_runs(self.cfg)[0]
        self.assertEqual(first['updated_at'],100)
        self.assertEqual(second['updated_at'],100)
        self.assertEqual(second['synced_at'],300)
        self.assertEqual(second['state'],'completed')

if __name__=='__main__':unittest.main()
