import json
from pathlib import Path

import pytest
from server_4090.eval_video_lineage import training_lineage,experiment_families
from server_4090.eval_video_browser import publish_index,VideoBrowser,group_id
from tests.test_eval_video_lineage import manifests,rows
from tests.test_eval_video_browser import Args


def config():
    return dict(rollout_experiment_families=[dict(id='ppo-series',label='PPO-Lag experiment series',video_source='libero-safety',
        experiments=[dict(id='training',label='Random scenes / PPO v3'),dict(id='training-previous-day',label='Fixed scenes')])])


def test_one_family_per_task_preserves_variants_and_independent_evaluations(tmp_path):
    records=[];videos=[]
    for task in ('fshoa-microwave','tsa-microwave'):
        records.extend(manifests(task));videos.extend(rows(tmp_path,task))
    publish_index(tmp_path,videos,{},lineage=training_lineage(records),families=experiment_families(config()))
    browser=VideoBrowser(tmp_path);groups=browser.groups()
    assert len(groups['groups'])==2 and groups['total']==8
    for task in ('fshoa-microwave','tsa-microwave'):
        group=next(g for g in groups['groups'] if g['task']==task)
        assert group['experiment']=='ppo-series' and group['count']==4
        assert {v['id']:v['count'] for v in group['variants']}=={'training':3,'training-previous-day':1}
        original=rows(tmp_path,task)
        old_scopes=[group_id(original[i]) for i in (0,2,3)]
        assert all(groups['scope_aliases'][scope]==group['id'] for scope in old_scopes)
        assert browser.page(Args(scope=old_scopes))['total']==4
        fixed=browser.page(Args(scope=[group['id']],variant='training-previous-day'))
        random=browser.page(Args(scope=[group['id']],variant='training'))
        assert fixed['total']==1 and random['total']==3
        assert {r['task'] for r in fixed['videos']+random['videos']}=={task}
        final=browser.page(Args(scope=[group['id']],variant='training',phase='independent_eval'))
        assert final['total']==1
        item=final['videos'][0]
        assert item['artifact_experiment']=='final32' and item['experiment_variant']=='training'
        assert item['training_run_id']=='train-v3' and item['run_id']=='evaluation'
        assert browser.page(Args(scope=[group['id']],variant='training-previous-day',phase='independent_eval'))['total']==0
        assert browser.page(Args(scope=[group['id']],experiment='training-previous-day'))['total']==1
        assert browser.page(Args(scope=[group['id']],q='Fixed scenes'))['total']==1


def test_unconfigured_experiments_and_other_video_sources_remain_separate(tmp_path):
    videos=rows(tmp_path,'task')
    publish_index(tmp_path,videos,{})
    assert len(VideoBrowser(tmp_path).groups()['groups'])==3
    other=[dict(row,video_source='robotwin') for row in videos]
    publish_index(tmp_path,other,{},families=experiment_families(config()))
    assert len(VideoBrowser(tmp_path).groups()['groups'])==3


def test_shared_bytes_keep_both_variant_memberships(tmp_path):
    (tmp_path/'a.mp4').write_bytes(b'identical-asset')
    (tmp_path/'b.mp4').write_bytes(b'identical-asset')
    common=dict(root=str(tmp_path),video_source='libero-safety',task='task',mtime=1,run_id='run',phase='eval')
    videos=[dict(common,relative_path='a.mp4',experiment='training'),dict(common,relative_path='b.mp4',experiment='training-previous-day')]
    publish_index(tmp_path,videos,{},families=experiment_families(config()))
    browser=VideoBrowser(tmp_path);group=browser.groups()['groups'][0]
    assert group['count']==1 and len(group['variants'])==2
    for variant in group['variants']:
        assert variant['count']==1
        assert browser.page(Args(scope=[group['id']],variant=variant['id']))['total']==1
    assert (tmp_path/'a.mp4').exists() and (tmp_path/'b.mp4').exists()


def test_conflicting_family_declarations_are_rejected():
    cfg=config();cfg['rollout_experiment_families'].append(dict(cfg['rollout_experiment_families'][0],id='conflicting'))
    with pytest.raises(ValueError,match='Conflicting'):experiment_families(cfg)
