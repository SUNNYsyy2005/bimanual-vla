import json
from pathlib import Path
from unittest import mock

import pytest

from server_4090.eval_video_lineage import training_lineage, run_key
from server_4090.eval_video_browser import publish_index, VideoBrowser, group_id
from tests.test_eval_video_browser import Args


def manifests(task='tsa-microwave'):
    common=dict(schema='experiment_run.v1',video_source='libero-safety',task=task,execution_target='h100')
    parent=dict(common,kind='train',experiment='training',run_id='train-v3',path='/canonical/'+task+'/train',
                origin=dict(legacy_path='/runs/'+task+'/train'))
    evaluation=dict(common,kind='eval',experiment='final32',run_id='evaluation',path='/eval/'+task,
                    training_run_ref=dict(path='/runs/'+task+'/train',checkpoint='/runs/'+task+'/train/actor-round032.pt'))
    return parent,evaluation


def test_declared_legacy_path_resolves_and_duplicate_manifest_copies_are_harmless():
    parent,evaluation=manifests()
    result=training_lineage([parent,evaluation,dict(evaluation,path='/mirror/eval')])
    assert result[run_key(evaluation)]['experiment']=='training'
    assert result[run_key(evaluation)]['training_run_id']=='train-v3'


@pytest.mark.parametrize('change',[dict(task='fshoa-microwave'),dict(video_source='robotwin'),dict(execution_target='h200-ali-02')])
def test_reference_never_crosses_task_source_or_execution_target(change):
    parent,evaluation=manifests()
    assert training_lineage([parent,dict(evaluation,**change)])=={}


def test_missing_reference_and_ambiguous_paths_are_not_guessed_from_names():
    parent,evaluation=manifests()
    assert training_lineage([parent,dict(evaluation,training_run_ref={})])=={}
    assert training_lineage([parent,evaluation,dict(parent,experiment='other-training')])=={}
    assert training_lineage([parent,dict(evaluation,kind='train')])=={}


def test_conflicting_eval_manifests_do_not_overwrite_one_another():
    parent,evaluation=manifests()
    other=dict(parent,experiment='different',path='/other',origin={})
    conflicting=dict(evaluation,training_run_ref=dict(path='/other'))
    assert training_lineage([parent,other,evaluation,conflicting])=={}


def rows(root,task):
    common=dict(root=str(root),video_source='libero-safety',task=task,mtime=1,experiment='training',
                experiment_label='PPO training',run_id='train-v3',size_bytes=12)
    return [dict(common,relative_path=task+'/train.mp4',phase='train'),
            dict(common,relative_path=task+'/periodic.mp4',phase='eval'),
            dict(common,relative_path=task+'/final.mp4',phase='eval',experiment='final32',
                 experiment_label='Final evaluation',run_id='evaluation'),
            dict(common,relative_path=task+'/old.mp4',phase='train',experiment='training-previous-day',run_id='old')]


def test_groups_merge_by_reference_without_losing_videos_filters_or_legacy_selection(tmp_path):
    records=[];videos=[]
    for task in ('tsa-microwave','fshoa-microwave'):
        records.extend(manifests(task));videos.extend(rows(tmp_path,task))
    publish_index(tmp_path,videos,{},lineage=training_lineage(records))
    browser=VideoBrowser(tmp_path)
    groups=browser.groups()
    assert len(groups['groups'])==4 and groups['total']==8
    task='tsa-microwave'
    original=rows(tmp_path,task)
    current,old=group_id(original[0]),group_id(original[2])
    merged=next(g for g in groups['groups'] if g['id']==current)
    assert merged['count']==3 and merged['independent_eval_count']==1
    assert merged['phase_counts']==dict(train=1,eval=2,unknown=0)
    assert merged['process_run_count']==1 and merged['independent_eval_run_count']==1
    assert merged['label']=='PPO training'
    assert groups['scope_aliases'][old]==current
    page=browser.page(Args(scope=[old,current]))
    assert page['total']==3 and {r['task'] for r in page['videos']}=={task}
    independent=browser.page(Args(scope=[current],phase='independent_eval'))
    assert independent['total']==1
    item=independent['videos'][0]
    assert item['artifact_experiment']=='final32' and item['experiment']=='training'
    assert item['run_id']=='evaluation' and item['training_run_id']=='train-v3'
    assert item['relative_path']==original[2]['relative_path']
    assert browser.page(Args(scope=[current],experiment='final32'))['total']==1
    assert browser.page(Args(scope=[current],q='Final evaluation'))['total']==1
    assert browser.page(Args(scope=[current],phase='train'))['total']==1


def test_api_metadata_hydration_preserves_resolved_experiment_label(tmp_path):
    from tests.test_dashboard_eval_videos import DashboardEvalVideoManagementTest
    app,token,root=DashboardEvalVideoManagementTest()._make_app(tmp_path)
    try:
        video=root/'final.mp4';video.write_bytes(b'video')
        parent,evaluation=manifests()
        raw=dict(root=str(root),relative_path=video.name,video_source='libero-safety',task=parent['task'],
                 experiment='final32',experiment_label='Raw final label',run_id='evaluation',mtime=1,phase='eval')
        video.with_name(video.name+'.metadata.json').write_text(json.dumps(dict(raw,schema='evaluation_video.v1')))
        catalog=tmp_path/'workspace/eval_video_catalog'
        publish_index(catalog,[raw],{},lineage=training_lineage([parent,evaluation]))
        scope=group_id(dict(raw,experiment='training'))
        response=app.test_client().get('/api/eval-videos',query_string=dict(browse='1',scope=scope),
                                       headers={'Authorization':'Bearer '+token})
        assert response.status_code==200,response.json
        item=response.json['videos'][0]
        assert item['experiment']=='training' and item['experiment_label']=='training'
        assert item['artifact_experiment']=='final32' and item['lineage_evidence']=='training_run_ref'
    finally:
        app.extensions['live_mirror'].close()


def test_background_refresh_resolves_local_eval_against_cached_remote_training_manifest(tmp_path):
    import time,os
    from server_4090.eval_video_catalog_runner import refresh
    parent,evaluation=manifests()
    artifacts=tmp_path/'artifacts'
    run=artifacts/'libero-safety'/evaluation['task']/evaluation['experiment']/evaluation['run_id']
    run.mkdir(parents=True)
    (run/'run.json').write_text(json.dumps(evaluation))
    video=run/'eval/round032/success/episode/review.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'video')
    os.utime(video,(time.time()-10,time.time()-10))
    workspace=tmp_path/'workspace';workspace.mkdir()
    (workspace/'artifact-run-inventory.json').write_text(json.dumps(dict(runs=[parent])))
    config=dict(workspace_root=str(workspace),experiment_artifact_root=str(artifacts))
    refresh(config,apply=True,local_only=True)
    groups=VideoBrowser(workspace/'eval_video_catalog').groups()['groups']
    assert len(groups)==1 and groups[0]['experiment']=='training'
    assert groups[0]['independent_eval_count']==1 and video.exists()
