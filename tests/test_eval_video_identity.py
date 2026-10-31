import hashlib
import json
import os
from pathlib import Path
from unittest import mock

from server_4090.eval_video_identity import deduplicate_videos, local_identities
from server_4090.eval_video_browser import publish_index, VideoBrowser, group_id
from tests.test_eval_video_browser import Args


def row(root, relative, experiment='a', **fields):
    return dict(root=str(root), relative_path=relative, experiment=experiment, task='task', run_id=experiment,
                video_source='robotwin', phase='eval', mtime=1, name=Path(relative).name, **fields)


def test_identical_content_across_experiments_keeps_single_scope_membership_and_pagination(tmp_path):
    (tmp_path/'a.mp4').write_bytes(b'same-video')
    (tmp_path/'b.mp4').write_bytes(b'same-video')
    (tmp_path/'c.mp4').write_bytes(b'other-file')
    rows=[row(tmp_path,'a.mp4'),row(tmp_path,'b.mp4','b'),row(tmp_path,'c.mp4','b')]
    publish_index(tmp_path,rows,{})
    browser=VideoBrowser(tmp_path)
    a,b=group_id(rows[0]),group_id(rows[1])
    assert browser.groups()['total']==2
    assert browser.groups()['duplicate_count']==1
    assert browser.page(Args(scope=[a]))['total']==1
    assert browser.page(Args(scope=[b]))['total']==2
    first=browser.page(Args(scope=[a,b],limit='1'))
    second=browser.page(Args(scope=[a,b],limit='1',offset='1'))
    assert first['total']==2 and first['duplicate_count']==1
    assert first['videos'][0]['id']!=second['videos'][0]['id']
    assert sum(v.get('duplicate_count',0) for v in first['videos']+second['videos'])==1
    assert all((tmp_path/name).exists() for name in ('a.mp4','b.mp4','c.mp4'))


def test_same_size_name_and_seed_with_different_bytes_remain_distinct(tmp_path):
    for name,content in [('left.mp4',b'left'),('right.mp4',b'rght')]:
        (tmp_path/name).write_bytes(content)
    rows=[row(tmp_path,name,episode_seed=7) for name in ('left.mp4','right.mp4')]
    publish_index(tmp_path,rows,{})
    assert VideoBrowser(tmp_path).groups()['total']==2


def test_cache_avoids_rehash_and_invalidates_changed_file(tmp_path):
    for name in ('a.mp4','b.mp4'):(tmp_path/name).write_bytes(b'same')
    rows=[row(tmp_path,name) for name in ('a.mp4','b.mp4')]
    local_identities(rows,tmp_path)
    original=Path.open
    def no_video_reads(path,*args,**kwargs):
        assert path.suffix!='.mp4'
        return original(path,*args,**kwargs)
    with mock.patch.object(Path,'open',no_video_reads):
        cached=local_identities(rows,tmp_path)
    assert cached[0]['content_sha256']==cached[1]['content_sha256']
    (tmp_path/'b.mp4').write_bytes(b'new!')
    stat=(tmp_path/'b.mp4').stat()
    os.utime(tmp_path/'b.mp4',ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
    changed=local_identities(rows,tmp_path)
    assert changed[0]['content_sha256']!=changed[1]['content_sha256']


def test_remote_file_aliases_merge_but_host_identity_is_preserved():
    identity=dict(host='h100',device=1,inode=10,size=123,mtime_ns=10)
    base=dict(id='a',source='one',root='/runs',relative_path='a.mp4',remote=True,file_identity=identity)
    alias=dict(base,id='b',source='two',root='/alias')
    other=dict(base,id='c',file_identity=dict(identity,host='h200'))
    result=deduplicate_videos([base,alias,other])
    assert len(result)==2 and sum(r.get('duplicate_count',0) for r in result)==1


def test_nested_root_provenance_and_local_preference():
    remote=dict(id='remote',source='h100',root='/runs',relative_path='a/review.mp4',remote=True)
    local=dict(id='local',source='local_4090',root='/cache',relative_path='copy.mp4',remote=False,
               provenance=dict(storage_source='h100',root='/runs/a',relative_path='review.mp4'))
    rows=deduplicate_videos([remote,local])
    assert len(rows)==1 and rows[0]['id']=='local'
    assert rows[0]['duplicate_records'][0]['id']=='remote'
