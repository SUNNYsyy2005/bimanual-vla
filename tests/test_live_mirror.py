import threading,time
from server_4090.live_mirror import LiveMirror


def wait_for(check):
    end=time.monotonic()+3
    while time.monotonic()<end:
        if check():return
        time.sleep(.01)
    raise AssertionError('background mirror did not finish')


def test_first_read_never_waits_coalesces_and_syncs_without_reads(tmp_path):
    mirror=LiveMirror(tmp_path,interval=.1,workers=1);gate=threading.Event();calls=[]
    def build():calls.append(1);gate.wait(2);return {'points':[len(calls)]}
    try:
        start=time.monotonic();first=mirror.get('task:one:metrics',build,seed={'points':[]})
        assert time.monotonic()-start<.2 and first['sync']['ready'] is False
        wait_for(lambda:len(calls)==1)
        for _ in range(10):mirror.get('task:one:metrics',build)
        assert len(calls)==1
        gate.set();wait_for(lambda:mirror.entries['task:one:metrics']['updated']>0)
        wait_for(lambda:len(calls)>1)
        assert mirror.get('task:one:metrics',build)['points']
    finally:gate.set();mirror.close()


def test_failure_preserves_copy_restart_restores_it(tmp_path):
    mirror=LiveMirror(tmp_path,interval=.01)
    try:
        mirror.get('rlinf:a:log',lambda:{'log':'last good'})
        wait_for(lambda:mirror.entries['rlinf:a:log']['updated']>0)
        def fail():raise RuntimeError('source offline')
        mirror.get('rlinf:a:log',fail)
        wait_for(lambda:mirror.entries['rlinf:a:log']['error'])
        value=mirror.get('rlinf:a:log',fail)
        assert value['log']=='last good' and 'offline' in value['sync']['error']
    finally:mirror.close()
    restarted=LiveMirror(tmp_path)
    try:assert restarted.get('rlinf:a:log',lambda:{})['log']=='last good'
    finally:restarted.close()


def test_deleted_history_cannot_be_resurrected_by_inflight_sync(tmp_path):
    mirror=LiveMirror(tmp_path);gate=threading.Event();started=threading.Event()
    def build():started.set();gate.wait(2);return {'log':'deleted data'}
    mirror.get('task:gone:log',build);assert started.wait(1)
    mirror.forget('task:gone');gate.set();mirror.close()
    assert not list(tmp_path.rglob('*.json'))


def test_unchanged_tensorboard_not_reparsed(monkeypatch):
    from server_4090 import rlinf_dashboard as rl
    rl._live_event_cache.clear();calls=[]
    def scan(source,root,depth,parse_events=False):
        calls.append(parse_events)
        return {'curves':{'training_curve.json':[{'step':1,'loss':2}]} if parse_events else {},'events':{'tensorboard':{'updated_at':1}}}
    monkeypatch.setattr(rl,'_scan_root',scan)
    run={'id':'a','path':'/fixture','state':'running'}
    rl.live_metrics_for_run(run);rl.live_metrics_for_run(run)
    assert calls==[False,True,False]


def test_selected_reads_do_not_refresh_global_task_lists():
    import ast
    from pathlib import Path
    tree=ast.parse(Path('server_4090/app.py').read_text())
    names={'task_log','task_metrics','selected_rlinf_run','rlinf_dashboard_run_metrics','rlinf_run_log'}
    found={node.name:ast.unparse(node) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names}
    assert set(found)==names
    for source in found.values():
        assert 'visible_rlinf_runs()' not in source
        assert 'tasks.get(' not in source
        assert 'tasks.list(' not in source


def test_snapshot_selected_read_copies_only_one_row(tmp_path):
    from server_4090.dashboard_snapshot import Snapshot
    snapshot=Snapshot(tmp_path/'index.json',lambda:(_ for _ in ()).throw(AssertionError('global scan')))
    snapshot.data=[{'id':'a','points':[1]},{'id':'b','points':[2]}]
    row=snapshot.find('a');row['points'].append(3)
    assert snapshot.find('a')['points']==[1]
    assert snapshot.find('missing') is None
    assert not snapshot.running
