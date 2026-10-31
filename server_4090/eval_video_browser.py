"""Indexed video browsing: group summaries first, bounded detail pages on demand."""
from __future__ import annotations

import base64
import json
import itertools
from pathlib import Path
import threading
import time

try:
    from .eval_video_identity import deduplicate_videos, local_identities
    from .eval_video_lineage import apply_lineage, apply_families
except ImportError:
    from eval_video_identity import deduplicate_videos, local_identities
    from eval_video_lineage import apply_lineage, apply_families


def group_id(row):
    key=[str(row.get(k) or 'unknown') for k in ('video_source','task','experiment')]
    return base64.urlsafe_b64encode(json.dumps(key,separators=(',',':')).encode()).decode().rstrip('=')


def publish_index(destination, local_rows, remote_sources, *, lineage=None, families=None):
    """Called only by the background catalog refresh, never by a page request."""
    rows=[];synced=set();seen=set()
    for original in local_identities(local_rows, destination):
        row=dict(original);path=Path(row['root'])/row['relative_path']
        row.update(source='local_4090',id=base64.urlsafe_b64encode(str(path.absolute()).encode()).decode().rstrip('='),
                   remote=False,playable=True,syncable=False,deletable=True)
        row['url']='/api/eval-videos/'+row['id']
        provenance=row.get('provenance',{})
        synced.add(tuple(provenance.get(k) for k in ('storage_source','root','relative_path')))
        rows.append(row)
    for source, inventory in remote_sources.items():
        for original in inventory.get('videos',[]):
            if (source,original['root'],original['relative_path']) in synced:continue
            row=dict(original)
            key=f"{source}\0{row['root']}\0{row['relative_path']}"
            row.update(source=source,id=base64.urlsafe_b64encode(key.encode()).decode().rstrip('='),
                       remote=True,playable=False,syncable=True,deletable=False)
            rows.append(row)
    rows = apply_families(apply_lineage(rows, lineage or {}), families or {})
    groups={};lean=[];scope_aliases={}
    fields={'video_source','task','experiment','experiment_label','run_id','phase','checkpoint_id','success','safe_success',
            'source','root','relative_path','catalog_relative_path','name','display_name','mtime','size_mib','size_bytes',
            'id','url','remote','playable','syncable','deletable','has_events','instruction','provenance','episode_seed',
            'episode_status','rollout_round','video_layout','video_views','model_name','score','counted_in_success_rate',
            'file_identity','content_sha256','artifact_experiment','artifact_experiment_label','artifact_role',
            'training_run_id','training_run_ref','lineage_evidence','experiment_family','experiment_variant','experiment_variant_label'}
    for row in rows:
        if row['id'] in seen:continue
        seen.add(row['id'])
        row={k:v for k,v in row.items() if k in fields}
        row['mtime']=float(row.get('mtime') or 0)
        row['updated_at']=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(row['mtime']))
        gid=group_id(row);row['group_id']=gid;lean.append(row)
        for old_experiment in (row.get('artifact_experiment'), row.get('experiment_variant')):
            if old_experiment and old_experiment != row['experiment']:
                scope_aliases[group_id(dict(row, experiment=old_experiment))] = gid
        group=groups.setdefault(gid,dict(id=gid,video_source=row.get('video_source','unknown'),task=row.get('task','unknown'),
               experiment=row.get('experiment','unknown'),label=row.get('experiment_label') or row.get('experiment','unknown'),
               count=0,local_count=0,remote_count=0,latest_mtime=0,run_ids=set()))
        group['count']+=1;group['remote_count' if row['remote'] else 'local_count']+=1
        group['latest_mtime']=max(group['latest_mtime'],row['mtime'])
        group['run_ids'].add(str(row.get('run_id') or 'unknown'))
    unique = deduplicate_videos(lean)
    for group in groups.values():
        members = deduplicate_videos(r for r in lean if r['group_id'] == group['id'])
        group['count'] = len(members)
        group['local_count'] = sum(not r['remote'] for r in members)
        group['remote_count'] = sum(r['remote'] for r in members)
        group['phase_counts'] = {phase:sum(r.get('phase') == phase for r in members) for phase in ('train','eval','unknown')}
        group['independent_eval_count'] = sum(r.get('artifact_role') == 'independent_eval' for r in members)
        group['independent_eval_run_count'] = len({str(r.get('run_id') or 'unknown') for r in members if r.get('artifact_role') == 'independent_eval'})
        group['process_run_count'] = len({str(r.get('run_id') or 'unknown') for r in members if r.get('artifact_role') != 'independent_eval'})
        group['artifact_experiments'] = sorted({name for r in members for name in (r.get('artifact_experiment'), r.get('experiment_variant'), r.get('experiment')) if name})
        variants = {}
        for row in lean:
            if row['group_id'] == group['id'] and row.get('experiment_variant'):
                variants.setdefault(row['experiment_variant'], []).append(row)
        if variants:
            group['variants'] = []
            for variant, values in sorted(variants.items()):
                unique_variant = deduplicate_videos(values)
                group['variants'].append(dict(id=variant,label=values[0]['experiment_variant_label'],count=len(unique_variant),
                    independent_eval_count=sum(r.get('artifact_role') == 'independent_eval' for r in unique_variant),
                    phase_counts={phase:sum(r.get('phase') == phase for r in unique_variant) for phase in ('train','eval')}))
        group['run_count']=len(group['run_ids']);group['run_ids']=sorted(group['run_ids'])
    ordered=sorted(groups.values(),key=lambda g:(-g['latest_mtime'],g['id']))
    updated=time.time()
    for filename,payload in [('browse-index.json',dict(version=1,updated_at=updated,videos=lean)),
                             ('browse-groups.json',dict(version=2,updated_at=updated,total=len(unique),raw_total=len(lean),
                                                       duplicate_count=len(lean)-len(unique),groups=ordered,scope_aliases=scope_aliases))]:
        path=Path(destination)/filename;temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
        temporary.replace(path)


class VideoBrowser:
    def __init__(self,destination):
        self.destination=Path(destination);self.lock=threading.RLock();self.signature=None
        self.by_group={};self.deleted=set()

    def groups(self):
        path=self.destination/'browse-groups.json'
        try:result = dict(json.loads(path.read_text(encoding='utf-8')),ready=True)
        except FileNotFoundError:result = dict(ready=False,total=0,groups=[],updated_at=None)
        try:result['repair'] = json.loads((self.destination.parent/'artifact-index-repair/status.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):pass
        return result

    def discard(self,ids):
        with self.lock:self.deleted.update(ids)

    def page(self,args):
        scopes=list(dict.fromkeys(args.getlist('scope')))
        if len(scopes)>256:raise ValueError('Select at most 256 experiment groups')
        limit=int(args.get('limit',25));offset=int(args.get('offset',0))
        if not 1<=limit<=1000 or not 0<=offset<=10000000:raise ValueError('Invalid video page')
        summary=self.groups()
        scopes=list(dict.fromkeys(summary.get('scope_aliases', {}).get(scope, scope) for scope in scopes))
        if not scopes:return dict(videos=[],total=0,offset=offset,limit=limit,selection_required=True,ready=summary['ready'],facets={})
        if not summary['ready']:return dict(videos=[],total=0,offset=offset,limit=limit,ready=False,facets={})
        allowed={g['id'] for g in summary['groups']}
        if set(scopes)-allowed:raise ValueError('Selected experiment is absent from the current index; refresh groups')
        path=self.destination/'browse-index.json'
        with self.lock:
            stat=path.stat();signature=(stat.st_mtime_ns,stat.st_size)
            if signature!=self.signature:
                by_group={}
                for row in json.loads(path.read_text(encoding='utf-8'))['videos']:
                    by_group.setdefault(row['group_id'],[]).append(row)
                self.by_group=by_group;self.signature=signature
            rows=[r for scope in scopes for r in self.by_group.get(scope,[]) if r['id'] not in self.deleted and matches(r,args)]
        raw_total = len(rows)
        rows = deduplicate_videos(rows)
        rows.sort(key=lambda r:(-r['mtime'],r['id']))
        if len(scopes)>1:
            # Each selected experiment contributes to a page before any dominates it.
            buckets={scope:[] for scope in scopes}
            for row in rows:
                owner = next((scope for scope in scopes if scope in row.get('duplicate_group_ids', [row['group_id']])), row['group_id'])
                buckets[owner].append(row)
            rows=[row for batch in itertools.zip_longest(*(buckets[s] for s in scopes)) for row in batch if row is not None]
        facets={'success':{k:sum(r.get('success')==k for r in rows) for k in ('success','failed','unknown')}}
        return dict(videos=[dict(r) for r in rows[offset:offset+limit]],total=len(rows),raw_total=raw_total,
                    duplicate_count=raw_total-len(rows),offset=offset,limit=limit,
                    ready=True,facets=facets,index_updated_at=summary['updated_at'])


def matches(row,args):
    events=args.get('events','all')
    if events in ('with','without') and bool(row.get('has_events'))!=(events=='with'):return False
    for key in ('video_source','phase','source','success','safe_success'):
        value=args.get(key,'').strip().lower()
        if key == 'phase' and value == 'independent_eval':
            if row.get('artifact_role') != 'independent_eval':return False
            continue
        if value and value!='all' and value!=str(row.get(key,'unknown')).lower():return False
    for key in ('task','experiment','run_id','checkpoint_id'):
        value=args.get(key,'').strip().lower().replace('_','-')
        haystack=str(row.get(key,'')).lower().replace('_','-')
        if key=='experiment':haystack+=' '+' '.join(str(row.get(k,'')) for k in ('experiment_label','artifact_experiment','artifact_experiment_label','experiment_variant','experiment_variant_label')).lower().replace('_','-')
        if value and value not in haystack:return False
    variant=args.get('variant','').strip()
    if variant and variant!='all' and variant!=row.get('experiment_variant'):return False
    query=args.get('q','').strip().lower()
    return not query or query in ' '.join(str(row.get(k,'')) for k in
        ('relative_path','catalog_relative_path','name','task','experiment','experiment_label','artifact_experiment','artifact_experiment_label','experiment_variant','experiment_variant_label','run_id','checkpoint_id','instruction')).lower()
