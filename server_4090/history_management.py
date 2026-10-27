"""Run diagnostics lifecycle, separate from model and rollout lifetimes."""
import json
import os
from pathlib import Path
from . import storage_worker
from .storage_catalog import locations, storage_worker as execute_worker

TERMINAL = {'completed','complete','succeeded','failed','stopped','cancelled','canceled','skipped','lost'}


class HistoryManager:
    def __init__(self, config, tasks):
        self.config, self.tasks = config, tasks
        self.path = Path(config['workspace_root'])/'deleted-run-history.json'

    def deleted(self):
        try: return json.loads(self.path.read_text())
        except FileNotFoundError: return {}

    def remember(self, rows):
        data = self.deleted()
        for row in rows:
            key = row.get('path') or row.get('run_id')
            if key:
                data[key] = {k:v for k,v in row.items() if not k.startswith('_') and k not in {'latest','metrics','training_points','evaluation_points'}}
                if row.get('_source'):
                    data[key]['_source'] = {k:v for k,v in row['_source'].items() if k in {'id','root','python','ssh_host','target','max_depth','storage_location'}}
        storage_worker.atomic(self.path, data)

    def cleanup(self, operations):
        # First validate the full selection on every location. Failure keeps history visible.
        receipts=[]
        for dry_run in (True, False):
            for location, request in operations:
                request=dict(request, action='history_cleanup', dry_run=dry_run)
                if location['local']:
                    receipt=storage_worker.history_cleanup(dict(request, task_root=str(self.tasks.root)))
                else: receipt=execute_worker(location, request)
                if not dry_run: receipts.append(receipt)
        return receipts

    def location(self, target):
        target = target.lower() if isinstance(target, str) else target
        configured=locations(self.config)
        if target in {'local_4090','4x4090',None,''}: return configured['local_4090']
        if target in configured: return configured[target]
        aliases={'h100':'h100-ksy-01','h200':'h200-ali-01'}
        if aliases.get(target) in configured: return configured[aliases[target]]
        raise ValueError('Cannot determine history storage node: '+str(target))

    def task_operations(self, selected):
        operations=[]
        for task in selected:
            meta=task.get('metadata') or {}
            if task.get('type') not in {'train','eval','norm'}: continue
            local_paths=[meta[k] for k in ('artifact_run_dir',) if meta.get(k)]
            # Legacy experiment directories can be shared by several training runs.
            # Only a positively identified per-run directory may be cleaned.
            if meta.get('result_path') and task.get('type')=='eval':
                # Evaluation result metadata belongs to the rollout and is retained.
                local_paths.append(str(Path(meta['result_path']).parent))
            logs=[meta['external_log_path']] if meta.get('external_log_path') else []
            remote_target=meta.get('cluster_target') or meta.get('slurm_target')
            if meta.get('rlinf_output_dir'):
                if meta.get('runtime')=='slurm':
                    operations.append((self.location(remote_target),dict(run_paths=[meta['rlinf_output_dir']],run={'state':task['state']})))
                else: local_paths.append(meta['rlinf_output_dir'])

            remote=self.tasks._remote_slurm_log_targets(task,self.tasks.log_tail(task['id']))
            if remote:
                operations.append((self.location(remote_target),dict(log_files=list(remote[1].values()))))
            if local_paths or logs:
                operations.append((self.location('local_4090'),dict(run_paths=list(dict.fromkeys(local_paths)),log_files=logs,run={'state':task['state']})))
        return operations

    def before_tasks_deleted(self, selected):
        from .selected_resource import job_status
        for task in selected:
            meta=task.get('metadata') or {}
            if meta.get('runtime')=='slurm':
                job=meta.get('slurm_job_id') or self.tasks._slurm_job_id_from_log(self.tasks.log_tail(task['id']))
                if job:
                    target=self.config.get('cluster_targets',{}).get(meta.get('cluster_target') or meta.get('slurm_target'),{})
                    status=job_status(meta.get('slurm_submit_host') or target.get('submit_host'),job)
                    if status['state'] not in TERMINAL: raise ValueError('Slurm 运行尚未确认结束，拒绝删除历史')
        operations=self.task_operations(selected)
        ids={t['id'] for t in selected}
        inventory=storage_worker.read(Path(self.config['workspace_root'])/'artifact-run-inventory.json',{})
        linked=[r for r in inventory.get('runs',[]) if r.get('origin',{}).get('dashboard_id') in ids]
        for row in linked: operations.extend(self.run_operations(row))
        receipts=self.cleanup(operations)
        self.remember(linked)
        # Keep provenance so models remain identifiable after their process record is gone.
        root=Path(self.config['workspace_root'])/'model-provenance'
        for task in selected:
            meta=task.get('metadata') or {}
            storage_worker.atomic(root/(task['id']+'.json'),{'task_id':task['id'],'type':task['type'],'metadata':meta})
        return receipts

    def run_operations(self, run):
        if str(run.get('state','')).lower() not in TERMINAL:
            raise ValueError('只能删除已确认结束的运行历史；当前状态为 '+str(run.get('state')))
        source=run.get('_source') or {}
        target=run.get('target') or source.get('target')
        host=run.get('storage_host') or source.get('ssh_host')
        if not host or host in {'local_4090','localhost'}: target='local_4090'
        if host and host not in {'local_4090','localhost'} and not target:
            target=next((k for k,v in locations(self.config).items() if v.get('submit_host')==host),None)
            if target is None: raise ValueError('Cannot identify remote history node')
        request=dict(run_paths=[run['path']],run={'state':run['state']})
        return [(self.location(target),request)]
