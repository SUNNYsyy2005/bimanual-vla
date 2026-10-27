"""Configuration and resource queries for user-defined Slurm extension nodes."""
import concurrent.futures
import re
import subprocess

# Remote paths are validated lexically, never resolved on the Dashboard host.
PATH_FIELDS = ('workdir','dashboard_repo','openpi_repo','openpi_python','inventory_python',
               'conda_sh','dataset_root','assets_base_dir','checkpoint_base_dir','base_checkpoint',
               'cache_root','log_dir','remote_job_dir','openpi_src','openpi_client_src','lerobot_src',
               'inventory_cache_path','inventory_source_path','nas_dataset_staging_root',
               'nas_checkpoint_staging_root','tmpdir','hf_home','xdg_cache_home','pip_cache_dir')
LIST_FIELDS = ('dataset_read_roots','checkpoint_read_roots','eval_video_roots','python_fallbacks')
REQUIRED = ('submit_host','partition','node','access_mode','openpi_python','workdir','dashboard_repo',
            'dataset_root','assets_base_dir','checkpoint_base_dir','base_checkpoint','cache_root',
            'log_dir','remote_job_dir')
RESERVED = {'local','local_4090','4x4090'}


def validate_node(name, value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',name) or name in RESERVED:
        raise ValueError('invalid or reserved extension node ID')
    if not isinstance(value,dict): raise ValueError('node must be an object')
    node=dict(value)
    for key in REQUIRED:
        if not isinstance(node.get(key),str) or not node[key].strip():
            raise ValueError(f'{name}: {key} is required')
    for key,val in node.items():
        if key not in LIST_FIELDS and isinstance(val,str) and any(c in val for c in '\r\n\x00'):
            raise ValueError(f'{key}: control characters are not allowed')
    if node['access_mode'] not in ('login_shared_slurm','slurm_only'):
        raise ValueError('access_mode must be login_shared_slurm or slurm_only')
    for key in ('submit_host','host','inventory_source_host','inventory_cache_host'):
        if node.get(key) and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:-]*',node[key]):
            raise ValueError(f'{key}: use an SSH alias or user@host, without SSH options')
    for key in ('partition','node','gpu_type','conda_env'):
        if node.get(key) and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',node[key]):
            raise ValueError(f'{key}: invalid name')
    for key in PATH_FIELDS:
        if key not in REQUIRED and node.get(key) in (None, ''):
            node.pop(key, None)
            continue
        if node.get(key) and not str(node[key]).startswith('/'):
            raise ValueError(f'{key}: use an absolute path on the remote node')
    for key in LIST_FIELDS:
        if key not in node: continue
        values=node[key]
        if isinstance(values,str): values=[line.strip() for line in values.splitlines() if line.strip()]
        if not isinstance(values,list) or any(not isinstance(v,str) or not v.startswith('/') or any(c in v for c in '\r\n\x00') for v in values):
            raise ValueError(f'{key}: use a list of absolute remote paths')
        node[key]=list(dict.fromkeys(values))
    for key,default,low,high in [('gpu_count',1,1,64),('cpus_per_task',8,1,1024)]:
        val=node.get(key,default)
        if isinstance(val,bool) or not re.fullmatch(r'\d+',str(val)): raise ValueError(f'{key}: integer required')
        node[key]=int(val)
        if not low<=node[key]<=high: raise ValueError(f'{key}: expected {low}..{high}')
    mem=str(node.get('mem','64G')).upper()
    match=re.fullmatch(r'([1-9][0-9]*)([MG])',mem)
    if not match or int(match[1])*(1024 if match[2]=='G' else 1)>512*1024:
        raise ValueError('mem must be positive M/G and no more than 512G')
    node['mem']=mem
    time=str(node.get('time','24:00:00'))
    if not re.fullmatch(r'(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d',time): raise ValueError('time must be HH:MM:SS or D-HH:MM:SS')
    node['time']=time
    for key in ('gpu_type','conda_env','inventory_cache_host','inventory_source_host'):
        if node.get(key) in (None, ''): node.pop(key, None)
    node.setdefault('label',name)
    node.setdefault('openpi_repo',node['workdir'])
    return node


def resource_snapshot(targets, *, all_jobs=False, native=False):
    from server_4090.slurm_resources import query_resources
    groups={}
    for name,target in targets.items():
        groups.setdefault(target['submit_host'],[]).append((name,target))
    def query(item):
        host,rows=item
        return query_resources(host, [target for _,target in rows], all_jobs=all_jobs,
                               native=native, runner=subprocess.run)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(4,len(groups)))) as pool:
        futures={pool.submit(query,item):item[0] for item in groups.items()}
        for future in concurrent.futures.as_completed(futures):
            try: results.append(future.result())
            except Exception as exc: results.append((futures[future],1,str(exc)))
    return {'ok':all(code==0 for _,code,_ in results),'returncode':int(any(code for _,code,_ in results)),
            'output':'\n\n'.join(f'{host}\n{output}' for host,_,output in sorted(results)),
            'note':'拓展节点通过配置的 SSH 提交主机查询 Slurm；不启动远端服务。'}
