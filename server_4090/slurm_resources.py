"""Read-only Slurm allocation snapshots, including pending jobs and GPU indices."""
import re
import shlex
import unicodedata


ACTIVE_STATES = {'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'}


def snapshot_command(targets, native=False):
    nodes = ','.join(dict.fromkeys(t['node'] for t in targets))
    partitions = ','.join(dict.fromkeys(t['partition'] for t in targets))
    # Collect the partition queue without a node/user filter: pending jobs may
    # not have a NodeList yet, and other users account for allocated GPUs.
    return '\n'.join([
        'overall=0',
        'section() {',
        '  name="$1"; shift',
        '  printf "__BEGIN_%s__\\n" "$name"',
        '  "$@" 2>&1',
        '  code=$?',
        '  printf "__RC_%s__=%s\\n" "$name" "$code"',
        '  if [ "$code" -ne 0 ]; then overall=1; fi',
        '}',
        'section META sh -c \'id -un; date -Is\'',
        'section NODES scontrol show node -o ' + shlex.quote(nodes),
        'printf "__BEGIN_JOBS__\\n"',
        "jobs=$(squeue -p " + shlex.quote(partitions) + " -h -o '%i|%P|%j|%u|%T|%M|%R|%b' 2>&1)",
        'code=$?',
        'printf "%s\\n__RC_JOBS__=%s\\n" "$jobs" "$code"',
        'if [ "$code" -ne 0 ]; then overall=1; fi',
        'printf "__BEGIN_DETAILS__\\n"',
        'details_code=0',
        'if [ "$code" -eq 0 ]; then',
        "  while IFS='|' read -r jobid part name user state rest; do",
        '    case "$state" in RUNNING|COMPLETING|CONFIGURING|SUSPENDED)',
        '      scontrol show job -dd -o "$jobid" 2>&1 || details_code=1 ;;',
        '    esac',
        '  done <<EOF_SLURM_JOBS',
        '$jobs',
        'EOF_SLURM_JOBS',
        'else details_code=1; fi',
        'printf "__RC_DETAILS__=%s\\n" "$details_code"',
        'if [ "$details_code" -ne 0 ]; then overall=1; fi',
        'section QUOTA myquota',
        *(['section NATIVE resources'] if native else []),
        'exit "$overall"',
    ])


def _fields(line):
    return dict(re.findall(r'(?:^|\s)([A-Za-z_]+)=([^\s]*)', line))


def _gpu_count(tres):
    entries = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
    if 'gres/gpu' in entries:
        return int(entries['gres/gpu'])
    return sum(int(v) for k, v in entries.items() if k.startswith('gres/gpu:'))


def _table(headers, rows):
    def width(value):
        return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in value)
    widths = [max(width(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    def line(row):
        return '  '.join(value + ' ' * (widths[i] - width(value))
                         for i, value in enumerate(row)).rstrip()
    return [line(headers), '─' * (sum(widths) + 2 * (len(headers)-1)), *map(line, rows)]


def render_snapshot(raw, targets, all_jobs=False):
    sections, codes, current = {}, {}, None
    for line in raw.splitlines():
        begin = re.fullmatch(r'__BEGIN_(\w+)__', line)
        rc = re.fullmatch(r'__RC_(\w+)__=(\d+)', line)
        if begin:
            current = begin[1]
            sections[current] = []
        elif rc:
            codes[rc[1]] = int(rc[2])
            current = None
        elif current:
            sections[current].append(line)
    errors = [f'{key} 查询失败或结果不完整 (rc={codes.get(key, "missing")})'
              for key in ('META', 'NODES', 'JOBS', 'DETAILS', 'QUOTA')
              if codes.get(key) != 0]
    if 'NATIVE' in sections and codes.get('NATIVE') != 0:
        errors.append('resources 查询失败')
    meta = sections.get('META', [])
    user = meta[0] if meta else ''
    output = ['=== 算力额度 ===', *sections.get('QUOTA', ['(查询未返回)']),
              '\n=== GPU 卡状态 ===',
              '更新: ' + (meta[1] if len(meta) > 1 else '未知')]
    node_rows = []
    seen = set()
    for line in sections.get('NODES', []):
        f = _fields(line)
        if 'NodeName' not in f:
            continue
        seen.add(f['NodeName'])
        # A missing allocation field is unknown, never zero/idle.
        total = _gpu_count(f['CfgTRES']) if 'CfgTRES' in f else None
        used = _gpu_count(f['AllocTRES']) if 'AllocTRES' in f else None
        free = max(0, total-used) if total is not None and used is not None else '?'
        def gib(key):
            return f"{int(f[key])/1024:.0f}" if f.get(key, '').isdigit() else '?'
        node_rows.append([f['NodeName'], f.get('State', '?'),
                          f"{used if used is not None else '?'}/{total if total is not None else '?'}",
                          str(free), f"{f.get('CPUAlloc', '?')}/{f.get('CPUTot', '?')}",
                          f"{gib('AllocMem')}/{gib('RealMemory')}"])
    for target in targets:
        if target['node'] not in seen:
            errors.append(f"节点 {target['node']} 未返回数据")
    output.extend(_table(['NODE', 'STATE', 'GPU_USED/TOTAL', 'FREE', 'CPU_USED/TOTAL', 'MEM_GiB'], node_rows))
    output.append('GPU 按 Slurm 分配统计；FREE 为未分配卡数，调度仍受节点状态和优先级限制。')
    details = {}
    for line in sections.get('DETAILS', []):
        f = _fields(line)
        if 'JobId' not in f:
            continue
        # squeue uses array IDs, while scontrol JobId is the internal ID.
        jobid = (f"{f['ArrayJobId']}_{f['ArrayTaskId']}" if 'ArrayTaskId' in f else f['JobId'])
        allocations = re.findall(r'\bNodes=(\S+).*?\bGRES=(\S+)', line)
        indices = ', '.join(f'{node}:{idx}' for node, gres in allocations
                            for idx in re.findall(r'\(IDX:([^)]*)\)', gres))
        no_gpu = 'AllocTRES' in f and _gpu_count(f['AllocTRES']) == 0
        details[jobid] = (f.get('AllocTRES', '?'), indices or ('-' if no_gpu else '未报告'))
    jobs = [line.split('|') for line in sections.get('JOBS', []) if len(line.split('|')) == 8]

    def table(title, selected):
        output.append('\n=== ' + title + ' ===')
        rows = []
        for job in selected:
            jid, part, name, owner, state, elapsed, reason, gres = job
            allocated, indices = details.get(jid, ('未知', '未报告') if state in ACTIVE_STATES else ('未分配', '-'))
            gpu = str(_gpu_count(allocated)) if allocated not in ('未知', '未分配', '?') else gres.replace('gres/', '')
            if indices.startswith(reason + ':'):
                indices = indices[len(reason)+1:]
            rows.append([jid, owner, state, elapsed, gpu, indices, reason, name])
        if rows:
            output.extend(_table(['JOBID', 'USER', 'STATE', 'ELAPSED', 'GPU', 'GPU_IDX', 'NODE / REASON', 'NAME'], rows))
        if not selected:
            output.append('(无)' if codes.get('JOBS') == 0 else '(查询失败，不能判断是否有作业)')

    table('运行作业（全部用户）', [j for j in jobs if j[4] in ACTIVE_STATES])
    table(f'我的队列 · {user}（含排队作业）', [j for j in jobs if j[3] == user])
    if all_jobs:
        table('配置分区完整队列（全部用户）', jobs)
    for section, title in [('NATIVE', '集群原生 resources')]:
        if section in sections:
            output.extend(['\n=== ' + title + ' ===', *sections[section]])
    if errors:
        output.extend(['\n=== 查询警告 ===', *errors])
        for key in ('META', 'NODES', 'JOBS', 'DETAILS'):
            if codes.get(key) != 0:
                output.extend(sections.get(key, []))
    return '\n'.join(output), bool(errors)


def query_resources(host, targets, *, all_jobs, native, runner):
    result = runner(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                     snapshot_command(targets, native)], text=True, capture_output=True, timeout=45)
    output, failed = render_snapshot(result.stdout, targets, all_jobs)
    if result.stderr:
        output += '\n' + result.stderr
    return host, result.returncode or int(failed), output
