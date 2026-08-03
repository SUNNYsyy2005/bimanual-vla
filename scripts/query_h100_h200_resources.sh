#!/usr/bin/env bash
# 查询 H100/H200 配额、Slurm GPU 占用和相关作业。
#
# 默认行为：
#   - 在 login-server 上运行时直接查询。
#   - 在未安装 Slurm 命令的机器上运行时，通过 ssh login-server 查询。
#
# 本脚本只执行只读命令，不会提交或取消 Slurm 作业，也不会登录计算节点运行 nvidia-smi。

set -uo pipefail

PROGRAM_NAME="${0##*/}"
QUERY_HOST="${SLURM_QUERY_HOST:-login-server}"
QUERY_USER=""
WATCH_INTERVAL="0"
FORCE_LOCAL=0
SHOW_ALL_JOBS=0
SHOW_NATIVE_RESOURCES=0
COMPACT=0
SSH_CONNECT_TIMEOUT="${SLURM_QUERY_SSH_TIMEOUT:-10}"
REMOTE_COMMAND_TIMEOUT="${SLURM_QUERY_COMMAND_TIMEOUT:-20}"

usage() {
    cat <<EOF_USAGE
用法: $PROGRAM_NAME [选项]

查询 H100/H200 额度、节点状态、GPU 卡占用和 Slurm 作业。

选项:
  -H, --host HOST       Slurm 登录节点 SSH 别名，默认: ${QUERY_HOST}
  -u, --user USER       要显示的用户作业，默认: SSH 登录用户
  -w, --watch SECONDS   每隔 SECONDS 秒刷新，Ctrl-C 退出
  -a, --all-jobs        显示 h100/h200 分区的完整队列
  -c, --compact         精简输出（不显示用户作业明细）
      --native          附加显示集群原生 resources 输出
      --local           强制在当前机器直接执行 Slurm 命令
  -h, --help            显示帮助

环境变量:
  SLURM_QUERY_HOST              默认 SSH 主机，默认 login-server
  SLURM_QUERY_SSH_TIMEOUT       SSH 连接超时秒数，默认 10
  SLURM_QUERY_COMMAND_TIMEOUT   单个远端命令超时秒数，默认 20

示例:
  $PROGRAM_NAME
  $PROGRAM_NAME -w 10
  $PROGRAM_NAME -u sunny --all-jobs
  $PROGRAM_NAME --host login-server --native
  $PROGRAM_NAME --local

说明:
  GPU 占用按 Slurm GRES 的 CfgTRES/AllocTRES 统计；这里的“空闲卡”表示
  Slurm 尚未分配的卡数；GPU_IDX 显示已运行作业的 Slurm 物理卡索引。
EOF_USAGE
}

warn() {
    printf '警告: %s\n' "$*" >&2
}

die() {
    printf '错误: %s\n' "$*" >&2
    exit 1
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while (($# > 0)); do
    case "$1" in
        -H|--host)
            (($# >= 2)) || die "$1 需要一个主机名"
            QUERY_HOST="$2"
            shift 2
            ;;
        -u|--user)
            (($# >= 2)) || die "$1 需要一个用户名"
            QUERY_USER="$2"
            shift 2
            ;;
        -w|--watch)
            (($# >= 2)) || die "$1 需要刷新间隔（秒）"
            WATCH_INTERVAL="$2"
            shift 2
            ;;
        -a|--all-jobs)
            SHOW_ALL_JOBS=1
            shift
            ;;
        -c|--compact)
            COMPACT=1
            shift
            ;;
        --native)
            SHOW_NATIVE_RESOURCES=1
            shift
            ;;
        --local)
            FORCE_LOCAL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            die "未知选项: $1（使用 --help 查看帮助）"
            ;;
    esac
done

(($# == 0)) || die "不接受位置参数: $*"
[[ -z "$QUERY_USER" || "$QUERY_USER" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "用户名包含不支持的字符: $QUERY_USER"
[[ "$QUERY_HOST" =~ ^[A-Za-z0-9._@-]+$ ]] \
    || die "SSH 主机名包含不支持的字符: $QUERY_HOST"
is_nonnegative_integer "$WATCH_INTERVAL" \
    || die "刷新间隔必须是非负整数: $WATCH_INTERVAL"
is_positive_integer "$SSH_CONNECT_TIMEOUT" \
    || die "SSH 超时必须是正整数: $SSH_CONNECT_TIMEOUT"
is_positive_integer "$REMOTE_COMMAND_TIMEOUT" \
    || die "命令超时必须是正整数: $REMOTE_COMMAND_TIMEOUT"

BACKEND="remote"
if ((FORCE_LOCAL)); then
    BACKEND="local"
elif command -v sinfo >/dev/null 2>&1 \
    && command -v squeue >/dev/null 2>&1 \
    && command -v scontrol >/dev/null 2>&1; then
    BACKEND="local"
fi

if [[ "$BACKEND" == "remote" ]]; then
    command -v ssh >/dev/null 2>&1 || die "当前机器没有 ssh，无法连接 $QUERY_HOST"
fi

# 用显式分节标记返回一次快照，保证节点、作业和额度尽量来自同一查询时刻。
read -r -d '' SNAPSHOT_SCRIPT <<EOF_SNAPSHOT || true
set +e
COMMAND_TIMEOUT='$REMOTE_COMMAND_TIMEOUT'

run_limited() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "\${COMMAND_TIMEOUT}s" "\$@"
    else
        "\$@"
    fi
}

printf '%s\n' '__BEGIN_META__'
printf 'host=%s\n' "\$(hostname 2>/dev/null || echo unknown)"
printf 'user=%s\n' "\$(id -un 2>/dev/null || echo unknown)"
printf 'time=%s\n' "\$(date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date)"
printf '%s\n' '__END_META__'

printf '%s\n' '__BEGIN_QUOTA__'
if command -v myquota >/dev/null 2>&1; then
    run_limited myquota 2>&1
    quota_rc=\$?
else
    echo 'myquota 命令不存在'
    quota_rc=127
fi
printf '%s=%s\n' '__RC_QUOTA__' "\$quota_rc"
printf '%s\n' '__END_QUOTA__'

printf '%s\n' '__BEGIN_NODES__'
node_rc=0
if ! command -v sinfo >/dev/null 2>&1 || ! command -v scontrol >/dev/null 2>&1; then
    echo '__ERROR__|sinfo 或 scontrol 命令不存在'
    node_rc=127
else
    node_list="\$(run_limited sinfo -N -h -p h100,h200 -o '%N' 2>&1)"
    sinfo_rc=\$?
    if ((sinfo_rc != 0)); then
        printf '__ERROR__|sinfo 查询失败: %s\n' "\$node_list"
        node_rc=\$sinfo_rc
    else
        node_list="\$(printf '%s\n' "\$node_list" | sed '/^[[:space:]]*\$/d' | sort -u)"
        if [[ -z "\$node_list" ]]; then
            echo '__ERROR__|h100/h200 分区未返回任何节点'
            node_rc=1
        else
            while IFS= read -r node; do
                [[ -n "\$node" ]] || continue
                node_info="\$(run_limited scontrol show node -o "\$node" 2>&1)"
                current_rc=\$?
                if ((current_rc == 0)); then
                    printf '%s\n' "\$node_info"
                else
                    printf '__NODE_ERROR__|%s|%s\n' "\$node" "\$node_info"
                    node_rc=\$current_rc
                fi
            done <<< "\$node_list"
        fi
    fi
fi
printf '%s=%s\n' '__RC_NODES__' "\$node_rc"
printf '%s\n' '__END_NODES__'

printf '%s\n' '__BEGIN_JOBS__'
jobs_output=''
if command -v squeue >/dev/null 2>&1; then
    jobs_output="\$(run_limited squeue -p h100,h200 -h \
        -o '%i|%P|%j|%u|%T|%M|%l|%D|%R|%b|%C|%m' 2>&1)"
    jobs_rc=\$?
    printf '%s\n' "\$jobs_output"
else
    echo '__ERROR__|squeue 命令不存在'
    jobs_rc=127
fi
printf '%s=%s\n' '__RC_JOBS__' "\$jobs_rc"
printf '%s\n' '__END_JOBS__'

# scontrol -dd 会给出 GRES=...\(IDX:n\)，可定位 Slurm 分配的物理卡编号。
printf '%s\n' '__BEGIN_JOB_DETAILS__'
details_rc=0
if ((jobs_rc == 0)) && command -v scontrol >/dev/null 2>&1; then
    while IFS='|' read -r job_id _ _ _ job_state _ _ _ _ job_gres _ _; do
        [[ "\$job_state" == 'RUNNING' && "\$job_gres" == *gpu* ]] || continue
        job_detail="\$(run_limited scontrol show job -dd -o "\$job_id" 2>&1)"
        detail_rc=\$?
        if ((detail_rc == 0)); then
            printf '%s|%s\n' "\$job_id" "\$job_detail"
        else
            printf '__JOB_ERROR__|%s|%s\n' "\$job_id" "\$job_detail"
            details_rc=\$detail_rc
        fi
    done <<< "\$jobs_output"
elif ((jobs_rc != 0)); then
    echo '__ERROR__|squeue 查询失败，无法查询 GPU 物理索引'
    details_rc=\$jobs_rc
else
    echo '__ERROR__|scontrol 命令不存在，无法查询 GPU 物理索引'
    details_rc=127
fi
printf '%s=%s\n' '__RC_JOB_DETAILS__' "\$details_rc"
printf '%s\n' '__END_JOB_DETAILS__'

printf '%s\n' '__BEGIN_NATIVE__'
if [[ '$SHOW_NATIVE_RESOURCES' == '1' ]]; then
    if command -v resources >/dev/null 2>&1; then
        run_limited resources 2>&1
        native_rc=\$?
    else
        echo 'resources 命令不存在'
        native_rc=127
    fi
else
    native_rc=0
fi
printf '%s=%s\n' '__RC_NATIVE__' "\$native_rc"
printf '%s\n' '__END_NATIVE__'
EOF_SNAPSHOT

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${PROGRAM_NAME}.XXXXXX")" \
    || die "无法创建临时目录"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM
SNAPSHOT_FILE="$TMP_DIR/snapshot.txt"
SNAPSHOT_ERR="$TMP_DIR/snapshot.err"

collect_snapshot() {
    local attempt rc=0 max_attempts=1
    [[ "$BACKEND" == "remote" ]] && max_attempts=3

    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        : >"$SNAPSHOT_FILE"
        : >"$SNAPSHOT_ERR"

        if [[ "$BACKEND" == "local" ]]; then
            bash -s >"$SNAPSHOT_FILE" 2>"$SNAPSHOT_ERR" <<<"$SNAPSHOT_SCRIPT"
            rc=$?
        else
            ssh \
                -o BatchMode=yes \
                -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
                -o ServerAliveInterval=10 \
                -o ServerAliveCountMax=2 \
                "$QUERY_HOST" 'bash -s' \
                >"$SNAPSHOT_FILE" 2>"$SNAPSHOT_ERR" <<<"$SNAPSHOT_SCRIPT"
            rc=$?
        fi
        ((rc == 0)) && return 0
        ((attempt < max_attempts)) && sleep "$attempt"
    done
    return "$rc"
}

section() {
    local name="$1"
    awk -v begin="__BEGIN_${name}__" -v end="__END_${name}__" '
        $0 == begin { inside=1; next }
        $0 == end   { inside=0; exit }
        inside      { print }
    ' "$SNAPSHOT_FILE"
}

section_rc() {
    local name="$1"
    awk -F= -v marker="__RC_${name}__" '
        $1 == marker { print $2; found=1; exit }
        END { if (!found) print 255 }
    ' "$SNAPSHOT_FILE"
}

meta_value() {
    local key="$1"
    section META | sed -n "s/^${key}=//p" | head -n 1
}

get_field() {
    local line="$1"
    local key="$2"
    local regex="(^|[[:space:]])${key}=([^[:space:]]+)"
    if [[ "$line" =~ $regex ]]; then
        printf '%s' "${BASH_REMATCH[2]}"
    fi
}

tres_value() {
    local tres="$1"
    local key="$2"
    local regex="(^|,)${key}=([^,]+)"
    if [[ "$tres" =~ $regex ]]; then
        printf '%s' "${BASH_REMATCH[2]}"
    fi
}

as_uint() {
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        printf '%s' "$1"
    else
        printf '0'
    fi
}

gib_from_mib() {
    local mib
    mib="$(as_uint "${1:-0}")"
    awk -v mib="$mib" 'BEGIN { printf "%.0f", mib / 1024 }'
}

short_gres() {
    local gres="${1:-N/A}"
    if [[ -z "$gres" || "$gres" == "N/A" ]]; then
        printf '-'
    else
        gres="${gres//gres\//}"
        printf '%s' "$gres"
    fi
}

print_rule() {
    printf '%s\n' '----------------------------------------------------------------------------------------------------'
}

print_node_status() {
    local rc
    rc="$(section_rc NODES)"
    printf '\n=== H100/H200 节点与 GPU 占用（Slurm GRES）===\n'

    if [[ "$rc" != "0" ]]; then
        warn "节点查询未完全成功（返回码 $rc）"
    fi

    local -a node_lines=()
    mapfile -t node_lines < <(section NODES)

    local -A total_by_type=([H100]=0 [H200]=0)
    local -A alloc_by_type=([H100]=0 [H200]=0)
    local -A sched_by_type=([H100]=0 [H200]=0)
    local -A nodes_by_type=([H100]=0 [H200]=0)
    local -a rows=()
    local line node state gres cfg_tres alloc_tres partitions gpu_type
    local total alloc free sched_free cpu_alloc cpu_total real_mem alloc_mem
    local mem_text type_label

    for line in "${node_lines[@]}"; do
        if [[ "$line" == __ERROR__\|* || "$line" == __NODE_ERROR__\|* ]]; then
            warn "${line#*|}"
            continue
        fi
        [[ "$line" == *"NodeName="* ]] || continue

        node="$(get_field "$line" NodeName)"
        state="$(get_field "$line" State)"
        gres="$(get_field "$line" Gres)"
        cfg_tres="$(get_field "$line" CfgTRES)"
        alloc_tres="$(get_field "$line" AllocTRES)"
        partitions="$(get_field "$line" Partitions)"

        gpu_type=""
        total="$(tres_value "$cfg_tres" 'gres/gpu:h100')"
        if [[ -n "$total" ]]; then
            gpu_type="h100"
        else
            total="$(tres_value "$cfg_tres" 'gres/gpu:h200')"
            [[ -n "$total" ]] && gpu_type="h200"
        fi
        if [[ -z "$gpu_type" && "$gres" =~ gpu:(h100|h200):([0-9]+) ]]; then
            gpu_type="${BASH_REMATCH[1]}"
            total="${BASH_REMATCH[2]}"
        fi
        if [[ -z "$gpu_type" ]]; then
            if [[ "$partitions" == *h100* ]]; then
                gpu_type="h100"
            elif [[ "$partitions" == *h200* ]]; then
                gpu_type="h200"
            else
                gpu_type="unknown"
            fi
        fi

        total="$(as_uint "${total:-$(tres_value "$cfg_tres" 'gres/gpu')}")"
        alloc="$(tres_value "$alloc_tres" "gres/gpu:${gpu_type}")"
        alloc="$(as_uint "${alloc:-$(tres_value "$alloc_tres" 'gres/gpu')}")"
        ((alloc > total)) && alloc="$total"
        free=$((total - alloc))

        # DRAIN/DOWN 等状态下，未分配的卡并不代表可提交作业立即调度。
        sched_free="$free"
        if [[ "$state" =~ (DOWN|DRAIN|FAIL|MAINT|POWER|UNKNOWN|NOT_RESPONDING) ]]; then
            sched_free=0
        fi

        cpu_alloc="$(as_uint "$(get_field "$line" CPUAlloc)")"
        cpu_total="$(as_uint "$(get_field "$line" CPUTot)")"
        real_mem="$(as_uint "$(get_field "$line" RealMemory)")"
        alloc_mem="$(as_uint "$(get_field "$line" AllocMem)")"
        mem_text="$(gib_from_mib "$alloc_mem")/$(gib_from_mib "$real_mem")G"
        type_label="${gpu_type^^}"

        rows+=("$type_label|$node|${state:-UNKNOWN}|$alloc/$total|$free|$sched_free|$cpu_alloc/$cpu_total|$mem_text")
        if [[ "$type_label" == "H100" || "$type_label" == "H200" ]]; then
            total_by_type[$type_label]=$((total_by_type[$type_label] + total))
            alloc_by_type[$type_label]=$((alloc_by_type[$type_label] + alloc))
            sched_by_type[$type_label]=$((sched_by_type[$type_label] + sched_free))
            nodes_by_type[$type_label]=$((nodes_by_type[$type_label] + 1))
        fi
    done

    if ((${#rows[@]} == 0)); then
        printf '(没有可显示的 H100/H200 节点数据)\n'
        return
    fi

    printf '%-5s  %-15s  %-14s  %-10s  %-6s  %-10s  %-11s  %-14s\n' \
        TYPE NODE STATE GPU_USED FREE SCHED_FREE CPU_USED MEM_USED
    print_rule
    printf '%s\n' "${rows[@]}" | while IFS='|' read -r type node state used free_count sched cpu mem; do
        printf '%-5s  %-15s  %-14s  %-10s  %-6s  %-10s  %-11s  %-14s\n' \
            "$type" "$node" "$state" "$used" "$free_count" "$sched" "$cpu" "$mem"
    done

    printf '\n汇总:\n'
    local label idle rate
    for label in H100 H200; do
        idle=$((total_by_type[$label] - alloc_by_type[$label]))
        if ((total_by_type[$label] > 0)); then
            rate="$(awk -v used="${alloc_by_type[$label]}" -v total="${total_by_type[$label]}" \
                'BEGIN { printf "%.1f", used * 100 / total }')"
        else
            rate="0.0"
        fi
        printf '  %-4s 节点 %d，GPU 总计 %d，占用 %d，未分配 %d，可调度空闲 %d，占用率 %s%%\n' \
            "$label" "${nodes_by_type[$label]}" "${total_by_type[$label]}" \
            "${alloc_by_type[$label]}" "$idle" "${sched_by_type[$label]}" "$rate"
    done
}

print_job_rows() {
    local title="$1"
    local mode="$2"
    local selected_user="$3"
    local -n source_lines="$4"
    local -n gpu_idx_map="$5"
    local -a selected=()
    local line jobid partition job_name user state elapsed limit nodes reason gres cpus min_mem extra idx

    for line in "${source_lines[@]}"; do
        [[ "$line" == __ERROR__\|* ]] && continue
        IFS='|' read -r jobid partition job_name user state elapsed limit nodes reason gres cpus min_mem extra <<<"$line"
        [[ -n "$jobid" && -n "$partition" ]] || continue

        case "$mode" in
            gpu-running)
                [[ "$state" == "RUNNING" && "$gres" == *gpu* ]] || continue
                ;;
            user)
                [[ "$user" == "$selected_user" ]] || continue
                ;;
            all)
                ;;
        esac
        idx="${gpu_idx_map[$jobid]:--}"
        selected+=("$jobid|$partition|$user|$state|$elapsed|$(short_gres "$gres")|$idx|$reason|$job_name")
    done

    printf '\n=== %s ===\n' "$title"
    if ((${#selected[@]} == 0)); then
        printf '(无)\n'
        return
    fi

    printf '%-13s %-5s %-13s %-10s %-12s %-13s %-9s %-24s %s\n' \
        JOBID PART USER STATE ELAPSED GPU GPU_IDX NODE_OR_REASON NAME
    print_rule
    printf '%s\n' "${selected[@]}" | while IFS='|' read -r jobid part user state elapsed gpu idx reason name; do
        printf '%-13.13s %-5.5s %-13.13s %-10.10s %-12.12s %-13.13s %-9.9s %-24.24s %s\n' \
            "$jobid" "$part" "$user" "$state" "$elapsed" "$gpu" "$idx" "$reason" "$name"
    done
}

print_jobs() {
    local effective_user="$1"
    local rc details_rc line jobid detail gres_detail
    rc="$(section_rc JOBS)"
    if [[ "$rc" != "0" ]]; then
        warn "作业队列查询失败（返回码 $rc）"
        section JOBS | sed '/^__RC_JOBS__=/d' >&2
        return
    fi

    local -a job_lines=() detail_lines=()
    local -A gpu_indices=()
    mapfile -t job_lines < <(section JOBS | sed '/^__RC_JOBS__=/d')
    mapfile -t detail_lines < <(section JOB_DETAILS | sed '/^__RC_JOB_DETAILS__=/d')
    details_rc="$(section_rc JOB_DETAILS)"
    [[ "$details_rc" == "0" ]] || warn "部分 GPU 物理索引查询失败（返回码 $details_rc）"

    for line in "${detail_lines[@]}"; do
        [[ "$line" == __ERROR__\|* || "$line" == __JOB_ERROR__\|* ]] && continue
        jobid="${line%%|*}"
        detail="${line#*|}"
        gres_detail="$(get_field "$detail" GRES)"
        if [[ "$gres_detail" =~ \(IDX:([^\)]+)\) ]]; then
            gpu_indices[$jobid]="${BASH_REMATCH[1]}"
        fi
    done

    print_job_rows '正在占用 GPU 的作业（全部用户）' gpu-running "$effective_user" job_lines gpu_indices

    if ((COMPACT == 0)); then
        print_job_rows "用户 ${effective_user} 的 H100/H200 作业" user "$effective_user" job_lines gpu_indices
    fi
    if ((SHOW_ALL_JOBS)); then
        print_job_rows 'H100/H200 分区完整队列' all "$effective_user" job_lines gpu_indices
    fi
}

print_quota() {
    local remote_user="$1"
    local effective_user="$2"
    local rc
    rc="$(section_rc QUOTA)"
    printf '\n=== 算力额度（myquota）===\n'
    if [[ "$rc" == "0" ]]; then
        section QUOTA | sed '/^__RC_QUOTA__=/d'
    else
        warn "myquota 查询失败（返回码 $rc）"
        section QUOTA | sed '/^__RC_QUOTA__=/d'
    fi
    if [[ "$effective_user" != "$remote_user" ]]; then
        printf '注意: myquota 始终显示 SSH 登录用户 %s 的额度；-u %s 只过滤作业列表。\n' \
            "$remote_user" "$effective_user"
    fi
}

query_once() {
    local collect_rc=0
    collect_snapshot || collect_rc=$?
    if ((collect_rc != 0)); then
        printf '错误: 无法从 %s 获取 Slurm 快照（返回码 %d）\n' \
            "$([[ "$BACKEND" == "local" ]] && echo 当前机器 || echo "$QUERY_HOST")" "$collect_rc" >&2
        if [[ -s "$SNAPSHOT_ERR" ]]; then
            sed 's/^/  /' "$SNAPSHOT_ERR" >&2
        fi
        return "$collect_rc"
    fi
    if ! grep -q '^__BEGIN_META__$' "$SNAPSHOT_FILE"; then
        warn "返回内容不完整，未找到快照标记"
        [[ -s "$SNAPSHOT_ERR" ]] && sed 's/^/  /' "$SNAPSHOT_ERR" >&2
        return 1
    fi

    local remote_host remote_user query_time effective_user source_label
    remote_host="$(meta_value host)"
    remote_user="$(meta_value user)"
    query_time="$(meta_value time)"
    effective_user="${QUERY_USER:-$remote_user}"
    if [[ "$BACKEND" == "local" ]]; then
        source_label="local"
    else
        source_label="ssh:$QUERY_HOST"
    fi

    printf 'H100/H200 资源查询 | 时间: %s | Slurm: %s (%s) | 作业用户: %s\n' \
        "${query_time:-unknown}" "${remote_host:-unknown}" "$source_label" "$effective_user"
    print_rule

    print_quota "$remote_user" "$effective_user"
    print_node_status
    print_jobs "$effective_user"

    if ((SHOW_NATIVE_RESOURCES)); then
        local native_rc
        native_rc="$(section_rc NATIVE)"
        printf '\n=== 集群原生 resources 输出 ===\n'
        section NATIVE | sed '/^__RC_NATIVE__=/d'
        [[ "$native_rc" == "0" ]] || warn "resources 查询失败（返回码 $native_rc）"
    fi

    printf '\n说明: GPU_IDX 是 Slurm 分配的物理卡索引；表中不包含 nvidia-smi 的实时利用率/显存占用。\n'
}

last_rc=0
while :; do
    if [[ "$WATCH_INTERVAL" != "0" && -t 1 ]]; then
        printf '\033[2J\033[H'
    fi
    query_once || last_rc=$?

    [[ "$WATCH_INTERVAL" != "0" ]] || break
    printf '\n%s 秒后刷新，按 Ctrl-C 退出...\n' "$WATCH_INTERVAL"
    sleep "$WATCH_INTERVAL" || break
done

exit "$last_rc"
