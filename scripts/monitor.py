#!/usr/bin/env python3
"""集群作业状态监控脚本。

用法:
  python monitor.py                       # 查看所有跟踪中的作业状态 + 当前队列
  python monitor.py --track <jobid> [标签]  # 开始跟踪一个作业
  python monitor.py --untrack <jobid>      # 停止跟踪
  python monitor.py --logs <jobid>         # 提交一次性诊断作业，直接在计算节点读日志尾部 + 进程状态
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SSH_HOST = "picluster"
STATE_FILE = Path(__file__).resolve().parent / ".monitor_jobs.json"
LOG_DIR = "/DATA/NAS/GPUServer/sunny/setup_jobs/logs"


def run_ssh(remote_cmd, timeout=30):
    result = subprocess.run(
        ["ssh", SSH_HOST, remote_cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"jobs": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_track(jobid, label):
    state = load_state()
    for job in state["jobs"]:
        if job["id"] == jobid:
            job["label"] = label or job.get("label", "")
            save_state(state)
            print(f"已更新跟踪: {jobid} ({job['label']})")
            return
    state["jobs"].append({"id": jobid, "label": label or ""})
    save_state(state)
    print(f"已开始跟踪: {jobid} ({label or '无标签'})")


def cmd_untrack(jobid):
    state = load_state()
    before = len(state["jobs"])
    state["jobs"] = [j for j in state["jobs"] if j["id"] != jobid]
    save_state(state)
    if len(state["jobs"]) < before:
        print(f"已停止跟踪: {jobid}")
    else:
        print(f"未找到跟踪记录: {jobid}")


def cmd_status():
    print("=== 当前队列 (squeue -u sunny) ===")
    out, err, _ = run_ssh("squeue -u sunny")
    print(out.strip() or "(队列为空)")
    if err.strip():
        print("[stderr]", err.strip())

    state = load_state()
    if not state["jobs"]:
        print("\n(未跟踪任何作业，使用 --track <jobid> 添加)")
        return

    print("\n=== 跟踪中的作业 (sacct) ===")
    ids = ",".join(j["id"] for j in state["jobs"])
    out, err, _ = run_ssh(
        f"sacct -j {ids} --format=JobID,JobName%20,State,ExitCode,Elapsed -P"
    )
    lines = [l for l in out.strip().splitlines() if l]
    if not lines:
        print("(sacct 无返回，作业号可能有误)")
        return

    header = lines[0]
    print(header.replace("|", "  "))
    label_map = {j["id"]: j.get("label", "") for j in state["jobs"]}
    for line in lines[1:]:
        parts = line.split("|")
        jobid_field = parts[0]
        base_id = jobid_field.split(".")[0]
        label = label_map.get(base_id, "")
        suffix = f"  <- {label}" if label and "." not in jobid_field else ""
        print(line.replace("|", "  ") + suffix)


def cmd_logs(jobid, lines=40, wait_timeout=120):
    print(f"提交诊断作业，读取作业 {jobid} 的日志与进程状态...")
    wrap_cmd = (
        f"for f in {LOG_DIR}/*_{jobid}.out {LOG_DIR}/*_{jobid}.err; do "
        f'[ -f "$f" ] && echo "--- $f ---" && tail -c 2000 "$f"; done; '
        f'echo "---PS---"; ps -u sunny -o pid,etime,pcpu,args | grep -E "uv|python" | grep -v grep'
    )
    submit_cmd = (
        f"sbatch -p h200 -w h200-ali-02 --job-name=monitor_logs "
        f"--cpus-per-task=1 --mem=1G --time=00:05:00 "
        f"--output={LOG_DIR}/monitorlogs_%j.out "
        f"--wrap '{wrap_cmd}'"
    )
    out, err, rc = run_ssh(submit_cmd)
    if rc != 0 or "Submitted batch job" not in out:
        print("提交诊断作业失败:", out, err)
        return
    diag_id = out.strip().split()[-1]
    print(f"诊断作业号: {diag_id}，等待完成...")

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        out, _, _ = run_ssh(f"sacct -j {diag_id} -n -o State -P")
        state = out.strip().splitlines()[0].strip() if out.strip() else ""
        if state and state not in ("RUNNING", "PENDING", ""):
            break
        time.sleep(5)
    else:
        print("诊断作业超时未完成，请稍后手动查看:", diag_id)
        return

    # NAS 挂载存在读后写延迟：诊断作业刚 COMPLETED 时，紧接着 cat
    # 结果文件可能短暂报 "No such file or directory"，需要重试几次。
    out = ""
    for attempt in range(6):
        out, err, _ = run_ssh(f"cat {LOG_DIR}/monitorlogs_{diag_id}.out")
        if out.strip() or "No such file or directory" not in err:
            break
        time.sleep(5)

    print("\n=== 结果 ===")
    print(out.strip() or "(无输出，NAS 可能仍有延迟，可稍后重试: python monitor.py --logs " + jobid + ")")
    if err.strip() and "No such file or directory" not in err:
        print("[stderr]", err.strip())


def main():
    parser = argparse.ArgumentParser(description="集群作业状态监控")
    parser.add_argument("--track", metavar="JOBID")
    parser.add_argument("--label", metavar="LABEL", default="", help="配合 --track 使用的标签")
    parser.add_argument("--untrack", metavar="JOBID")
    parser.add_argument("--logs", metavar="JOBID", help="提交诊断作业查看该作业最新日志和进程状态")
    args = parser.parse_args()

    try:
        if args.track:
            cmd_track(args.track, args.label)
        elif args.untrack:
            cmd_untrack(args.untrack)
        elif args.logs:
            cmd_logs(args.logs)
        else:
            cmd_status()
    except subprocess.TimeoutExpired:
        print("SSH 命令超时，请检查网络或服务器状态。")
        sys.exit(1)


if __name__ == "__main__":
    main()
