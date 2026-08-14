#!/usr/bin/env python3
import argparse, json, os, sys, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(os.environ.get("ROBOTWIN_PROJECT", "/home/sunny/robotwin_ws/RoboTwin"))
os.chdir(PROJECT)
sys.path.append(str(PROJECT))
sys.path.append(str(PROJECT / "policy"))
sys.path.append(str(PROJECT / "description" / "utils"))

from envs import CONFIGS_PATH  # noqa: E402
from script.eval_policy import class_decorator, get_embodiment_config, get_camera_config  # noqa: E402


def _bool(v):
    return str(v).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def prepare_args(task_name, task_config, *, video=False, step_limit=None):
    with open(PROJECT / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["policy_name"] = "replay_lerobot"
    args["eval_mode"] = True
    args["eval_video_log"] = bool(video)
    args["eval_result_log"] = True
    args["observation_joint_state_source"] = os.getenv("ROBOTWIN_OBSERVATION_STATE_SOURCE") or "real_qpos"
    if step_limit is not None:
        os.environ["ROBOTWIN_EVAL_STEP_LIMIT"] = str(step_limit)

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # Camera dimensions are normally filled by eval_policy.main before setup_demo.
    with open(Path(CONFIGS_PATH) / "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)
    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_type]["h"]
    args["head_camera_w"] = camera_config[head_type]["w"]
    return args


def maybe_start_video(env, args, out_dir):
    if not args.get("eval_video_log"):
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env.eval_video_path = str(out_dir)
    camera_config = get_camera_config(args["camera"]["head_camera_type"])
    video_size = f"{camera_config['w']}x{camera_config['h']}"
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", video_size, "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
        "-vcodec", "libx264", "-crf", "23", str(out_dir / "replay.mp4")
    ], stdin=subprocess.PIPE)
    env._set_eval_video_ffmpeg(ffmpeg)
    return ffmpeg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--task-name", default="put_bottles_dustbin")
    ap.add_argument("--task-config", default="put_bottles_dustbin_piper_eval1_video")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out-dir", default="/home/sunny/replay_lerobot_actions_out")
    args_ns = ap.parse_args()

    parquet = Path(args_ns.dataset) / "data" / "chunk-000" / f"episode_{args_ns.episode:06d}.parquet"
    df = pd.read_parquet(parquet, columns=["action"])
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    max_steps = args_ns.max_steps or len(actions)
    actions = actions[:max_steps]

    args = prepare_args(args_ns.task_name, args_ns.task_config, video=args_ns.video, step_limit=len(actions))
    env = class_decorator(args_ns.task_name)
    env.suc = 0
    env.test_num = 0
    env.setup_demo(now_ep_num=args_ns.episode, seed=args_ns.seed, is_test=True, **args)
    if args_ns.instruction:
        env.set_instruction(args_ns.instruction)
    if hasattr(env, "_set_motion_trace_label"):
        env._set_motion_trace_label(f"replay_seed{args_ns.seed}_ep{args_ns.episode}")
    maybe_start_video(env, args, Path(args_ns.out_dir))

    print("[REPLAY] start", {"parquet": str(parquet), "actions": len(actions), "seed": args_ns.seed, "chunk": args_ns.chunk, "step_lim": env.step_lim})
    ok = True
    while env.take_action_cnt < len(actions):
        t = int(env.take_action_cnt)
        chunk = actions[t:t + args_ns.chunk]
        if len(chunk) == 0:
            break
        done = env.take_action_chunk(chunk)
        if not done:
            ok = False
            print("[REPLAY] take_action_chunk returned false at", t, flush=True)
            break
        if env.eval_success:
            break
    if getattr(env, "eval_video_ffmpeg", None) is not None:
        env._del_eval_video_ffmpeg()
    official = bool(env.check_success())
    strict = bool(getattr(env, "strict_success", lambda: False)())
    diag = getattr(env, "strict_diagnostics", lambda: None)()
    summary = {
        "ok": bool(ok),
        "eval_success": bool(env.eval_success),
        "official_success": official,
        "strict_success": strict,
        "steps": int(env.take_action_cnt),
        "stage_reward": float(env.stage_reward()),
        "strict_diagnostics": diag,
    }
    print("[REPLAY_RESULT]", json.dumps(summary, ensure_ascii=False))
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
