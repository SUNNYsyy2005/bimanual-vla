"""Resume the existing joint trim run from its last complete Orbax state."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESPONSE = json.loads((ROOT / "train_final_response.json").read_text())
CHECKPOINT_DIR = Path(RESPONSE["metadata"]["checkpoint_dir"])
CHECKPOINT = CHECKPOINT_DIR / "3000"
EXIT_FILE = ROOT / "resume_after_oom_exit.json"
TASK_RUNNER = Path(__file__).resolve().parents[2] / "server_4090/task_runner.py"
TRAIN_CWD = "/home/sunny/robotwin_ws/RoboTwin/policy/pi05"


def main() -> None:
    for marker in ("_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"):
        if not (CHECKPOINT / marker).is_file():
            raise RuntimeError(f"incomplete recovery checkpoint: {CHECKPOINT / marker}")
    if EXIT_FILE.exists():
        raise RuntimeError(f"refusing to overwrite existing exit record: {EXIT_FILE}")
    command = list(RESPONSE["command"])
    if "--resume" in command or "--overwrite" in command:
        raise RuntimeError("original command unexpectedly has a resume mode")
    interval = command.index("--save-interval") + 1
    if command[interval] != "1000":
        raise RuntimeError("original save interval has changed")
    command[interval] = "5000"
    command.append("--resume")
    os.execv(
        command[0],
        [command[0], str(TASK_RUNNER), "--exit-json", str(EXIT_FILE),
         "--cwd", TRAIN_CWD, "--", *command],
    )


if __name__ == "__main__":
    main()
