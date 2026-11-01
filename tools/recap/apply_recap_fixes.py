"""Apply a narrow checked RECAP data fix to the isolated source snapshot."""
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["TASK_ROOT"])
repo = Path(os.environ["REPO_PATH"])
target = repo / "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py"
source = target.read_text()
before = hashlib.sha256(source.encode()).hexdigest()
old = "is_success = bool(is_success_col[ep_end - 1])"
new = "is_success = coerce_episode_success(is_success_col[ep_end - 1])"
helper = '''def coerce_episode_success(value) -> bool:
    """Read scalar or one-element LeRobot boolean fields without list truthiness."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("is_success must contain exactly one boolean per frame")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, float, np.integer, np.floating)) and scalar in (0, 1):
        return bool(scalar)
    raise ValueError(f"Invalid is_success value: {value!r}")


'''
if old in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace("def compute_returns_for_episode(", helper + "def compute_returns_for_episode(", 1)
    target.write_text(source)
elif new not in source or "def coerce_episode_success" not in source:
    raise RuntimeError("Unknown compute_returns.py revision; refusing an unchecked patch")
record = {"path": str(target), "fix": "scalar_or_single_element_success_flag",
          "before_sha256": before, "after_sha256": hashlib.sha256(source.encode()).hexdigest()}
for name in ["fsdp_value_sft_worker.py", "fsdp_cfg_worker.py"]:
    worker_path = repo / "rlinf/workers/sft" / name
    worker_source = worker_path.read_text()
    marker = "# Optional cap for standalone RECAP validation on a shared workstation."
    if marker not in worker_source:
        needle = "        self.device = Worker.torch_platform.current_device()"
        assert worker_source.count(needle) == 1, worker_path
        replacement = needle + '''
        # Optional cap for standalone RECAP validation on a shared workstation.
        memory_fraction = os.environ.get("RECAP_GPU_MEMORY_FRACTION")
        if memory_fraction:
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
'''
        worker_path.write_text(worker_source.replace(needle, replacement, 1))
(root / "records/source-fixes.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
