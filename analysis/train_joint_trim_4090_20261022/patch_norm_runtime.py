"""Apply only the audited norm loading changes, preserving other runtime edits."""
import hashlib
import json
from pathlib import Path

path = Path("/home/sunny/bimanual-vla/server_4090/openpi_single_arm.py")
before = path.read_text()
start = before.index("def _create_torch_dataset_for_episodes(")
end = before.index("\ndef _install_training_episode_subset(", start)
chunk = before[start:end]
assert chunk.count("    action_offset: int,\n") == 1
chunk = chunk.replace("    action_offset: int,\n", "    action_offset: int,\n    load_images: bool = True,\n")
needle = "    dataset = data_loader.lerobot_dataset.LeRobotDataset(\n"
assert chunk.count(needle) == 1
chunk = chunk.replace(needle, "    dataset_class = data_loader.lerobot_dataset.LeRobotDataset\n"
    "    if not load_images:\n"
    "        from bimanual_vla.data.norm_video import NormOnlyLeRobotDataset\n"
    "        dataset_class = NormOnlyLeRobotDataset\n"
    "    dataset = dataset_class(\n")
after = before[:start]+chunk+before[end:]
start = after.index("def run_norm(")
end = after.index("\ndef _first_client_value(", start)
chunk = after[start:end]
needle = "        action_offset=int(config.policy_metadata[\"action_offset\"]),\n"
assert chunk.count(needle) == 1
chunk = chunk.replace(needle, needle+"        load_images=False,\n")
after = after[:start]+chunk+after[end:]
compile(after, str(path), "exec")
Path(__file__).with_name("runtime_before_norm_patch.py").write_text(before)
path.write_text(after)
Path(__file__).with_name("norm_patch.json").write_text(json.dumps(dict(
    before_sha256=hashlib.sha256(before.encode()).hexdigest(),
    after_sha256=hashlib.sha256(after.encode()).hexdigest()), indent=2))
print("PATCHED_NORM_ONLY")
