"""Derive small, real-model pipeline configs from RLinf's e2e examples."""
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["TASK_ROOT"])
repo = Path(os.environ["REPO_PATH"])
out = root / "configs"
out.mkdir(exist_ok=True)
train = str(root / "datasets/SYNTHETIC_recap_train")
heldout = str(root / "datasets/SYNTHETIC_recap_heldout")
tag = "synthetic_20261101"
siglip = str(root / "models/siglip2-so400m-patch14-224")
gemma = str(root / "models/gemma-3-270m")

def save(name, cfg):
    (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

returns = yaml.safe_load((repo / "examples/offline_rl/config/recap_compute_returns.yaml").read_text())
returns["data"].update(tag=tag, num_workers=2, failure_reward=-20.0,
                      train_data_paths=[{"dataset_path": p, "type": "rollout"} for p in [train, heldout]])
save("returns", returns)

value = yaml.safe_load((repo / "tests/e2e_tests/offline_rl/recap_value_model_sft_libero.yaml").read_text())
value["defaults"].insert(-1, "_self_")
value["cluster"]["component_placement"] = {"actor,env,rollout": "0"}
value["runner"].update(max_steps=10, save_interval=10, val_check_interval=5)
value["runner"]["logger"].update(log_path=str(root / "runs/value"), experiment_name="synthetic_value")
value["data"].update(tag=tag, action_dim=32, train_num_workers=0, eval_num_workers=0,
                     do_augment=False, train_data_paths=[{"dataset_path": train, "type": "rollout", "weight": 1.0}],
                     eval_data_paths=[{"dataset_path": heldout, "max_samples": 26}])
value["actor"].update(micro_batch_size=1, global_batch_size=2)
value["actor"]["model"].update(siglip_path=siglip, gemma3_path=gemma, tokenizer_path=gemma, action_dim=32)
value["actor"]["optim"].update(lr_warmup_steps=0, total_training_steps=12)
if os.environ.get("RECAP_LOW_MEMORY") == "1":
    value["actor"]["model"].update(freeze_vision_encoder=True, freeze_vlm=True)
    value["actor"]["fsdp_config"]["use_orig_params"] = True
save("value", value)
resume = copy.deepcopy(value)
resume["runner"].update(max_steps=12, save_interval=12, val_check_interval=12,
                        resume_dir=str(root / "runs/value/synthetic_value/checkpoints/global_step_10"))
save("value_resume", resume)

advantages = yaml.safe_load((repo / "examples/offline_rl/config/recap_compute_advantages.yaml").read_text())
advantages["advantage"].update(value_checkpoint=str(root / "runs/value/synthetic_value/checkpoints/global_step_12/actor"),
                                batch_size=2, num_dataloader_workers_per_gpu=0, tag=tag, returns_tag=tag)
advantages["advantage"]["model"].update(siglip_path=siglip, gemma3_path=gemma, tokenizer_path=gemma,
                                        action_dim=32, action_horizon=10)
advantages["data"].update(advantage_lookahead_step=3, train_data_paths=[
    {"dataset_path": p, "robot_type": "libero", "type": "rollout", "weight": 1.0} for p in [train, heldout]])
advantages["distributed"]["enabled"] = False
save("advantages", advantages)

cfg = yaml.safe_load((repo / "tests/e2e_tests/offline_rl/recap_cfg_openpi_libero.yaml").read_text())
cfg["defaults"].insert(-1, "_self_")
cfg["cluster"]["component_placement"] = {"actor,env,rollout": "0"}
cfg["runner"].update(max_steps=4, save_interval=4)
cfg["runner"]["logger"].update(log_path=str(root / "runs/cfg"), experiment_name="synthetic_cfg")
cfg["data"].update(num_workers=0, advantage_tag=tag, train_data_paths=[
    {"dataset_path": train, "type": "rollout", "weight": 1.0}])
cfg["actor"].update(micro_batch_size=1, global_batch_size=2)
cfg["actor"]["model"]["model_path"] = str(root / "models/pi05_libero")
cfg["actor"]["optim"].update(total_training_steps=6)
if os.environ.get("RECAP_LOW_MEMORY") == "1":
    cfg["actor"]["model"]["openpi"]["train_expert_only"] = True
    cfg["actor"]["fsdp_config"]["use_orig_params"] = True
save("cfg", cfg)
resume = copy.deepcopy(cfg)
resume["runner"].update(max_steps=6, save_interval=6,
                        resume_dir=str(root / "runs/cfg/synthetic_cfg/checkpoints/global_step_4"))
save("cfg_resume", resume)
