"""Fetch a dedicated pinned RLinf checkout and install Piper-only field adapters.

Run with a Conda Python. This does not install packages, download weights or
modify an existing RLinf checkout/environment.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess

from bimanual_vla.training.recap_settings import RLINF_COMMIT


def insert_function_branch(path, function, text):
    source = path.read_text()
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
    if len(nodes) != 1:
        raise ValueError(f"upstream API changed: {path}:{function}")
    first = nodes[0].body[0]
    lines = source.splitlines(keepends=True)
    indent = " " * first.col_offset
    lines[first.lineno - 1:first.lineno - 1] = [indent + line + "\n" for line in text.splitlines()]
    path.write_text("".join(lines))


def install(root):
    root = Path(root).expanduser().resolve()
    if root.exists():
        raise FileExistsError("use a new directory; existing RLinf trees are never rewritten")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/RLinf/RLinf.git", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", RLINF_COMMIT], check=True)
    adapt(root)
    print(root)


def adapt(root):
    """Apply small, checked adapters to the dedicated pinned checkout."""
    insert_function_branch(root / "rlinf/data/datasets/recap/value_dataset.py", "_build_transform",
        'if robot_type == "piper_bimanual":\n'
        '    from bimanual_vla.training.recap_adapter import PiperValueInputs\n'
        '    return PiperValueInputs()')
    insert_function_branch(root / "rlinf/models/embodiment/value_model/recap/checkpoint_utils.py", "build_input_transforms",
        'if env_type == "piper_bimanual":\n'
        '    from bimanual_vla.training.recap_adapter import PiperValueInputs\n'
        '    return [PiperValueInputs()]')
    path = root / "examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py"
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "KEY_MAPPINGS" for t in n.targets))
    lines = text.splitlines(keepends=True)
    lines[node.end_lineno:node.end_lineno] = ['\nKEY_MAPPINGS["piper_bimanual"] = {\n',
        '    **{f"observation.images.{key}": f"observation.images.{key}" for key in\n',
        '       ("cam_high", "cam_left_wrist", "cam_right_wrist")},\n',
        '    "observation.state": "observation.state", "task": "prompt",\n',
        '}\n']
    path.write_text("".join(lines))
    path = root / "rlinf/runners/sft_runner.py"
    source = path.read_text()
    anchor = "        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()"
    if source.count(anchor) != 1:
        raise ValueError("upstream checkpoint completion API changed")
    source = source.replace(anchor, anchor + '\n'
        '        # Piper adapter: mark only fully completed distributed saves.\n'
        '        from pathlib import Path\n'
        '        Path(base_output_dir, "bimanual_complete").touch()')
    path.write_text(source)
    (root / "bimanual_recap_adapter.json").write_text(json.dumps({"upstream_commit": RLINF_COMMIT, "adapter_version": 2}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    install(parser.parse_args().destination)
