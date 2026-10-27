"""Regression coverage for remote metadata inventory reads."""
import ast
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "server_4090" / "app.py"

def load_reader():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "remote_dataset_inventory")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(Any=Any, Path=Path, json=json, shlex=shlex, subprocess=subprocess,
                 SSH_COMMAND=["ssh", "-o", "BatchMode=yes"],
                 REMOTE_DATASET_INVENTORY_SCRIPT="print('[]')")
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["remote_dataset_inventory"]

class InventoryReadsTest(unittest.TestCase):
    def setUp(self):
        self.read = load_reader()

    def cache_read(self, payload):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inventory.json"
            path.write_text(json.dumps(payload))
            return self.read(dict(kind="slurm_only", inventory_cache_host="local_4090",
                                  inventory_cache_path=str(path)))

    def test_valid_empty_cache_is_success(self):
        self.assertEqual(self.cache_read(dict(datasets=[], roots=[
            dict(root="/DATA/node/lerobot", exists=False, datasets=0)])), ([], None))

    def test_nonempty_cache_is_preserved(self):
        rows = [dict(id="bottles", total_episodes=25)]
        self.assertEqual(self.cache_read(dict(datasets=rows, roots=[])), (rows, None))

    def test_malformed_dataset_list_is_not_an_empty_success(self):
        rows, error = self.cache_read(dict(datasets="invalid"))
        self.assertEqual(rows, [])
        self.assertIn("invalid dataset list", error)

    def test_remote_read_uses_configured_conda_and_shell_quotes_it(self):
        result = subprocess.CompletedProcess([], 0, "[]", "")
        with mock.patch.object(subprocess, "run", return_value=result) as run:
            self.assertEqual(self.read(dict(kind="ssh", host="login-server",
                dataset_root="/DATA/data", inventory_python="/DATA/conda env/bin/python")), ([], None))
        command = run.call_args.args[0][-1]
        self.assertIn("'/DATA/conda env/bin/python' -", command)
        self.assertNotIn(" python3 ", command)

    def test_missing_interpreter_does_not_fall_back_to_system_python(self):
        with mock.patch.object(subprocess, "run") as run:
            rows, error = self.read(dict(kind="ssh", host="login-server", dataset_root="/DATA/data"))
        run.assert_not_called()
        self.assertIn("missing configured", error)

    def test_ssh_failure_remains_an_error(self):
        result = subprocess.CompletedProcess([], 255, "", "kex_exchange_identification: Connection reset")
        with mock.patch.object(subprocess, "run", return_value=result):
            rows, error = self.read(dict(kind="ssh", host="login-server",
                dataset_root="/DATA/data", openpi_python="/DATA/conda/bin/python"))
        self.assertEqual(rows, [])
        self.assertIn("Connection reset", error)

if __name__ == "__main__":
    unittest.main()

