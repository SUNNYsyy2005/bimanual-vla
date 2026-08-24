from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTest(unittest.TestCase):
    def test_root_contains_no_python_source_files(self):
        self.assertEqual(list(REPO_ROOT.glob("*.py")), [])

    def test_stable_command_entrypoint_and_packages_exist(self):
        entrypoint = REPO_ROOT / "bin/bimanual-vla"
        self.assertTrue(entrypoint.is_file())
        self.assertTrue(entrypoint.stat().st_mode & 0o111)
        for package in ("collection", "data", "deployment"):
            self.assertTrue((REPO_ROOT / "bimanual_vla" / package / "__init__.py").is_file())

    def test_launch_and_deploy_scripts_use_package_paths(self):
        gui_launcher = (REPO_ROOT / "start_gui.sh").read_text(encoding="utf-8")
        self.assertIn("bin/bimanual-vla", gui_launcher)
        self.assertNotIn("collect_gui.py", gui_launcher)

        for name in ("deploy_4090_server.sh", "deploy_4090_sim_dashboard.sh"):
            source = (REPO_ROOT / name).read_text(encoding="utf-8")
            self.assertIn('"$LOCAL_ROOT/./bimanual_vla"', source)
            self.assertNotIn('"$LOCAL_ROOT/./rtc_client.py"', source)
            self.assertNotIn('"$LOCAL_ROOT/./piper_data_contract.py"', source)


if __name__ == "__main__":
    unittest.main()
