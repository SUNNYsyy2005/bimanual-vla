from pathlib import Path
import unittest


class DashboardTemplateNullGuardTest(unittest.TestCase):
    def test_timed_target_helpers_guard_null_object_values(self):
        template = (
            Path(__file__).parent / "server_4090/templates/index.html"
        ).read_text(encoding="utf-8")
        guarded = (
            "t && t.client_timed_target && "
            "typeof t.client_timed_target === 'object'"
        )
        # Both helpers dereference target_time_error_s/target_at.  In JS,
        # typeof null is also "object", so the truthiness check is required.
        self.assertGreaterEqual(template.count(guarded), 2)
        self.assertNotIn(
            "const target = t && typeof t.client_timed_target === 'object' ? t.client_timed_target : {};",
            template,
        )


if __name__ == "__main__":
    unittest.main()
