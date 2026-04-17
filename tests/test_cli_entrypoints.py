import subprocess
import sys
import unittest

from gsv_cli import app


class CliEntrypointTests(unittest.TestCase):
    def test_main_version_exits_zero(self):
        exit_code = app.main(["--version"])
        self.assertEqual(exit_code, 0)

    def test_module_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "gsv_cli", "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("GPT-SoVITS CLI", result.stdout)
        self.assertIn("init", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("infer", result.stdout)


if __name__ == "__main__":
    unittest.main()
