import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from gsv_cli import app
from gsv_cli.config import load_config


class CliEntrypointTests(unittest.TestCase):
    def test_main_version_exits_zero(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = app.main(["--version"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "0.1.0")

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

    def test_init_writes_selected_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gsv.yaml"
            with redirect_stdout(StringIO()):
                exit_code = app.main(["init", "voice", "--config", str(config_path), "--version", "v2"])
            self.assertEqual(exit_code, 0)
            self.assertEqual(load_config(config_path).version, "v2")


if __name__ == "__main__":
    unittest.main()
