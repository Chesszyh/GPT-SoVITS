import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from gsv_cli import app
from gsv_cli.config import GsvConfig, write_config
from gsv_cli.subprocesses import CommandResult
from gsv_cli.train import (
    build_gpt_train_command,
    build_sovits_train_command,
    run_gpt_training,
    write_gpt_config,
    write_sovits_config,
)


class TrainTests(unittest.TestCase):
    def test_sovits_command_targets_s2_train(self):
        cmd = build_sovits_train_command("python", "tmp_s2.json")
        self.assertEqual(cmd, ["python", "-s", "GPT_SoVITS/s2_train.py", "--config", "tmp_s2.json"])

    def test_gpt_command_targets_s1_train(self):
        cmd = build_gpt_train_command("python", "tmp_s1.yaml")
        self.assertEqual(cmd, ["python", "-s", "GPT_SoVITS/s1_train.py", "--config_file", "tmp_s1.yaml"])

    def test_generated_configs_include_training_paths_and_weights(self):
        cfg = GsvConfig.default("voice")
        with tempfile.TemporaryDirectory() as tmp:
            s2_path = write_sovits_config(cfg, Path(tmp))
            s1_path = write_gpt_config(cfg, Path(tmp))
            self.assertTrue(s2_path.exists())
            self.assertTrue(s1_path.exists())
            self.assertIn("tmp_s2", s2_path.name)
            self.assertIn("tmp_s1", s1_path.name)
            s2_data = json.loads(s2_path.read_text(encoding="utf-8"))
            s1_data = yaml.safe_load(s1_path.read_text(encoding="utf-8"))
        self.assertEqual(s2_data["train"]["batch_size"], 2)
        self.assertEqual(s2_data["model"]["version"], "v2ProPlus")
        self.assertEqual(s2_data["save_weight_dir"], "SoVITS_weights_v2ProPlus")
        self.assertEqual(s1_data["train"]["batch_size"], 1)
        self.assertEqual(s1_data["train"]["half_weights_save_dir"], "GPT_weights_v2ProPlus")
        self.assertEqual(s1_data["train_semantic_path"], "logs/voice/6-name2semantic.tsv")
        self.assertEqual(s1_data["train_phoneme_path"], "logs/voice/2-name2text.txt")

    def test_cli_train_sovits_routes_batch_override(self):
        cfg = GsvConfig.default("voice")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gsv.yaml"
            write_config(config_path, cfg)
            with patch("gsv_cli.app.run_sovits_training", return_value=CommandResult(["python"], 0)) as run:
                with redirect_stdout(StringIO()):
                    exit_code = app.main(
                        ["train", "sovits", "-c", str(config_path), "--batch-size", "4", "--dry-run"]
                    )
        self.assertEqual(exit_code, 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0].train.sovits_batch_size, 4)
        self.assertTrue(run.call_args.kwargs["dry_run"])

    def test_gpt_training_sets_cuda_environment(self):
        cfg = GsvConfig.default("voice")
        with patch("gsv_cli.train.write_gpt_config", return_value=Path("tmp_s1.yaml")):
            with patch("gsv_cli.train.run_command", return_value=CommandResult(["python"], 0)) as run:
                run_gpt_training(cfg, dry_run=True, python_exec="python")
        self.assertEqual(run.call_args.kwargs["env"]["_CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["hz"], "25hz")


if __name__ == "__main__":
    unittest.main()
