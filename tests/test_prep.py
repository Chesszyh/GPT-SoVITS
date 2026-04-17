import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from gsv_cli import app
from gsv_cli.config import GsvConfig, write_config
from gsv_cli.prep import (
    build_asr_command,
    build_feature_commands,
    build_slice_command,
    expected_asr_output_path,
    merge_feature_outputs,
    run_feature_commands,
)
from gsv_cli.subprocesses import CommandResult


class PrepTests(unittest.TestCase):
    def test_slice_command_uses_existing_slicer_script(self):
        cmd = build_slice_command(
            python_exec="python",
            input_path="data/raw",
            output_dir="data/sliced",
            threshold=-34,
            min_length=4000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500,
            max_audio=0.9,
            alpha=0.25,
            part_index=0,
            part_count=1,
        )
        self.assertEqual(cmd[0:3], ["python", "-s", "tools/slice_audio.py"])
        self.assertIn("data/raw", cmd)
        self.assertIn("data/sliced", cmd)

    def test_asr_command_uses_faster_whisper(self):
        cmd = build_asr_command("python", "data/sliced", "output/asr", "large-v3", "zh", "float16")
        self.assertEqual(cmd[0:3], ["python", "-s", "tools/asr/fasterwhisper_asr.py"])
        self.assertIn("-l", cmd)
        self.assertIn("zh", cmd)

    def test_expected_asr_output_path_matches_script_output_name(self):
        self.assertEqual(expected_asr_output_path("data/sliced", "output/asr"), Path("output/asr/sliced.list"))

    def test_v2proplus_feature_commands_include_sv(self):
        cfg = GsvConfig.default("voice")
        commands = build_feature_commands(cfg, python_exec="python")
        joined = [" ".join(command) for command in commands]
        self.assertTrue(any("1-get-text.py" in command for command in joined))
        self.assertTrue(any("2-get-hubert-wav32k.py" in command for command in joined))
        self.assertTrue(any("2-get-sv.py" in command for command in joined))
        self.assertTrue(any("3-get-semantic.py" in command for command in joined))

    def test_v2_feature_commands_skip_sv(self):
        cfg = GsvConfig.default("voice").with_overrides({"version": "v2"})
        commands = build_feature_commands(cfg, python_exec="python")
        self.assertFalse(any("2-get-sv.py" in " ".join(command) for command in commands))

    def test_run_feature_commands_sets_training_environment(self):
        cfg = GsvConfig.default("voice")
        with patch("gsv_cli.prep.run_command", return_value=CommandResult(["python"], 0)) as run:
            run_feature_commands(cfg, dry_run=True, python_exec="python")
        first_env = run.call_args_list[0].kwargs["env"]
        self.assertEqual(first_env["inp_text"], "data/train.list")
        self.assertEqual(first_env["inp_wav_dir"], "data/sliced")
        self.assertEqual(first_env["exp_name"], "voice")
        self.assertEqual(first_env["version"], "v2ProPlus")

    def test_merge_feature_outputs_writes_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GsvConfig.default("voice").with_overrides({"paths.exp_root": tmp})
            exp_dir = Path(tmp) / "voice"
            exp_dir.mkdir(parents=True)
            (exp_dir / "2-name2text-0.txt").write_text("a\tphones\t[1]\tnorm\n", encoding="utf-8")
            (exp_dir / "6-name2semantic-0.tsv").write_text("a\t1 2 3\n", encoding="utf-8")
            merge_feature_outputs(cfg, part_count=1)
            self.assertEqual((exp_dir / "2-name2text.txt").read_text(encoding="utf-8"), "a\tphones\t[1]\tnorm\n")
            self.assertEqual(
                (exp_dir / "6-name2semantic.tsv").read_text(encoding="utf-8"),
                "item_name\tsemantic_audio\na\t1 2 3\n",
            )

    def test_cli_prep_features_loads_config(self):
        cfg = GsvConfig.default("voice")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gsv.yaml"
            write_config(config_path, cfg)
            with patch("gsv_cli.app.run_feature_commands", return_value=[]) as run:
                with redirect_stdout(StringIO()):
                    exit_code = app.main(["prep", "features", "-c", str(config_path), "--dry-run"])
        self.assertEqual(exit_code, 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0].project.name, "voice")
        self.assertTrue(run.call_args.kwargs["dry_run"])

    def test_cli_prep_slice_routes_to_runner(self):
        cfg = GsvConfig.default("voice")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gsv.yaml"
            write_config(config_path, cfg)
            with patch("gsv_cli.app.run_slice", return_value=CommandResult(["python"], 0)) as run:
                with redirect_stdout(StringIO()):
                    exit_code = app.main(
                        [
                            "prep",
                            "slice",
                            "-c",
                            str(config_path),
                            "--input",
                            "data/raw",
                            "--output",
                            "data/sliced",
                            "--dry-run",
                        ]
                    )
        self.assertEqual(exit_code, 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["input_path"], "data/raw")
        self.assertEqual(run.call_args.kwargs["output_dir"], "data/sliced")


if __name__ == "__main__":
    unittest.main()
