import unittest
from unittest.mock import patch

from gsv_cli import app
from gsv_cli.subprocesses import CommandResult
from gsv_cli.separate import AudioSeparatorMissing, build_audio_separator_command, ensure_audio_separator


class SeparateTests(unittest.TestCase):
    def test_missing_audio_separator_raises(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaises(AudioSeparatorMissing):
                ensure_audio_separator("audio-separator")

    def test_build_file_command(self):
        cmd = build_audio_separator_command(
            command="audio-separator",
            input_path="song.wav",
            output_dir="out",
            stem="vocals",
            model="model.onnx",
            output_format="WAV",
        )
        self.assertEqual(
            cmd,
            [
                "audio-separator",
                "song.wav",
                "--output_dir",
                "out",
                "--output_single_stem",
                "vocals",
                "--model_filename",
                "model.onnx",
                "--output_format",
                "WAV",
            ],
        )

    def test_cli_separate_routes_command_option(self):
        with patch("gsv_cli.app.run_separation", return_value=CommandResult(["audio-separator"], 0)) as run:
            exit_code = app.main(
                [
                    "separate",
                    "song.wav",
                    "--output-dir",
                    "out",
                    "--command",
                    "audio-separator",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with(
            command="audio-separator",
            input_path="song.wav",
            output_dir="out",
            stem="vocals",
            model=None,
            output_format="WAV",
            dry_run=True,
        )


if __name__ == "__main__":
    unittest.main()
