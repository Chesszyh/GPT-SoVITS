import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from gsv_cli import app
from gsv_cli.config import GsvConfig, write_config
from gsv_cli.infer import build_tts_inputs, make_tts_config


class InferTests(unittest.TestCase):
    def test_make_tts_config_uses_configured_weights(self):
        cfg = GsvConfig.default("voice").with_overrides(
            {
                "infer.gpt_weight": "GPT_weights_v2ProPlus/voice-e15.ckpt",
                "infer.sovits_weight": "SoVITS_weights_v2ProPlus/voice-e8.pth",
            }
        )
        tts_cfg = make_tts_config(cfg, device="cpu", is_half=False)
        custom = tts_cfg["custom"]
        self.assertEqual(custom["version"], "v2ProPlus")
        self.assertEqual(custom["t2s_weights_path"], "GPT_weights_v2ProPlus/voice-e15.ckpt")
        self.assertEqual(custom["vits_weights_path"], "SoVITS_weights_v2ProPlus/voice-e8.pth")

    def test_build_tts_inputs_validates_languages(self):
        inputs = build_tts_inputs(
            text="hello",
            text_language="en",
            ref_audio="ref.wav",
            ref_text="prompt",
            ref_language="en",
        )
        self.assertEqual(inputs["text_lang"], "en")
        self.assertEqual(inputs["prompt_lang"], "en")
        with self.assertRaises(ValueError):
            build_tts_inputs("hello", "ko", "ref.wav", "prompt", "en")

    def test_cli_infer_dry_run_prints_config_and_skips_synthesis(self):
        cfg = GsvConfig.default("voice")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gsv.yaml"
            write_config(config_path, cfg)
            stdout = StringIO()
            with patch("gsv_cli.app.synthesize_to_file") as synthesize:
                with redirect_stdout(stdout):
                    exit_code = app.main(
                        [
                            "infer",
                            "-c",
                            str(config_path),
                            "--text",
                            "hello",
                            "--ref",
                            "ref.wav",
                            "--ref-text",
                            "prompt",
                            "--out",
                            str(Path(tmp) / "out.wav"),
                            "--device",
                            "cpu",
                            "--dry-run",
                        ]
                    )
        self.assertEqual(exit_code, 0)
        synthesize.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("t2s_weights_path", output)
        self.assertIn("hello", output)
        self.assertIn("ref.wav", output)


if __name__ == "__main__":
    unittest.main()
