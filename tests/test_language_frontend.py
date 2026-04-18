import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path("GPT_SoVITS").resolve()))

from text.cleaner import clean_text  # noqa: E402


class LanguageFrontendTests(unittest.TestCase):
    def test_cleaner_rejects_unsupported_languages(self):
        for language in ["ko", "yue"]:
            with self.assertRaises(ValueError):
                clean_text("sample", language, version="v2")

    def test_tts_config_excludes_korean_and_cantonese(self):
        text = Path("GPT_SoVITS/TTS_infer_pack/TTS.py").read_text(encoding="utf-8")

        self.assertIn('SUPPORTED_TTS_LANGUAGES = ("zh", "en", "ja", "auto", "all_zh", "all_ja")', text)
        self.assertIn('v2_languages: list = ["auto", "en", "zh", "ja", "all_zh", "all_ja"]', text)
        self.assertNotRegex(text, r'"ko"')
        self.assertNotRegex(text, r'"yue"')

    def test_inference_core_does_not_import_runtime_i18n(self):
        paths = [
            Path("GPT_SoVITS/TTS_infer_pack/TTS.py"),
            Path("GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py"),
            Path("GPT_SoVITS/process_ckpt.py"),
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("tools.i18n", text, str(path))
            self.assertNotIn("I18nAuto", text, str(path))
            self.assertNotIn("scan_language_list", text, str(path))

    def test_asr_config_is_faster_whisper_only(self):
        from tools.asr.config import asr_dict

        self.assertEqual(set(asr_dict), {"Faster Whisper"})
        self.assertEqual(asr_dict["Faster Whisper"]["lang"], ["zh", "en", "ja"])

    def test_runtime_language_paths_have_no_removed_backend_tokens(self):
        paths = [
            Path("GPT_SoVITS/text/LangSegmenter/langsegmenter.py"),
            Path("tools/asr/fasterwhisper_asr.py"),
            Path("tools/asr/config.py"),
            Path("requirements.txt"),
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("funasr", text, str(path))
            self.assertNotIn("modelscope", text, str(path))
            self.assertNotRegex(text, r"\bko\b", str(path))
            self.assertNotRegex(text, r"\byue\b", str(path))


if __name__ == "__main__":
    unittest.main()
