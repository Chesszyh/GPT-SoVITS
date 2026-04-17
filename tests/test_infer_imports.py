from pathlib import Path
import unittest


class InferImportTests(unittest.TestCase):
    def test_tts_uses_package_local_process_ckpt(self):
        source = Path("GPT_SoVITS/TTS_infer_pack/TTS.py").read_text(encoding="utf-8")

        self.assertIn("from GPT_SoVITS.process_ckpt import", source)
        self.assertNotIn("from process_ckpt import", source)

    def test_tts_prefers_repository_package_root_for_legacy_imports(self):
        source = Path("GPT_SoVITS/TTS_infer_pack/TTS.py").read_text(encoding="utf-8")

        self.assertIn("__file__", source)
        self.assertIn("sys.path.insert(0, now_dir)", source)
        self.assertNotIn("sys.path.append(now_dir)", source)


if __name__ == "__main__":
    unittest.main()
