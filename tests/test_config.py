import tempfile
import unittest
from pathlib import Path

from gsv_cli.config import GsvConfig, load_config, write_default_config


class ConfigTests(unittest.TestCase):
    def test_default_config_uses_v2proplus(self):
        cfg = GsvConfig.default(project_name="voice")
        self.assertEqual(cfg.version, "v2ProPlus")
        self.assertEqual(cfg.language, "zh")
        self.assertEqual(cfg.train.sovits_batch_size, 2)
        self.assertEqual(cfg.train.gpt_batch_size, 1)

    def test_write_and_load_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsv.yaml"
            write_default_config(path, project_name="voice")
            loaded = load_config(path)
            self.assertEqual(loaded.project.name, "voice")
            self.assertEqual(loaded.paths.annotation, "data/train.list")

    def test_cli_override_updates_nested_values(self):
        cfg = GsvConfig.default(project_name="voice")
        updated = cfg.with_overrides({"train.sovits_batch_size": 4, "version": "v2"})
        self.assertEqual(updated.version, "v2")
        self.assertEqual(updated.train.sovits_batch_size, 4)
        self.assertEqual(cfg.train.sovits_batch_size, 2)


if __name__ == "__main__":
    unittest.main()
