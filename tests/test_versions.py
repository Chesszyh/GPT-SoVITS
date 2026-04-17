import unittest

from gsv_cli.versions import SUPPORTED_LANGUAGES, SUPPORTED_VERSIONS, get_version, validate_language


class VersionRegistryTests(unittest.TestCase):
    def test_supported_versions_exclude_v1(self):
        self.assertEqual(set(SUPPORTED_VERSIONS), {"v2", "v2Pro", "v2ProPlus"})

    def test_v2proplus_paths(self):
        spec = get_version("v2ProPlus")
        self.assertEqual(spec.sovits_config, "GPT_SoVITS/configs/s2v2ProPlus.json")
        self.assertEqual(spec.gpt_config, "GPT_SoVITS/configs/s1longer-v2.yaml")
        self.assertEqual(spec.sovits_weight_dir, "SoVITS_weights_v2ProPlus")
        self.assertEqual(spec.gpt_weight_dir, "GPT_weights_v2ProPlus")

    def test_unsupported_version_raises(self):
        with self.assertRaises(ValueError):
            get_version("v1")

    def test_languages_are_zh_en_ja_only(self):
        self.assertEqual(SUPPORTED_LANGUAGES, ("zh", "en", "ja"))
        validate_language("zh")
        validate_language("en")
        validate_language("ja")
        with self.assertRaises(ValueError):
            validate_language("ko")
        with self.assertRaises(ValueError):
            validate_language("yue")


if __name__ == "__main__":
    unittest.main()
