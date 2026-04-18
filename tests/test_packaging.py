import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def test_pyproject_limits_editable_package_discovery(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.setuptools]", text)
        self.assertIn('packages = ["gsv_cli"]', text)

    def test_install_script_uses_hf_sources_only(self):
        text = Path("install.sh").read_text(encoding="utf-8")
        self.assertIn("HF|HF-Mirror", text)
        self.assertNotIn("ModelScope", text)
        self.assertNotIn("modelscope.cn", text)

    def test_repository_has_no_docker_configuration(self):
        forbidden_paths = [
            ".dockerignore",
            "Docker",
            "docker_build.sh",
            ".github/workflows/docker-publish.yaml",
        ]

        for path in forbidden_paths:
            self.assertFalse(Path(path).exists(), path)

    def test_repository_has_no_windows_packaging_configuration(self):
        forbidden_paths = [
            ".github/build_windows_packages.ps1",
            ".github/workflows/build_windows_packages.yaml",
        ]

        for path in forbidden_paths:
            self.assertFalse(Path(path).exists(), path)

    def test_extra_requirements_file_uses_standard_name(self):
        self.assertTrue(Path("requirements-extra.txt").is_file())
        self.assertFalse(Path("extra-req.txt").exists())

        install_text = Path("install.sh").read_text(encoding="utf-8")
        readme_text = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("requirements-extra.txt", install_text)
        self.assertIn("requirements-extra.txt", readme_text)
        self.assertNotIn("extra-req.txt", install_text)
        self.assertNotIn("extra-req.txt", readme_text)

    def test_docs_use_standard_language_directory_names(self):
        self.assertTrue(Path("docs/zh/README.md").is_file())
        self.assertTrue(Path("docs/ja/README.md").is_file())
        self.assertFalse(Path("docs/cn").exists())

    def test_runtime_code_has_no_legacy_docker_or_windows_branches(self):
        runtime_files = [
            Path("GPT_SoVITS/AR/data/dataset.py"),
            Path("GPT_SoVITS/s1_train.py"),
            Path("GPT_SoVITS/prepare_datasets/1-get-text.py"),
            Path("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"),
            Path("GPT_SoVITS/prepare_datasets/3-get-semantic.py"),
            Path("GPT_SoVITS/feature_extractor/cnhubert.py"),
        ]

        combined = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
        self.assertNotIn("/data/docker", combined)
        self.assertNotIn('platform.system() != "Windows"', combined)


if __name__ == "__main__":
    unittest.main()
