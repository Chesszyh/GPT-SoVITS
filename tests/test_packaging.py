import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def test_pyproject_limits_editable_package_discovery(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.setuptools]", text)
        self.assertIn('packages = ["gsv_cli"]', text)


if __name__ == "__main__":
    unittest.main()
