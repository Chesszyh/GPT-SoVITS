import tempfile
import unittest
from pathlib import Path

from gsv_cli.wiki import publish_github_wiki_source


class WikiPublisherTests(unittest.TestCase):
    def test_publisher_flattens_language_dirs_and_rewrites_sidebar_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            (source / "en").mkdir(parents=True)
            (source / "zh").mkdir(parents=True)
            (source / "Home.md").write_text(
                "- [Overview](en/1-overview)\n- [概览](zh/1-overview)\n",
                encoding="utf-8",
            )
            (source / "_Sidebar.md").write_text(
                "- [Overview](en/1-overview)\n- [概览](zh/1-overview)\n",
                encoding="utf-8",
            )
            (source / "en" / "1-overview.md").write_text(
                "# Overview\n\nSee [Architecture](/RVC-Boss/GPT-SoVITS/2-system-architecture).\n",
                encoding="utf-8",
            )
            (source / "zh" / "1-overview.md").write_text(
                "# Overview (概览)\n\n参见 [系统架构](/RVC-Boss/GPT-SoVITS/2-system-architecture)。\n",
                encoding="utf-8",
            )
            (source / "en" / "2-system-architecture.md").write_text(
                "# System Architecture\n",
                encoding="utf-8",
            )
            (source / "zh" / "2-system-architecture.md").write_text(
                "# System Architecture (系统架构)\n",
                encoding="utf-8",
            )

            publish_github_wiki_source(source, target)

            self.assertTrue((target / "en-1-overview.md").exists())
            self.assertTrue((target / "zh-1-overview.md").exists())
            self.assertFalse((target / "en" / "1-overview.md").exists())
            sidebar = (target / "_Sidebar.md").read_text(encoding="utf-8")
            self.assertIn("(en-1-overview)", sidebar)
            self.assertIn("(zh-1-overview)", sidebar)
            self.assertNotIn("(en/", sidebar)
            self.assertNotIn("(zh/", sidebar)
            en_page = (target / "en-1-overview.md").read_text(encoding="utf-8")
            zh_page = (target / "zh-1-overview.md").read_text(encoding="utf-8")
            self.assertIn("](en-2-system-architecture)", en_page)
            self.assertIn("](zh-2-system-architecture)", zh_page)


if __name__ == "__main__":
    unittest.main()
