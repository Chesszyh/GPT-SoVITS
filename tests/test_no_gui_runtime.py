import unittest
from pathlib import Path


class NoGuiRuntimeTests(unittest.TestCase):
    def test_obsolete_gui_entrypoints_are_removed(self):
        obsolete_paths = [
            "go-webui.bat",
            "go-webui.ps1",
            "install.ps1",
            "Colab-WebUI.ipynb",
            "Colab-Inference.ipynb",
            "webui.py",
            "GPT_SoVITS/inference_webui.py",
            "GPT_SoVITS/inference_webui_fast.py",
            "GPT_SoVITS/inference_gui.py",
            "api.py",
            "api_v2.py",
            "tools/subfix_webui.py",
            "tools/uvr5",
        ]
        for path in obsolete_paths:
            self.assertFalse(Path(path).exists(), path)

    def test_runtime_python_no_longer_imports_gui_modules(self):
        for root in [Path("gsv_cli"), Path("GPT_SoVITS"), Path("tools")]:
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("import gradio", text, str(path))
                self.assertNotIn("from inference_webui", text, str(path))
                self.assertNotIn("GPT_SoVITS.inference_webui", text, str(path))


if __name__ == "__main__":
    unittest.main()
