# CLI-First Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-first GPT-SoVITS training and inference loop for Linux training and macOS inference, then remove obsolete GUI/Windows/i18n-heavy paths after replacement commands are verified.

**Architecture:** Add a new `gsv_cli` package with command parsing, typed config loading, version/language registry, subprocess orchestration, training config generation, and direct inference through `GPT_SoVITS.TTS_infer_pack.TTS`. Use existing low-level dataset and training scripts during the first pass, then delete GUI and unsupported platform/language branches in separate cleanup tasks.

**Tech Stack:** Python 3.10, stdlib `argparse`/`dataclasses`/`unittest`, PyYAML, existing GPT-SoVITS training scripts, Faster-Whisper, optional external `audio-separator`.

---

## File Structure

Create:

- `pyproject.toml`: editable install metadata and console scripts for `gsv` and `gpt-sovits`.
- `gsv_cli/__init__.py`: package version.
- `gsv_cli/__main__.py`: `python -m gsv_cli` entrypoint.
- `gsv_cli/app.py`: top-level CLI parser and subcommand dispatch.
- `gsv_cli/config.py`: `gsv.yaml` dataclasses, defaults, load/write, CLI overrides.
- `gsv_cli/versions.py`: supported versions and languages.
- `gsv_cli/paths.py`: repository/project path helpers.
- `gsv_cli/subprocesses.py`: testable subprocess wrapper.
- `gsv_cli/separate.py`: optional `audio-separator` wrapper.
- `gsv_cli/prep.py`: slice, ASR, text/Hubert/SV/semantic feature orchestration.
- `gsv_cli/train.py`: SoVITS/GPT config generation and training command orchestration.
- `gsv_cli/infer.py`: direct TTS inference command implementation.
- `tests/__init__.py`: unittest package marker.
- `tests/test_cli_entrypoints.py`: CLI smoke tests.
- `tests/test_config.py`: config behavior tests.
- `tests/test_versions.py`: registry/language tests.
- `tests/test_separate.py`: external separation wrapper tests.
- `tests/test_prep.py`: prep command generation tests.
- `tests/test_train.py`: training config generation tests.
- `tests/test_infer.py`: inference input validation tests.

Modify:

- `install.sh`: keep Linux/macOS installation working and install this repo editable.
- `requirements.txt`: remove GUI/unsupported ASR dependencies after CLI verification.
- `README.md`: replace broad beginner/WebUI instructions with concise CLI workflow.
- `docs/cn/README.md` and `docs/ja/README.md`: either update to CLI workflow or remove if maintaining only root README.
- Existing GPT-SoVITS scripts only where they must accept CLI-friendly arguments instead of WebUI environment coupling.

Delete after CLI verification:

- Windows launcher/installer files.
- Gradio/WebUI entrypoints.
- UVR5 WebUI/tooling.
- UI i18n locale files and runtime wrappers.
- unsupported docs and notebooks.
- unsupported Korean/Cantonese frontend files from the supported path.

---

### Task 1: CLI Package Skeleton And Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `gsv_cli/__init__.py`
- Create: `gsv_cli/__main__.py`
- Create: `gsv_cli/app.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Write entrypoint tests**

Create `tests/__init__.py` as an empty file.

Create `tests/test_cli_entrypoints.py`:

```python
import subprocess
import sys
import unittest

from gsv_cli import app


class CliEntrypointTests(unittest.TestCase):
    def test_main_version_exits_zero(self):
        exit_code = app.main(["--version"])
        self.assertEqual(exit_code, 0)

    def test_module_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "gsv_cli", "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("GPT-SoVITS CLI", result.stdout)
        self.assertIn("init", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("infer", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_cli_entrypoints -v
```

Expected: FAIL or ERROR because `gsv_cli` does not exist.

- [ ] **Step 3: Add minimal package and parser**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gpt-sovits-cli"
version = "0.1.0"
description = "CLI-first workflow for this GPT-SoVITS fork"
requires-python = ">=3.10,<3.13"

[project.scripts]
gsv = "gsv_cli.app:console_main"
gpt-sovits = "gsv_cli.app:console_main"
```

Create `gsv_cli/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `gsv_cli/__main__.py`:

```python
from .app import console_main


if __name__ == "__main__":
    console_main()
```

Create `gsv_cli/app.py`:

```python
from __future__ import annotations

import argparse
from typing import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsv", description="GPT-SoVITS CLI")
    parser.add_argument("--version", action="store_true", help="Print CLI version and exit")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init", help="Create a gsv.yaml project config")
    subparsers.add_parser("separate", help="Run optional audio-separator wrapper")

    prep = subparsers.add_parser("prep", help="Prepare dataset artifacts")
    prep_sub = prep.add_subparsers(dest="prep_command")
    prep_sub.add_parser("slice")
    prep_sub.add_parser("asr")
    prep_sub.add_parser("features")
    prep_sub.add_parser("all")

    train = subparsers.add_parser("train", help="Train GPT-SoVITS weights")
    train_sub = train.add_subparsers(dest="train_command")
    train_sub.add_parser("sovits")
    train_sub.add_parser("gpt")
    train_sub.add_parser("all")

    subparsers.add_parser("infer", help="Synthesize audio")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    print(f"Command '{args.command}' is not implemented yet")
    return 2


def console_main() -> None:
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_cli_entrypoints -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml gsv_cli tests
git commit -m "feat: add cli entrypoint skeleton"
```

---

### Task 2: Config File Model And CLI Overrides

**Files:**
- Create: `gsv_cli/config.py`
- Modify: `gsv_cli/app.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_config -v
```

Expected: ERROR because `gsv_cli.config` does not exist.

- [ ] **Step 3: Implement config dataclasses**

Create `gsv_cli/config.py` with dataclasses for `ProjectConfig`, `PathConfig`, `AsrConfig`, `SeparationConfig`, `TrainConfig`, `InferConfig`, and `GsvConfig`. Implement:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    root: str = "."


@dataclass(frozen=True)
class PathConfig:
    raw_audio: str = "data/raw"
    separated_audio: str = "data/separated"
    sliced_audio: str = "data/sliced"
    annotation: str = "data/train.list"
    exp_root: str = "logs"
    pretrained_root: str = "GPT_SoVITS/pretrained_models"


@dataclass(frozen=True)
class AsrConfig:
    engine: str = "faster-whisper"
    model: str = "large-v3"
    precision: str = "float16"


@dataclass(frozen=True)
class SeparationConfig:
    enabled: bool = False
    command: str = "audio-separator"
    stem: str = "vocals"
    output_format: str = "WAV"


@dataclass(frozen=True)
class TrainConfig:
    gpu: str = "0"
    sovits_batch_size: int = 2
    gpt_batch_size: int = 1
    sovits_epochs: int = 8
    gpt_epochs: int = 15
    save_every_epoch: int = 4
    save_latest: bool = True
    save_every_weights: bool = True
    grad_ckpt: bool = True


@dataclass(frozen=True)
class InferConfig:
    gpt_weight: str = ""
    sovits_weight: str = ""
    ref_audio: str = ""
    ref_text: str = ""
    ref_language: str = "zh"
    text_language: str = "zh"


@dataclass(frozen=True)
class GsvConfig:
    project: ProjectConfig
    version: str = "v2ProPlus"
    language: str = "zh"
    speaker: str = "voice"
    paths: PathConfig = field(default_factory=PathConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    @classmethod
    def default(cls, project_name: str) -> "GsvConfig":
        return cls(project=ProjectConfig(name=project_name), speaker=project_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any]) -> "GsvConfig":
        cfg: GsvConfig = self
        for dotted_key, value in overrides.items():
            if "." not in dotted_key:
                cfg = replace(cfg, **{dotted_key: value})
                continue
            section, key = dotted_key.split(".", 1)
            section_obj = getattr(cfg, section)
            cfg = replace(cfg, **{section: replace(section_obj, **{key: value})})
        return cfg


def _from_dict(data: dict[str, Any]) -> GsvConfig:
    return GsvConfig(
        project=ProjectConfig(**data["project"]),
        version=data.get("version", "v2ProPlus"),
        language=data.get("language", "zh"),
        speaker=data.get("speaker", data["project"]["name"]),
        paths=PathConfig(**data.get("paths", {})),
        asr=AsrConfig(**data.get("asr", {})),
        separation=SeparationConfig(**data.get("separation", {})),
        train=TrainConfig(**data.get("train", {})),
        infer=InferConfig(**data.get("infer", {})),
    )


def load_config(path: str | Path) -> GsvConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _from_dict(data)


def write_default_config(path: str | Path, project_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = GsvConfig.default(project_name)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)
```

- [ ] **Step 4: Add `gsv init` behavior**

Modify `gsv_cli/app.py` so `init` accepts `name`, `--config`, and `--version`; call `write_default_config`, apply `version` override when provided, and write the result to `<name>/gsv.yaml` unless `--config` is provided.

Run:

```bash
python -m unittest tests.test_config tests.test_cli_entrypoints -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gsv_cli/app.py gsv_cli/config.py tests/test_config.py
git commit -m "feat: add cli config model"
```

---

### Task 3: Version And Language Registry

**Files:**
- Create: `gsv_cli/versions.py`
- Create: `tests/test_versions.py`
- Modify: `gsv_cli/config.py`

- [ ] **Step 1: Write registry tests**

Create `tests/test_versions.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_versions -v
```

Expected: ERROR because `gsv_cli.versions` does not exist.

- [ ] **Step 3: Implement registry**

Create `gsv_cli/versions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_LANGUAGES = ("zh", "en", "ja")


@dataclass(frozen=True)
class VersionSpec:
    name: str
    sovits_config: str
    gpt_config: str
    pretrained_sovits_g: str
    pretrained_sovits_d: str
    pretrained_gpt: str
    sovits_weight_dir: str
    gpt_weight_dir: str
    needs_sv_features: bool


SUPPORTED_VERSIONS: dict[str, VersionSpec] = {
    "v2": VersionSpec(
        name="v2",
        sovits_config="GPT_SoVITS/configs/s2.json",
        gpt_config="GPT_SoVITS/configs/s1longer-v2.yaml",
        pretrained_sovits_g="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        pretrained_sovits_d="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
        pretrained_gpt="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        sovits_weight_dir="SoVITS_weights_v2",
        gpt_weight_dir="GPT_weights_v2",
        needs_sv_features=False,
    ),
    "v2Pro": VersionSpec(
        name="v2Pro",
        sovits_config="GPT_SoVITS/configs/s2v2Pro.json",
        gpt_config="GPT_SoVITS/configs/s1longer-v2.yaml",
        pretrained_sovits_g="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        pretrained_sovits_d="GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
        pretrained_gpt="GPT_SoVITS/pretrained_models/s1v3.ckpt",
        sovits_weight_dir="SoVITS_weights_v2Pro",
        gpt_weight_dir="GPT_weights_v2Pro",
        needs_sv_features=True,
    ),
    "v2ProPlus": VersionSpec(
        name="v2ProPlus",
        sovits_config="GPT_SoVITS/configs/s2v2ProPlus.json",
        gpt_config="GPT_SoVITS/configs/s1longer-v2.yaml",
        pretrained_sovits_g="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        pretrained_sovits_d="GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
        pretrained_gpt="GPT_SoVITS/pretrained_models/s1v3.ckpt",
        sovits_weight_dir="SoVITS_weights_v2ProPlus",
        gpt_weight_dir="GPT_weights_v2ProPlus",
        needs_sv_features=True,
    ),
}


def get_version(name: str) -> VersionSpec:
    try:
        return SUPPORTED_VERSIONS[name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_VERSIONS)
        raise ValueError(f"Unsupported version '{name}'. Supported versions: {supported}") from exc


def validate_language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported}")
    return language
```

- [ ] **Step 4: Validate config version/language on load**

Modify `gsv_cli/config.py` so `_from_dict` calls `get_version(version)` and `validate_language(language)`. Also validate `infer.ref_language` and `infer.text_language`.

Run:

```bash
python -m unittest tests.test_versions tests.test_config -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gsv_cli/versions.py gsv_cli/config.py tests/test_versions.py
git commit -m "feat: add version and language registry"
```

---

### Task 4: Optional `audio-separator` Wrapper

**Files:**
- Create: `gsv_cli/subprocesses.py`
- Create: `gsv_cli/separate.py`
- Modify: `gsv_cli/app.py`
- Create: `tests/test_separate.py`

- [ ] **Step 1: Write separation tests**

Create `tests/test_separate.py`:

```python
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_separate -v
```

Expected: ERROR because `gsv_cli.separate` does not exist.

- [ ] **Step 3: Implement subprocess helper and wrapper**

Create `gsv_cli/subprocesses.py`:

```python
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int


def run_command(command: Sequence[str], env: Mapping[str, str] | None = None, dry_run: bool = False) -> CommandResult:
    command_list = [str(part) for part in command]
    if dry_run:
        print(" ".join(command_list))
        return CommandResult(command=command_list, returncode=0)
    completed = subprocess.run(command_list, check=False, env=dict(env) if env is not None else None)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command_list)
    return CommandResult(command=command_list, returncode=completed.returncode)
```

Create `gsv_cli/separate.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path

from .subprocesses import CommandResult, run_command


class AudioSeparatorMissing(RuntimeError):
    pass


def ensure_audio_separator(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise AudioSeparatorMissing(
            "audio-separator was not found in PATH. Install and configure python-audio-separator, then retry."
        )
    return resolved


def build_audio_separator_command(
    command: str,
    input_path: str,
    output_dir: str,
    stem: str,
    model: str | None,
    output_format: str,
) -> list[str]:
    cmd = [
        command,
        input_path,
        "--output_dir",
        output_dir,
        "--output_single_stem",
        stem,
    ]
    if model:
        cmd.extend(["--model_filename", model])
    cmd.extend(["--output_format", output_format])
    return cmd


def run_separation(
    command: str,
    input_path: str,
    output_dir: str,
    stem: str,
    model: str | None,
    output_format: str,
    dry_run: bool = False,
) -> CommandResult:
    ensure_audio_separator(command)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_audio_separator_command(command, input_path, output_dir, stem, model, output_format)
    return run_command(cmd, dry_run=dry_run)
```

- [ ] **Step 4: Wire `gsv separate`**

Modify `gsv_cli/app.py` so `separate` accepts:

```text
input
--output-dir
--stem
--model
--output-format
--command
--dry-run
```

Call `run_separation`. Catch `AudioSeparatorMissing`, print its message to stderr, and return exit code `127`.

Run:

```bash
python -m unittest tests.test_separate tests.test_cli_entrypoints -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gsv_cli/subprocesses.py gsv_cli/separate.py gsv_cli/app.py tests/test_separate.py
git commit -m "feat: wrap external audio separator"
```

---

### Task 5: Dataset Preparation Orchestration

**Files:**
- Create: `gsv_cli/paths.py`
- Create: `gsv_cli/prep.py`
- Modify: `gsv_cli/app.py`
- Create: `tests/test_prep.py`

- [ ] **Step 1: Write prep tests**

Create `tests/test_prep.py`:

```python
import unittest

from gsv_cli.config import GsvConfig
from gsv_cli.prep import build_asr_command, build_feature_commands, build_slice_command


class PrepTests(unittest.TestCase):
    def test_slice_command_uses_existing_slicer_script(self):
        cmd = build_slice_command(
            python_exec="python",
            input_path="data/raw",
            output_dir="data/sliced",
            threshold=-34,
            min_length=4000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500,
            max_audio=0.9,
            alpha=0.25,
            part_index=0,
            part_count=1,
        )
        self.assertEqual(cmd[0:3], ["python", "-s", "tools/slice_audio.py"])
        self.assertIn("data/raw", cmd)
        self.assertIn("data/sliced", cmd)

    def test_asr_command_uses_faster_whisper(self):
        cmd = build_asr_command("python", "data/sliced", "output/asr", "large-v3", "zh", "float16")
        self.assertEqual(cmd[0:3], ["python", "-s", "tools/asr/fasterwhisper_asr.py"])
        self.assertIn("-l", cmd)
        self.assertIn("zh", cmd)

    def test_v2proplus_feature_commands_include_sv(self):
        cfg = GsvConfig.default("voice")
        commands = build_feature_commands(cfg, python_exec="python")
        joined = [" ".join(command) for command in commands]
        self.assertTrue(any("1-get-text.py" in command for command in joined))
        self.assertTrue(any("2-get-hubert-wav32k.py" in command for command in joined))
        self.assertTrue(any("2-get-sv.py" in command for command in joined))
        self.assertTrue(any("3-get-semantic.py" in command for command in joined))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_prep -v
```

Expected: ERROR because `gsv_cli.prep` does not exist.

- [ ] **Step 3: Implement path and prep command builders**

Create `gsv_cli/paths.py`:

```python
from __future__ import annotations

from pathlib import Path


def project_path(root: str, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path
```

Create `gsv_cli/prep.py` with pure command builders first:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import GsvConfig
from .subprocesses import CommandResult, run_command
from .versions import get_version, validate_language


def build_slice_command(
    python_exec: str,
    input_path: str,
    output_dir: str,
    threshold: int,
    min_length: int,
    min_interval: int,
    hop_size: int,
    max_sil_kept: int,
    max_audio: float,
    alpha: float,
    part_index: int,
    part_count: int,
) -> list[str]:
    return [
        python_exec,
        "-s",
        "tools/slice_audio.py",
        input_path,
        output_dir,
        str(threshold),
        str(min_length),
        str(min_interval),
        str(hop_size),
        str(max_sil_kept),
        str(max_audio),
        str(alpha),
        str(part_index),
        str(part_count),
    ]


def build_asr_command(
    python_exec: str,
    input_dir: str,
    output_dir: str,
    model: str,
    language: str,
    precision: str,
) -> list[str]:
    validate_language(language)
    return [
        python_exec,
        "-s",
        "tools/asr/fasterwhisper_asr.py",
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-s",
        model,
        "-l",
        language,
        "-p",
        precision,
    ]


def _feature_env(cfg: GsvConfig, part_index: int, part_count: int) -> dict[str, str]:
    version = get_version(cfg.version)
    exp_dir = str(Path(cfg.paths.exp_root) / cfg.project.name)
    return {
        "inp_text": cfg.paths.annotation,
        "inp_wav_dir": cfg.paths.sliced_audio,
        "exp_name": cfg.project.name,
        "opt_dir": exp_dir,
        "i_part": str(part_index),
        "all_parts": str(part_count),
        "_CUDA_VISIBLE_DEVICES": cfg.train.gpu,
        "is_half": "True",
        "bert_pretrained_dir": str(Path(cfg.paths.pretrained_root) / "chinese-roberta-wwm-ext-large"),
        "cnhubert_base_dir": str(Path(cfg.paths.pretrained_root) / "chinese-hubert-base"),
        "sv_path": str(Path(cfg.paths.pretrained_root) / "sv/pretrained_eres2netv2w24s4ep4.ckpt"),
        "pretrained_s2G": version.pretrained_sovits_g,
        "s2config_path": version.sovits_config,
    }


def build_feature_commands(cfg: GsvConfig, python_exec: str = sys.executable) -> list[list[str]]:
    version = get_version(cfg.version)
    commands = [
        [python_exec, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        [python_exec, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
    ]
    if version.needs_sv_features:
        commands.append([python_exec, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"])
    commands.append([python_exec, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"])
    return commands


def run_feature_commands(cfg: GsvConfig, dry_run: bool = False) -> list[CommandResult]:
    env = os.environ.copy()
    env.update(_feature_env(cfg, part_index=0, part_count=1))
    return [run_command(command, env=env, dry_run=dry_run) for command in build_feature_commands(cfg)]
```

- [ ] **Step 4: Wire `gsv prep` subcommands**

Modify `gsv_cli/app.py` to support:

```text
gsv prep slice -c gsv.yaml --input data/vocals --output data/sliced --dry-run
gsv prep asr -c gsv.yaml --input data/sliced --language zh --dry-run
gsv prep features -c gsv.yaml --dry-run
gsv prep all -c gsv.yaml --dry-run
```

Load config with `load_config`. For `prep all`, run slice, ASR only when annotation does not exist, then features.

Run:

```bash
python -m unittest tests.test_prep tests.test_versions tests.test_config -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gsv_cli/paths.py gsv_cli/prep.py gsv_cli/app.py tests/test_prep.py
git commit -m "feat: add dataset prep orchestration"
```

---

### Task 6: Training Config Generation And Dry-Run Commands

**Files:**
- Create: `gsv_cli/train.py`
- Modify: `gsv_cli/app.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write training tests**

Create `tests/test_train.py`:

```python
import tempfile
import unittest
from pathlib import Path

from gsv_cli.config import GsvConfig
from gsv_cli.train import build_gpt_train_command, build_sovits_train_command, write_gpt_config, write_sovits_config


class TrainTests(unittest.TestCase):
    def test_sovits_command_targets_s2_train(self):
        cmd = build_sovits_train_command("python", "tmp_s2.json")
        self.assertEqual(cmd, ["python", "-s", "GPT_SoVITS/s2_train.py", "--config", "tmp_s2.json"])

    def test_gpt_command_targets_s1_train(self):
        cmd = build_gpt_train_command("python", "tmp_s1.yaml")
        self.assertEqual(cmd, ["python", "-s", "GPT_SoVITS/s1_train.py", "--config_file", "tmp_s1.yaml"])

    def test_generated_configs_include_weight_dirs(self):
        cfg = GsvConfig.default("voice")
        with tempfile.TemporaryDirectory() as tmp:
            s2_path = write_sovits_config(cfg, Path(tmp))
            s1_path = write_gpt_config(cfg, Path(tmp))
            self.assertTrue(s2_path.exists())
            self.assertTrue(s1_path.exists())
            self.assertIn("tmp_s2", s2_path.name)
            self.assertIn("tmp_s1", s1_path.name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_train -v
```

Expected: ERROR because `gsv_cli.train` does not exist.

- [ ] **Step 3: Implement config generation**

Create `gsv_cli/train.py`:

```python
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

from .config import GsvConfig
from .subprocesses import CommandResult, run_command
from .versions import get_version


def build_sovits_train_command(python_exec: str, config_path: str) -> list[str]:
    return [python_exec, "-s", "GPT_SoVITS/s2_train.py", "--config", config_path]


def build_gpt_train_command(python_exec: str, config_path: str) -> list[str]:
    return [python_exec, "-s", "GPT_SoVITS/s1_train.py", "--config_file", config_path]


def write_sovits_config(cfg: GsvConfig, output_dir: Path) -> Path:
    version = get_version(cfg.version)
    with Path(version.sovits_config).open("r", encoding="utf-8") as f:
        data = json.load(f)
    exp_dir = str(Path(cfg.paths.exp_root) / cfg.project.name)
    data = copy.deepcopy(data)
    data["train"]["batch_size"] = cfg.train.sovits_batch_size
    data["train"]["epochs"] = cfg.train.sovits_epochs
    data["train"]["pretrained_s2G"] = version.pretrained_sovits_g
    data["train"]["pretrained_s2D"] = version.pretrained_sovits_d
    data["train"]["if_save_latest"] = cfg.train.save_latest
    data["train"]["if_save_every_weights"] = cfg.train.save_every_weights
    data["train"]["save_every_epoch"] = cfg.train.save_every_epoch
    data["train"]["gpu_numbers"] = cfg.train.gpu
    data["train"]["grad_ckpt"] = cfg.train.grad_ckpt
    data["model"]["version"] = cfg.version
    data["data"]["exp_dir"] = exp_dir
    data["s2_ckpt_dir"] = exp_dir
    data["save_weight_dir"] = version.sovits_weight_dir
    data["name"] = cfg.project.name
    data["version"] = cfg.version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tmp_s2.json"
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_gpt_config(cfg: GsvConfig, output_dir: Path) -> Path:
    version = get_version(cfg.version)
    with Path(version.gpt_config).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    exp_dir = str(Path(cfg.paths.exp_root) / cfg.project.name)
    data = copy.deepcopy(data)
    data["train"]["batch_size"] = cfg.train.gpt_batch_size
    data["train"]["epochs"] = cfg.train.gpt_epochs
    data["pretrained_s1"] = version.pretrained_gpt
    data["train"]["save_every_n_epoch"] = cfg.train.save_every_epoch
    data["train"]["if_save_every_weights"] = cfg.train.save_every_weights
    data["train"]["if_save_latest"] = cfg.train.save_latest
    data["train"]["half_weights_save_dir"] = version.gpt_weight_dir
    data["train"]["exp_name"] = cfg.project.name
    data["train_semantic_path"] = f"{exp_dir}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"{exp_dir}/2-name2text.txt"
    data["output_dir"] = f"{exp_dir}/logs_s1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tmp_s1.yaml"
    output_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_path


def run_sovits_training(cfg: GsvConfig, dry_run: bool = False, python_exec: str = sys.executable) -> CommandResult:
    config_path = write_sovits_config(cfg, Path(cfg.paths.exp_root) / cfg.project.name)
    return run_command(build_sovits_train_command(python_exec, str(config_path)), dry_run=dry_run)


def run_gpt_training(cfg: GsvConfig, dry_run: bool = False, python_exec: str = sys.executable) -> CommandResult:
    config_path = write_gpt_config(cfg, Path(cfg.paths.exp_root) / cfg.project.name)
    return run_command(build_gpt_train_command(python_exec, str(config_path)), dry_run=dry_run)
```

- [ ] **Step 4: Wire `gsv train` subcommands**

Modify `gsv_cli/app.py` to support:

```text
gsv train sovits -c gsv.yaml --batch-size 2 --dry-run
gsv train gpt -c gsv.yaml --batch-size 1 --dry-run
gsv train all -c gsv.yaml --dry-run
```

`--batch-size` maps to `train.sovits_batch_size` for `sovits`, and `train.gpt_batch_size` for `gpt`. `train all` uses config values.

Run:

```bash
python -m unittest tests.test_train tests.test_config tests.test_versions -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gsv_cli/train.py gsv_cli/app.py tests/test_train.py
git commit -m "feat: add training command generation"
```

---

### Task 7: Direct Inference Command Without WebUI Import

**Files:**
- Create: `gsv_cli/infer.py`
- Modify: `gsv_cli/app.py`
- Create: `tests/test_infer.py`

- [ ] **Step 1: Write inference validation tests**

Create `tests/test_infer.py`:

```python
import unittest

from gsv_cli.config import GsvConfig
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_infer -v
```

Expected: ERROR because `gsv_cli.infer` does not exist.

- [ ] **Step 3: Implement inference adapter**

Create `gsv_cli/infer.py`:

```python
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .config import GsvConfig
from .versions import get_version, validate_language


def make_tts_config(cfg: GsvConfig, device: str, is_half: bool) -> dict:
    version = get_version(cfg.version)
    return {
        "custom": {
            "device": device,
            "is_half": is_half,
            "version": cfg.version,
            "t2s_weights_path": cfg.infer.gpt_weight or version.pretrained_gpt,
            "vits_weights_path": cfg.infer.sovits_weight or version.pretrained_sovits_g,
            "cnhuhbert_base_path": str(Path(cfg.paths.pretrained_root) / "chinese-hubert-base"),
            "bert_base_path": str(Path(cfg.paths.pretrained_root) / "chinese-roberta-wwm-ext-large"),
        }
    }


def build_tts_inputs(
    text: str,
    text_language: str,
    ref_audio: str,
    ref_text: str,
    ref_language: str,
) -> dict:
    validate_language(text_language)
    validate_language(ref_language)
    return {
        "text": text,
        "text_lang": text_language,
        "ref_audio_path": ref_audio,
        "prompt_text": ref_text,
        "prompt_lang": ref_language,
        "top_k": 15,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut1",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": -1,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }


def synthesize_to_file(
    cfg: GsvConfig,
    text: str,
    ref_audio: str,
    ref_text: str,
    output_path: str,
    text_language: str,
    ref_language: str,
    device: str,
    is_half: bool,
) -> Path:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS

    tts = TTS(make_tts_config(cfg, device=device, is_half=is_half))
    sampling_rate, audio = tts.run(
        build_tts_inputs(
            text=text,
            text_language=text_language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            ref_language=ref_language,
        )
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sampling_rate)
    return output
```

- [ ] **Step 4: Wire `gsv infer`**

Modify `gsv_cli/app.py` to support:

```text
gsv infer -c gsv.yaml --text "hello" --ref ref.wav --ref-text "sample text" --out out.wav --device cpu
gsv infer -c gsv.yaml --text-file text.txt --ref ref.wav --ref-text-file ref.txt --out out.wav
```

If `--dry-run` is passed, print the resolved TTS config and inputs without importing `GPT_SoVITS.TTS_infer_pack.TTS`.

Run:

```bash
python -m unittest tests.test_infer tests.test_versions tests.test_config -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gsv_cli/infer.py gsv_cli/app.py tests/test_infer.py
git commit -m "feat: add cli inference adapter"
```

---

### Task 8: Installer Integration And Existing Linux Fixes

**Files:**
- Modify: `install.sh`
- Modify: `requirements.txt`

- [ ] **Step 1: Write installer regression checks as shell commands**

Run these commands before editing:

```bash
rg -n "unzip -q -o nltk_data" install.sh
rg -n "run_pip_quiet torch torchcodec" install.sh
rg -n "gradio<5|fastapi\\[standard\\]" requirements.txt
```

Expected: current branch still shows the old installer behavior.

- [ ] **Step 2: Apply installer fixes**

Modify `install.sh` to:

- Avoid failing when `tput` cannot handle the active terminal.
- Install `torchaudio` from the same PyTorch index as `torch`.
- Unzip `nltk_data.zip`, not `nltk_data`.
- Run `pip install -e .` after requirements install so `gsv` and `gpt-sovits` scripts exist.

The relevant changes must match:

```bash
if tput cuu1 >/dev/null 2>&1; then
    tput el
fi
```

```bash
run_pip_quiet torch torchaudio torchcodec --index-url "https://download.pytorch.org/whl/cu128"
```

```bash
unzip -q -o nltk_data.zip -d "$PY_PREFIX"
```

```bash
run_pip_quiet -e .
```

- [ ] **Step 3: Keep requirements compatible with CLI**

Keep `PyYAML`, `soundfile`, `faster-whisper`, `torch`, `torchaudio`, and training dependencies. Do not remove Gradio yet in this task because GUI deletion happens after CLI verification.

If `starlette` remains necessary for the still-present WebUI during migration, pin:

```text
starlette<1
```

- [ ] **Step 4: Verify installer text changes**

Run:

```bash
rg -n "torchaudio|nltk_data.zip|pip install -e|starlette<1" install.sh requirements.txt
```

Expected: output includes all intended changes.

- [ ] **Step 5: Commit**

```bash
git add install.sh requirements.txt
git commit -m "fix: harden linux cli installation"
```

---

### Task 9: README CLI Workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace installation and usage sections**

Edit `README.md` to document:

```markdown
## Supported Workflow

- Train on Linux with NVIDIA CUDA.
- Run inference on Linux or macOS.
- Use `gsv` or `gpt-sovits`; both commands are equivalent.

## Install

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device CU128 --source HF-Mirror
```

## CLI Quick Start

```bash
gsv init myvoice --version v2ProPlus
gsv separate input.wav --output-dir data/separated --stem vocals
gsv prep slice -c myvoice/gsv.yaml --input data/separated --output data/sliced
gsv prep asr -c myvoice/gsv.yaml --input data/sliced --language zh
gsv prep features -c myvoice/gsv.yaml
gsv train sovits -c myvoice/gsv.yaml --batch-size 2
gsv train gpt -c myvoice/gsv.yaml --batch-size 1
gsv infer -c myvoice/gsv.yaml --text "你好，这是测试。" --ref ref.wav --ref-text "参考文本。" --out out.wav
```

## Supported Versions

- `v2`
- `v2Pro`
- `v2ProPlus`

`v2ProPlus` is the default.

## Supported Languages

- `zh`
- `en`
- `ja`
```

- [ ] **Step 2: Remove Windows/WebUI-first claims**

Remove root README sections that present Windows packages, `.bat`, `.ps1`, WebUI-first training, Colab-first instructions, and UVR5 as the primary separation path.

- [ ] **Step 3: Verify docs mention CLI path**

Run:

```bash
rg -n "gsv init|v2ProPlus|Supported Languages|go-webui|install.ps1|UVR5" README.md
```

Expected: CLI terms are present; Windows/WebUI/UVR5 terms are absent or clearly marked as removed legacy notes.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document cli-first workflow"
```

---

### Task 10: CLI Verification Gate

**Files:**
- No source changes unless verification exposes a bug.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 2: Verify CLI help and dry-run commands**

Run:

```bash
python -m gsv_cli --help
python -m gsv_cli init /tmp/gsv-smoke --version v2ProPlus
python -m gsv_cli prep features -c /tmp/gsv-smoke/gsv.yaml --dry-run
python -m gsv_cli train sovits -c /tmp/gsv-smoke/gsv.yaml --dry-run
python -m gsv_cli train gpt -c /tmp/gsv-smoke/gsv.yaml --dry-run
python -m gsv_cli infer -c /tmp/gsv-smoke/gsv.yaml --text "hello" --ref ref.wav --ref-text "hello" --out /tmp/out.wav --dry-run
```

Expected: all commands exit 0 and print commands/configuration without launching long training.

- [ ] **Step 3: Verify editable script install**

Run:

```bash
pip install -e .
gsv --version
gpt-sovits --version
```

Expected: both commands print `0.1.0`.

- [ ] **Step 4: Commit fixes if required**

If any verification command fails, fix the failing code and commit:

```bash
git add gsv_cli tests pyproject.toml README.md install.sh requirements.txt
git commit -m "fix: complete cli verification gate"
```

If no source changes are required, do not create an empty commit.

---

### Task 11: Remove Windows, Colab, GUI Entrypoints, And UVR5 Primary Path

**Files:**
- Delete: `go-webui.bat`
- Delete: `go-webui.ps1`
- Delete: `install.ps1`
- Delete: `Colab-WebUI.ipynb`
- Delete: `Colab-Inference.ipynb`
- Delete: `webui.py`
- Delete: `GPT_SoVITS/inference_webui.py`
- Delete: `GPT_SoVITS/inference_webui_fast.py`
- Delete: `GPT_SoVITS/inference_gui.py`
- Delete: `tools/subfix_webui.py`
- Delete: `tools/uvr5/`
- Modify: `requirements.txt`

- [ ] **Step 1: Confirm CLI verification is complete**

Run:

```bash
python -m unittest discover -s tests -v
python -m gsv_cli train sovits -c /tmp/gsv-smoke/gsv.yaml --dry-run
python -m gsv_cli infer -c /tmp/gsv-smoke/gsv.yaml --text "hello" --ref ref.wav --ref-text "hello" --out /tmp/out.wav --dry-run
```

Expected: PASS/exit 0.

- [ ] **Step 2: Delete obsolete files**

Run:

```bash
rm -f go-webui.bat go-webui.ps1 install.ps1 Colab-WebUI.ipynb Colab-Inference.ipynb
rm -f webui.py GPT_SoVITS/inference_webui.py GPT_SoVITS/inference_webui_fast.py GPT_SoVITS/inference_gui.py tools/subfix_webui.py
rm -rf tools/uvr5
```

- [ ] **Step 3: Remove GUI dependencies**

Edit `requirements.txt` and remove:

```text
gradio<5
fastapi[standard]>=0.115.2
starlette<1
```

Keep `ffmpeg-python` because inference still uses FFmpeg paths.

- [ ] **Step 4: Verify deleted imports are gone from CLI path**

Run:

```bash
rg -n "gradio|inference_webui|tools/uvr5|go-webui|install.ps1" .
python -m unittest discover -s tests -v
```

Expected: `rg` may only find historical changelog or spec references; no runtime imports remain. Tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove gui and windows entrypoints"
```

---

### Task 12: Trim UI i18n And Unsupported Language Frontends

**Files:**
- Delete: `tools/i18n/`
- Delete or move out of supported path: `GPT_SoVITS/text/korean.py`
- Delete or move out of supported path: `GPT_SoVITS/text/cantonese.py`
- Modify: `GPT_SoVITS/text/cleaner.py`
- Modify: `GPT_SoVITS/TTS_infer_pack/TTS.py`
- Modify: `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`
- Modify: `requirements.txt`
- Modify: tests as needed.

- [ ] **Step 1: Identify remaining runtime i18n imports**

Run:

```bash
rg -n "tools\\.i18n|I18nAuto|scan_language_list|\\bko\\b|\\byue\\b|korean|cantonese" GPT_SoVITS gsv_cli tools requirements.txt
```

Expected: output lists remaining imports/usages that must be removed or guarded.

- [ ] **Step 2: Replace runtime i18n calls in inference core**

In `GPT_SoVITS/TTS_infer_pack/TTS.py`, replace calls shaped like `i18n("message")` with plain string messages. Replace language list usage with:

```python
SUPPORTED_TTS_LANGUAGES = ("zh", "en", "ja", "auto", "all_zh", "all_ja")
```

Then restrict `TTS_Config.v2_languages` to:

```python
v2_languages: list = ["auto", "en", "zh", "ja", "all_zh", "all_ja"]
```

- [ ] **Step 3: Remove unsupported cleaner branches**

In `GPT_SoVITS/text/cleaner.py`, remove public paths that dispatch to Korean or Cantonese. Keep Chinese, English, Japanese, and the existing mixed-language behavior only when it uses those three.

- [ ] **Step 4: Remove unsupported dependencies**

Edit `requirements.txt` and remove Korean/Cantonese-only dependencies:

```text
ToJyutping
g2pk2
ko_pron
python_mecab_ko; sys_platform != 'win32'
```

Do not remove dependencies used by Chinese, English, or Japanese text processing.

- [ ] **Step 5: Delete i18n and unsupported language files**

Run:

```bash
rm -rf tools/i18n
rm -f GPT_SoVITS/text/korean.py GPT_SoVITS/text/cantonese.py
```

- [ ] **Step 6: Verify CLI tests and import smoke**

Run:

```bash
python -m unittest discover -s tests -v
python - <<'PY'
from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config
cfg = TTS_Config({"custom": {"version": "v2", "device": "cpu", "is_half": False, "t2s_weights_path": "", "vits_weights_path": "", "bert_base_path": "", "cnhuhbert_base_path": ""}})
assert "ko" not in cfg.languages
assert "yue" not in cfg.languages
print(cfg.languages)
PY
```

Expected: unit tests pass; smoke script prints a list without `ko` or `yue`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: trim i18n and unsupported languages"
```

---

### Task 13: Final End-To-End Verification

**Files:**
- No source changes unless verification exposes a bug.

- [ ] **Step 1: Run full CLI unit suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 2: Run dry-run training loop**

Run:

```bash
rm -rf /tmp/gsv-e2e
python -m gsv_cli init /tmp/gsv-e2e --version v2ProPlus
python -m gsv_cli prep features -c /tmp/gsv-e2e/gsv.yaml --dry-run
python -m gsv_cli train all -c /tmp/gsv-e2e/gsv.yaml --dry-run
python -m gsv_cli infer -c /tmp/gsv-e2e/gsv.yaml --text "hello" --ref ref.wav --ref-text "hello" --out /tmp/gsv-e2e/out.wav --dry-run
```

Expected: all commands exit 0.

- [ ] **Step 3: Run repository search for removed paths**

Run:

```bash
rg -n "go-webui|install.ps1|gradio|tools/i18n|tools/uvr5|FunASR|ModelScope|\\byue\\b|\\bko\\b" .
```

Expected: matches are absent from runtime code. Matches in committed design/plan docs are acceptable.

- [ ] **Step 4: Optional real inference**

If local weights and reference audio are available, run:

```bash
gsv infer -c /tmp/gsv-e2e/gsv.yaml --text "你好，这是测试。" --ref /absolute/path/ref.wav --ref-text "参考文本。" --out /tmp/gsv-e2e/out.wav
```

Expected: `/tmp/gsv-e2e/out.wav` exists and is playable.

- [ ] **Step 5: Commit verification fixes if required**

If verification required source changes:

```bash
git add -A
git commit -m "fix: complete cli-first cleanup verification"
```

If no source changes were required, do not create an empty commit.
