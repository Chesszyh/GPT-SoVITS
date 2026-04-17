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
