from __future__ import annotations

import os
import shutil
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


def expected_asr_output_path(input_dir: str, output_dir: str) -> Path:
    return Path(output_dir) / f"{Path(input_dir).name}.list"


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
        "version": cfg.version,
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


def merge_feature_outputs(cfg: GsvConfig, part_count: int = 1) -> None:
    exp_dir = Path(cfg.paths.exp_root) / cfg.project.name
    text_lines: list[str] = []
    semantic_lines = ["item_name\tsemantic_audio"]
    for part_index in range(part_count):
        text_part = exp_dir / f"2-name2text-{part_index}.txt"
        semantic_part = exp_dir / f"6-name2semantic-{part_index}.tsv"
        text_lines.extend(line for line in text_part.read_text(encoding="utf-8").splitlines() if line)
        semantic_lines.extend(line for line in semantic_part.read_text(encoding="utf-8").splitlines() if line)
    (exp_dir / "2-name2text.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (exp_dir / "6-name2semantic.tsv").write_text("\n".join(semantic_lines) + "\n", encoding="utf-8")


def run_slice(
    input_path: str,
    output_dir: str,
    threshold: int = -34,
    min_length: int = 4000,
    min_interval: int = 300,
    hop_size: int = 10,
    max_sil_kept: int = 500,
    max_audio: float = 0.9,
    alpha: float = 0.25,
    part_index: int = 0,
    part_count: int = 1,
    dry_run: bool = False,
    python_exec: str = sys.executable,
) -> CommandResult:
    command = build_slice_command(
        python_exec=python_exec,
        input_path=input_path,
        output_dir=output_dir,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept,
        max_audio=max_audio,
        alpha=alpha,
        part_index=part_index,
        part_count=part_count,
    )
    return run_command(command, dry_run=dry_run)


def run_asr(
    cfg: GsvConfig,
    input_dir: str,
    output_dir: str,
    language: str,
    model: str,
    precision: str,
    dry_run: bool = False,
    python_exec: str = sys.executable,
) -> CommandResult:
    command = build_asr_command(
        python_exec=python_exec,
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        language=language,
        precision=precision,
    )
    result = run_command(command, dry_run=dry_run)
    if not dry_run:
        annotation = Path(cfg.paths.annotation)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(expected_asr_output_path(input_dir, output_dir), annotation)
    return result


def run_feature_commands(
    cfg: GsvConfig,
    dry_run: bool = False,
    python_exec: str = sys.executable,
    part_count: int = 1,
) -> list[CommandResult]:
    env = os.environ.copy()
    env.update(_feature_env(cfg, part_index=0, part_count=part_count))
    results = [run_command(command, env=env, dry_run=dry_run) for command in build_feature_commands(cfg, python_exec)]
    if not dry_run:
        merge_feature_outputs(cfg, part_count=part_count)
    return results
