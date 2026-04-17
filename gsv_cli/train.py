from __future__ import annotations

import copy
import json
import os
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
    data = copy.deepcopy(data)
    exp_dir = str(Path(cfg.paths.exp_root) / cfg.project.name)
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
    data = copy.deepcopy(data)
    exp_dir = str(Path(cfg.paths.exp_root) / cfg.project.name)
    data["train"]["batch_size"] = cfg.train.gpt_batch_size
    data["train"]["epochs"] = cfg.train.gpt_epochs
    data["pretrained_s1"] = version.pretrained_gpt
    data["train"]["save_every_n_epoch"] = cfg.train.save_every_epoch
    data["train"]["if_save_every_weights"] = cfg.train.save_every_weights
    data["train"]["if_save_latest"] = cfg.train.save_latest
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = version.gpt_weight_dir
    data["train"]["exp_name"] = cfg.project.name
    data["train_semantic_path"] = f"{exp_dir}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"{exp_dir}/2-name2text.txt"
    data["output_dir"] = f"{exp_dir}/logs_s1_{cfg.version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tmp_s1.yaml"
    output_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_path


def run_sovits_training(
    cfg: GsvConfig,
    dry_run: bool = False,
    python_exec: str = sys.executable,
) -> CommandResult:
    config_path = write_sovits_config(cfg, Path(cfg.paths.exp_root) / cfg.project.name)
    return run_command(build_sovits_train_command(python_exec, str(config_path)), dry_run=dry_run)


def run_gpt_training(
    cfg: GsvConfig,
    dry_run: bool = False,
    python_exec: str = sys.executable,
) -> CommandResult:
    config_path = write_gpt_config(cfg, Path(cfg.paths.exp_root) / cfg.project.name)
    env = os.environ.copy()
    env["_CUDA_VISIBLE_DEVICES"] = cfg.train.gpu.replace("-", ",")
    env["hz"] = "25hz"
    return run_command(build_gpt_train_command(python_exec, str(config_path)), env=env, dry_run=dry_run)
