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


def write_config(path: str | Path, cfg: GsvConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)


def write_default_config(path: str | Path, project_name: str) -> None:
    write_config(path, GsvConfig.default(project_name))
