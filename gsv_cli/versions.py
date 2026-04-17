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
