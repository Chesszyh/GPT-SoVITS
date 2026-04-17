from __future__ import annotations

from pathlib import Path

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
        "aux_ref_audio_paths": [],
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
        "sample_steps": 32,
        "super_sampling": False,
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
    import soundfile as sf

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS

    tts = TTS(make_tts_config(cfg, device=device, is_half=is_half))
    sampling_rate, audio = next(
        tts.run(
            build_tts_inputs(
                text=text,
                text_language=text_language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                ref_language=ref_language,
            )
        )
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sampling_rate)
    return output
