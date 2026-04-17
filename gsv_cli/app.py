from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from . import __version__
from .config import GsvConfig, load_config, write_config
from .infer import build_tts_inputs, make_tts_config, synthesize_to_file
from .prep import run_asr, run_feature_commands, run_slice
from .separate import AudioSeparatorMissing, run_separation
from .train import run_gpt_training, run_sovits_training
from .versions import get_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsv", description="GPT-SoVITS CLI")
    parser.add_argument("--version", dest="show_version", action="store_true", help="Print CLI version and exit")
    subparsers = parser.add_subparsers(dest="command")
    init = subparsers.add_parser("init", help="Create a gsv.yaml project config")
    init.add_argument("name", help="Project name or project directory")
    init.add_argument("--config", help="Config path to write")
    init.add_argument("--version", default="v2ProPlus", help="Model version")
    separate = subparsers.add_parser("separate", help="Run optional audio-separator wrapper")
    separate.add_argument("input", help="Input audio file")
    separate.add_argument("--output-dir", default="data/separated", help="Directory for separated stems")
    separate.add_argument("--stem", default="vocals", help="Single stem to output")
    separate.add_argument("--model", help="audio-separator model filename")
    separate.add_argument("--output-format", default="WAV", help="Separated audio format")
    separate.add_argument(
        "--command",
        dest="separator_command",
        default="audio-separator",
        help="audio-separator executable",
    )
    separate.add_argument("--dry-run", action="store_true", help="Print command without running it")

    prep_common = argparse.ArgumentParser(add_help=False)
    prep_common.add_argument("-c", "--config", default="gsv.yaml", help="Project config path")
    prep_common.add_argument("--dry-run", action="store_true", help="Print commands without running them")

    prep = subparsers.add_parser("prep", help="Prepare dataset artifacts")
    prep_sub = prep.add_subparsers(dest="prep_command")
    prep_slice = prep_sub.add_parser("slice", parents=[prep_common])
    prep_slice.add_argument("--input", help="Input audio file or directory")
    prep_slice.add_argument("--output", help="Sliced audio output directory")
    prep_slice.add_argument("--threshold", type=int, default=-34)
    prep_slice.add_argument("--min-length", type=int, default=4000)
    prep_slice.add_argument("--min-interval", type=int, default=300)
    prep_slice.add_argument("--hop-size", type=int, default=10)
    prep_slice.add_argument("--max-sil-kept", type=int, default=500)
    prep_slice.add_argument("--max-audio", type=float, default=0.9)
    prep_slice.add_argument("--alpha", type=float, default=0.25)
    prep_slice.add_argument("--part-index", type=int, default=0)
    prep_slice.add_argument("--part-count", type=int, default=1)

    prep_asr = prep_sub.add_parser("asr", parents=[prep_common])
    prep_asr.add_argument("--input", help="Sliced audio input directory")
    prep_asr.add_argument("--output", help="ASR list output directory")
    prep_asr.add_argument("--language", help="ASR language: zh, en, or ja")
    prep_asr.add_argument("--model", help="Faster-Whisper model size")
    prep_asr.add_argument("--precision", help="Faster-Whisper compute precision")

    prep_sub.add_parser("features", parents=[prep_common])
    prep_sub.add_parser("all", parents=[prep_common])

    train_common = argparse.ArgumentParser(add_help=False)
    train_common.add_argument("-c", "--config", default="gsv.yaml", help="Project config path")
    train_common.add_argument("--dry-run", action="store_true", help="Print commands without running them")

    train = subparsers.add_parser("train", help="Train GPT-SoVITS weights")
    train_sub = train.add_subparsers(dest="train_command")
    train_sovits = train_sub.add_parser("sovits", parents=[train_common])
    train_sovits.add_argument("--batch-size", type=int)
    train_sovits.add_argument("--epochs", type=int)
    train_gpt = train_sub.add_parser("gpt", parents=[train_common])
    train_gpt.add_argument("--batch-size", type=int)
    train_gpt.add_argument("--epochs", type=int)
    train_sub.add_parser("all", parents=[train_common])

    infer = subparsers.add_parser("infer", help="Synthesize audio")
    infer.add_argument("-c", "--config", default="gsv.yaml", help="Project config path")
    infer.add_argument("--text", help="Text to synthesize")
    infer.add_argument("--text-file", help="UTF-8 file containing text to synthesize")
    infer.add_argument("--ref", dest="ref_audio", help="Reference audio path")
    infer.add_argument("--ref-text", help="Reference audio transcript")
    infer.add_argument("--ref-text-file", help="UTF-8 file containing reference transcript")
    infer.add_argument("--text-language", help="Text language: zh, en, or ja")
    infer.add_argument("--ref-language", help="Reference text language: zh, en, or ja")
    infer.add_argument("--out", required=True, help="Output wav path")
    infer.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, cuda, or mps")
    infer.add_argument("--half", action="store_true", help="Use half precision when the device supports it")
    infer.add_argument("--dry-run", action="store_true", help="Print resolved TTS config and inputs")
    return parser


def _read_text_value(value: str | None, file_path: str | None, label: str) -> str:
    if value and file_path:
        raise ValueError(f"Use either --{label} or --{label}-file, not both")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    if value:
        return value
    raise ValueError(f"Missing required --{label} or --{label}-file")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.show_version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "init":
        get_version(args.version)
        cfg = GsvConfig.default(Path(args.name).name).with_overrides({"version": args.version})
        config_path = Path(args.config) if args.config else Path(args.name) / "gsv.yaml"
        write_config(config_path, cfg)
        print(f"Wrote {config_path}")
        return 0
    if args.command == "separate":
        try:
            run_separation(
                command=args.separator_command,
                input_path=args.input,
                output_dir=args.output_dir,
                stem=args.stem,
                model=args.model,
                output_format=args.output_format,
                dry_run=args.dry_run,
            )
        except AudioSeparatorMissing as exc:
            print(str(exc), file=sys.stderr)
            return 127
        return 0
    if args.command == "prep":
        cfg = load_config(args.config)
        if args.prep_command == "slice":
            run_slice(
                input_path=args.input or cfg.paths.raw_audio,
                output_dir=args.output or cfg.paths.sliced_audio,
                threshold=args.threshold,
                min_length=args.min_length,
                min_interval=args.min_interval,
                hop_size=args.hop_size,
                max_sil_kept=args.max_sil_kept,
                max_audio=args.max_audio,
                alpha=args.alpha,
                part_index=args.part_index,
                part_count=args.part_count,
                dry_run=args.dry_run,
            )
            return 0
        if args.prep_command == "asr":
            run_asr(
                cfg=cfg,
                input_dir=args.input or cfg.paths.sliced_audio,
                output_dir=args.output or str(Path(cfg.paths.annotation).parent),
                language=args.language or cfg.language,
                model=args.model or cfg.asr.model,
                precision=args.precision or cfg.asr.precision,
                dry_run=args.dry_run,
            )
            return 0
        if args.prep_command == "features":
            run_feature_commands(cfg, dry_run=args.dry_run)
            return 0
        if args.prep_command == "all":
            run_slice(
                input_path=cfg.paths.raw_audio,
                output_dir=cfg.paths.sliced_audio,
                dry_run=args.dry_run,
            )
            annotation = Path(cfg.paths.annotation)
            if args.dry_run or not annotation.exists():
                run_asr(
                    cfg=cfg,
                    input_dir=cfg.paths.sliced_audio,
                    output_dir=str(annotation.parent),
                    language=cfg.language,
                    model=cfg.asr.model,
                    precision=cfg.asr.precision,
                    dry_run=args.dry_run,
                )
            run_feature_commands(cfg, dry_run=args.dry_run)
            return 0
    if args.command == "train":
        cfg = load_config(args.config)
        if args.train_command == "sovits":
            overrides = {}
            if args.batch_size is not None:
                overrides["train.sovits_batch_size"] = args.batch_size
            if args.epochs is not None:
                overrides["train.sovits_epochs"] = args.epochs
            run_sovits_training(cfg.with_overrides(overrides), dry_run=args.dry_run)
            return 0
        if args.train_command == "gpt":
            overrides = {}
            if args.batch_size is not None:
                overrides["train.gpt_batch_size"] = args.batch_size
            if args.epochs is not None:
                overrides["train.gpt_epochs"] = args.epochs
            run_gpt_training(cfg.with_overrides(overrides), dry_run=args.dry_run)
            return 0
        if args.train_command == "all":
            run_sovits_training(cfg, dry_run=args.dry_run)
            run_gpt_training(cfg, dry_run=args.dry_run)
            return 0
    if args.command == "infer":
        cfg = load_config(args.config)
        text = _read_text_value(args.text, args.text_file, "text")
        ref_text = _read_text_value(args.ref_text, args.ref_text_file, "ref-text")
        ref_audio = args.ref_audio or cfg.infer.ref_audio
        if not ref_audio:
            raise ValueError("Missing required --ref or infer.ref_audio in config")
        text_language = args.text_language or cfg.infer.text_language
        ref_language = args.ref_language or cfg.infer.ref_language
        tts_config = make_tts_config(cfg, device=args.device, is_half=args.half)
        tts_inputs = build_tts_inputs(
            text=text,
            text_language=text_language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            ref_language=ref_language,
        )
        if args.dry_run:
            print(yaml.safe_dump({"tts_config": tts_config, "inputs": tts_inputs}, allow_unicode=True, sort_keys=False))
            return 0
        synthesize_to_file(
            cfg=cfg,
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_path=args.out,
            text_language=text_language,
            ref_language=ref_language,
            device=args.device,
            is_half=args.half,
        )
        return 0
    print(f"Command '{args.command}' is not implemented yet")
    return 2


def console_main() -> None:
    raise SystemExit(main())
