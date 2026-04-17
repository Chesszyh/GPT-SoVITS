from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import GsvConfig, load_config, write_config
from .prep import run_asr, run_feature_commands, run_slice
from .separate import AudioSeparatorMissing, run_separation
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
    print(f"Command '{args.command}' is not implemented yet")
    return 2


def console_main() -> None:
    raise SystemExit(main())
