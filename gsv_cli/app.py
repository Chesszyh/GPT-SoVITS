from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import GsvConfig, write_config
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
    print(f"Command '{args.command}' is not implemented yet")
    return 2


def console_main() -> None:
    raise SystemExit(main())
