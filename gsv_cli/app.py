from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import GsvConfig, write_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsv", description="GPT-SoVITS CLI")
    parser.add_argument("--version", action="store_true", help="Print CLI version and exit")
    subparsers = parser.add_subparsers(dest="command")
    init = subparsers.add_parser("init", help="Create a gsv.yaml project config")
    init.add_argument("name", help="Project name or project directory")
    init.add_argument("--config", help="Config path to write")
    init.add_argument("--version", default="v2ProPlus", help="Model version")
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
    if args.command == "init":
        cfg = GsvConfig.default(Path(args.name).name).with_overrides({"version": args.version})
        config_path = Path(args.config) if args.config else Path(args.name) / "gsv.yaml"
        write_config(config_path, cfg)
        print(f"Wrote {config_path}")
        return 0
    print(f"Command '{args.command}' is not implemented yet")
    return 2


def console_main() -> None:
    raise SystemExit(main())
