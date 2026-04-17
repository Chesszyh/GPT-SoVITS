from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int


def run_command(
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> CommandResult:
    command_list = [str(part) for part in command]
    if dry_run:
        print(" ".join(command_list))
        return CommandResult(command=command_list, returncode=0)
    completed = subprocess.run(command_list, check=False, env=dict(env) if env is not None else None)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command_list)
    return CommandResult(command=command_list, returncode=completed.returncode)
