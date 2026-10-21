"""Open the existing official agy session with the project-owned operator prompt."""

from __future__ import annotations

import os
import subprocess

from .contracts import PACKAGE, read_regular
from .runtime import profile_from
from .subscription import GeminiSubscriptionRunner


def main():
    repo = PACKAGE.parent
    profile = profile_from(PACKAGE / "run-profile.json")
    runner = GeminiSubscriptionRunner(
        cli_bin=profile["cli_bin"], default_model=profile["default_model"]
    )
    runner.preflight()
    environment = dict(runner.environment)
    # Interactive operator cwd is the project, so its local runtime is inside
    # that sandbox. Isolated role workspaces deliberately use only system tools.
    environment["PATH"] = (
        str(repo / ".venv/bin")
        + ":/usr/local/bin:/opt/homebrew/bin:"
        + environment["PATH"]
    )
    environment.update(
        {
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "HARNESS2_SOURCE": os.environ["HARNESS2_SOURCE"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "DOCKER_HOST": os.environ["DOCKER_HOST"],
        }
    )
    if os.environ.get("CRAVING_CAMPAIGN"):
        environment["CRAVING_CAMPAIGN"] = os.environ["CRAVING_CAMPAIGN"]
    completed = subprocess.run(
        [
            runner.cli_bin,
            "--sandbox",
            "--mode",
            "auto",
            "--model",
            profile["default_model"],
            "--disable-slash-commands",
            "--prompt-interactive",
            read_regular(PACKAGE / "operator-prompt.txt"),
        ],
        cwd=repo,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
