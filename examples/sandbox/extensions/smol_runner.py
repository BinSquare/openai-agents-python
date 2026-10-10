"""Run a SandboxAgent inside a Smol local or cloud microVM.

    uv run --with 'smolmachines[openai-agents]' python examples/sandbox/extensions/smol_runner.py
    uv run --with 'smolmachines[openai-agents]' python examples/sandbox/extensions/smol_runner.py --target cloud

Set OPENAI_API_KEY for model calls. Cloud runs also need SMOL_CLOUD_TOKEN
(or a previous `smol auth login` on the host). The VM receives neither key.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.entries import File
from agents.sandbox.manifest import Manifest

try:
    from smol.openai_sandbox import SmolSandboxClient, SmolSandboxClientOptions
except ImportError as exc:
    raise SystemExit(
        "Install the Smol sandbox provider with: uv run --with 'smolmachines[openai-agents]' python examples/sandbox/extensions/smol_runner.py"
    ) from exc


async def run_example(
    *,
    target: Literal["local", "cloud"] = "local",
    image: str = "alpine:3.20",
    model: str = "gpt-5.4",
) -> str:
    """Ask an agent to inspect a small file in an isolated VM."""
    manifest = Manifest(
        entries={
            "README.md": File(
                content=b"# Release notes\nThe release candidate passed the sandbox checks.\n"
            ),
        }
    )
    agent = SandboxAgent(
        name="Release reviewer",
        model=model,
        instructions="Read README.md using the shell and give a brief summary.",
        capabilities=[Shell()],
    )
    client = SmolSandboxClient(target=target)  # Guest network access stays disabled.
    result = await Runner.run(
        agent,
        "What does README.md say about the release candidate?",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=client,
                manifest=manifest,
                options=SmolSandboxClientOptions(image=image, memory_mb=768),
            )
        ),
    )
    return str(result.final_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("local", "cloud"), default="local")
    parser.add_argument("--image", default="alpine:3.20")
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()
    print(asyncio.run(run_example(target=args.target, image=args.image, model=args.model)))


if __name__ == "__main__":
    main()
