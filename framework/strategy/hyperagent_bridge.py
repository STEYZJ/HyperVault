from __future__ import annotations

import asyncio
import logging
import shlex

from framework.config import Settings
from framework.strategy.schemas import HyperAgentSummaryRequest, HyperAgentSummaryResponse

logger = logging.getLogger(__name__)


class HyperAgentBridge:
    """External process bridge; never imports HyperAgent Python modules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def summarize(self, request: HyperAgentSummaryRequest) -> HyperAgentSummaryResponse:
        command = self._build_command(request)
        command.extend(request.extra_args)
        logger.info("Calling external HyperAgent runner: %s", command)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.settings.hyperagent_workdir) if self.settings.hyperagent_workdir else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.settings.hyperagent_timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutError("HyperAgent summary call timed out") from exc

        output = stdout.decode("utf-8", errors="replace").strip()
        error = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(f"HyperAgent returned {process.returncode}: {error or output}")
        if not output:
            raise RuntimeError("HyperAgent returned empty summary output")
        return HyperAgentSummaryResponse(
            topic=request.topic,
            content=output,
            return_code=int(process.returncode or 0),
        )

    def _build_command(self, request: HyperAgentSummaryRequest) -> list[str]:
        if self.settings.hyperagent_command_template:
            return shlex.split(
                self.settings.hyperagent_command_template.format(
                    hyperagent_cli=str(self.settings.hyperagent_cli or ""),
                    topic=request.topic,
                    input_path=str(request.input_path or ""),
                )
            )
        if not self.settings.hyperagent_cli:
            raise RuntimeError("HYPERAGENT_CLI is not configured")
        if not self.settings.hyperagent_cli.exists():
            raise FileNotFoundError(self.settings.hyperagent_cli)
        if request.input_path:
            return [
                str(self.settings.hyperagent_cli),
                "research",
                "extract",
                "--paper",
                str(request.input_path),
                "--json",
                "--no-write",
            ]
        return [
            str(self.settings.hyperagent_cli),
            "research",
            "search",
            "--query",
            request.topic,
            "--dimension",
            "research_pattern",
            "--top-k",
            "5",
            "--json",
        ]
