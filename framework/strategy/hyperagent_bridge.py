from __future__ import annotations

import asyncio
import logging

from framework.config import Settings
from framework.strategy.schemas import HyperAgentSummaryRequest, HyperAgentSummaryResponse

logger = logging.getLogger(__name__)


class HyperAgentBridge:
    """External process bridge; never imports HyperAgent Python modules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def summarize(self, request: HyperAgentSummaryRequest) -> HyperAgentSummaryResponse:
        if not self.settings.hyperagent_cli:
            raise RuntimeError("HYPERAGENT_CLI is not configured")
        if not self.settings.hyperagent_cli.exists():
            raise FileNotFoundError(self.settings.hyperagent_cli)

        command = [
            str(self.settings.hyperagent_cli),
            "research",
            "summarize",
            "--topic",
            request.topic,
        ]
        if request.input_path:
            command.extend(["--input", str(request.input_path)])
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
