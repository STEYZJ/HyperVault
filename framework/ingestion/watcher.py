from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from framework.rag.indexing_service import IndexingService

logger = logging.getLogger(__name__)


class VaultEventHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[Path]) -> None:
        self.loop = loop
        self.queue = queue

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".md":
            return
        self.loop.call_soon_threadsafe(self.queue.put_nowait, path)


class VaultWatcherService:
    def __init__(self, indexing_service: IndexingService, debounce_seconds: float) -> None:
        self.indexing_service = indexing_service
        self.debounce_seconds = debounce_seconds

    async def run(self) -> None:
        await self.indexing_service.initialize()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Path] = asyncio.Queue()
        handler = VaultEventHandler(loop, queue)
        observer = Observer()
        observer.schedule(
            handler,
            str(self.indexing_service.settings.vault_path),
            recursive=True,
        )
        observer.start()
        logger.info("Watching vault %s", self.indexing_service.settings.vault_path)
        try:
            pending: set[Path] = set()
            while True:
                path = await queue.get()
                pending.add(path)
                await asyncio.sleep(self.debounce_seconds)
                while not queue.empty():
                    pending.add(queue.get_nowait())
                await self._process_pending(pending)
                pending.clear()
        finally:
            observer.stop()
            observer.join(timeout=5)

    async def _process_pending(self, paths: set[Path]) -> None:
        for path in sorted(paths):
            try:
                if path.exists():
                    await self.indexing_service.index_file(path)
                else:
                    relative_path = path.relative_to(
                        self.indexing_service.settings.vault_path
                    ).as_posix()
                    await self.indexing_service.delete_file_by_relative_path(relative_path)
            except Exception:
                logger.exception("Watcher failed to process %s", path)

