from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and local .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    vault_path: Path = Field(default=PROJECT_ROOT / "knowledge-vault")
    runtime_path: Path = Field(default=PROJECT_ROOT / "runtime")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "hypervault_chunks"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    offline_test_embeddings: bool = False

    strategy_llm_provider: str = "openai"
    strategy_llm_model: str = "gpt-4o-mini"
    strategy_extraction_temperature: float = 0.2
    strategy_max_evidence_items: int = 80
    strategy_llm_max_retries: int = 2
    vision_provider: str = "none"
    vision_model: str | None = None
    export_figure_pages: bool = False
    hyperagent_cli: Path | None = None
    hyperagent_workdir: Path | None = None
    hyperagent_command_template: str | None = None
    hyperagent_timeout_seconds: float = 120.0

    index_batch_size: int = 64
    chunk_target_chars: int = 1600
    chunk_overlap_chars: int = 220
    watch_debounce_seconds: float = 1.5

    api_host: str = "0.0.0.0"
    api_port: int = 8088
    log_level: str = "INFO"

    @field_validator(
        "vault_path",
        "runtime_path",
        "hyperagent_cli",
        "hyperagent_workdir",
        mode="before",
    )
    @classmethod
    def resolve_project_relative_paths(cls, value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(value)
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path.resolve()

    @property
    def sqlite_path(self) -> Path:
        return self.runtime_path / "hypervault.sqlite3"

    @property
    def vault_memory_path(self) -> Path:
        return self.vault_path / "memory"

    @property
    def paper_assets_path(self) -> Path:
        return self.vault_path / "assets" / "papers"

    @property
    def paper_markdown_path(self) -> Path:
        return self.vault_path / "research" / "papers"

    @property
    def paper_strategy_path(self) -> Path:
        return self.vault_path / "summaries" / "paper-strategies"

    @property
    def research_strategy_memory_path(self) -> Path:
        return self.vault_path / "memory" / "research-strategy"

    @property
    def hyperagent_experience_path(self) -> Path:
        return self.vault_path / "research" / "hyperagent-experience"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
