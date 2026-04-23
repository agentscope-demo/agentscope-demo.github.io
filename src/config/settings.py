from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single typed config object for the entire project.
    Values are loaded from environment variables / .env file.
    Access anywhere via:  from config.settings import get_settings
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    model:          str = Field("gpt-4o-mini", alias="MODEL")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str  = Field("INFO",  alias="LOG_LEVEL")
    log_dir:   Path = Field(Path("logs"), alias="LOG_DIR")

    # ── Run matrix ────────────────────────────────────────────────────────────
    agent_counts: list[int] = Field(
        default=[4, 8, 16, 32],
        alias="AGENT_COUNTS",
    )

    # ── AutoGen ───────────────────────────────────────────────────────────────
    max_turns:        int   = Field(30,    alias="MAX_TURNS")
    request_timeout:  int   = Field(120,   alias="REQUEST_TIMEOUT")
    temperature:      float = Field(0.7,   alias="TEMPERATURE")

    # ── ToM annotator ─────────────────────────────────────────────────────────
    tom_prompt_version:    str = Field("v1",  alias="TOM_PROMPT_VERSION")
    tom_context_window:    int = Field(10,    alias="TOM_CONTEXT_WINDOW")
    tom_model:             str = Field("gpt-4o-mini", alias="TOM_MODEL")

    # ── Paths (derived) ───────────────────────────────────────────────────────
    @property
    def raw_log_dir(self) -> Path:
        return self.log_dir / "raw"

    @property
    def annotated_log_dir(self) -> Path:
        return self.log_dir / "annotated"

    @property
    def scenarios_index_path(self) -> Path:
        return self.log_dir / "scenarios.index.json"

    def ensure_dirs(self) -> None:
        """Create all log directories if they don't exist."""
        self.raw_log_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_log_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings singleton — import and call this everywhere.
    The cache means .env is only read once per process.
    """
    return Settings()