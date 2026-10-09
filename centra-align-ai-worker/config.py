"""Central configuration for the CentrAlign Autonomous Task Worker.

Every module reads settings through ``get_settings()`` so that the model name,
portal URL, database path and limits are defined in exactly one place.
Values come from environment variables (optionally loaded from ``.env``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env from the project root regardless of the current working directory.
# Existing environment variables win, so tests can override values safely.
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PORTAL_URL = "http://127.0.0.1:8000"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    portal_base_url: str
    invoice_dir: Path
    artifacts_dir: Path
    database_path: Path
    max_tool_calls: int
    headless: bool
    slow_mo_ms: int
    browser_timeout_ms: int
    portal_fault_mode: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Read settings from the environment each time (cheap, and test friendly)."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL,
        portal_base_url=(os.getenv("PORTAL_BASE_URL", "").strip() or DEFAULT_PORTAL_URL).rstrip("/"),
        invoice_dir=_env_path("INVOICE_DIR", PROJECT_ROOT / "data" / "invoices"),
        artifacts_dir=_env_path("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"),
        database_path=_env_path("PORTAL_DB_PATH", PROJECT_ROOT / "data" / "portal.db"),
        max_tool_calls=_env_int("MAX_TOOL_CALLS", 15),
        headless=_env_bool("HEADLESS", True),
        slow_mo_ms=_env_int("BROWSER_SLOW_MO_MS", 0),
        browser_timeout_ms=_env_int("BROWSER_TIMEOUT_MS", 5000),
        portal_fault_mode=os.getenv("PORTAL_FAULT_MODE", "none").strip().lower() or "none",
    )
