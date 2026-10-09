"""Shared fixtures. No test here needs an OpenAI API key."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.browser import BrowserController  # noqa: E402
from agent.llm import LLMTurn, ToolCall  # noqa: E402
from agent.memory import WorkingMemory  # noqa: E402
from agent.safety import RunPolicy  # noqa: E402
from agent.tools import ToolContext  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from mock_portal.app import create_app  # noqa: E402

INVOICE_DIR = PROJECT_ROOT / "data" / "invoices"
AMBIGUOUS_DIR = PROJECT_ROOT / "data" / "scenarios" / "ambiguous_latest"


# ------------------------------------------------------------------ helpers
def make_settings(tmp_path: Path, base_url: str = "http://127.0.0.1:8000", **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(openai_api_key=None, portal_base_url=base_url,
                                  database_path=tmp_path / "portal.db", artifacts_dir=tmp_path / "artifacts",
                                  invoice_dir=INVOICE_DIR, headless=True, slow_mo_ms=0, max_tool_calls=15)
    values.update(overrides)
    return replace(get_settings(), **values)


def make_context(settings: Settings, policy: RunPolicy | None = None, invoice_dir: Path = INVOICE_DIR) -> ToolContext:
    policy = policy or RunPolicy()
    memory = WorkingMemory(request="test task", policy=policy)
    return ToolContext(
        settings=settings, policy=policy, memory=memory, invoice_dir=invoice_dir, db_path=settings.database_path,
        browser_factory=lambda: BrowserController(settings.portal_base_url, settings.artifacts_dir / "shots",
                                                  headless=True, timeout_ms=4000),
    )


class ScriptedLLM:
    """TEST DOUBLE ONLY: returns pre-written turns. Not evidence of real LLM autonomy."""

    label = "scripted mock LLM (tests only)"

    def __init__(self, turns: list[LLMTurn | Callable[[list[dict]], LLMTurn]]) -> None:
        self.turns = list(turns)
        self.inputs: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[str]] = []

    def next_turn(self, instructions: str, input_items: list[dict], tools: list[dict]) -> LLMTurn:
        self.inputs.append(input_items)
        self.tools_seen.append([t["name"] for t in tools])
        if not self.turns:
            return LLMTurn(text="(script exhausted)")
        turn = self.turns.pop(0)
        return turn(input_items) if callable(turn) else turn


_counter = 0


def call(_tool: str, **args: Any) -> LLMTurn:
    global _counter
    _counter += 1
    return LLMTurn(tool_calls=[ToolCall(f"call_{_counter}", _tool, json.dumps(args))])


# ------------------------------------------------------------------ fixtures
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def start_portal(tmp_path: Path) -> Iterator[Callable[..., str]]:
    """Factory: start a real uvicorn portal (isolated temp DB) and return its base URL."""
    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    def _start(fault_mode: str = "none", db_path: Path | None = None) -> str:
        port = _free_port()
        app = create_app(db_path=db_path or tmp_path / "portal.db", fault_mode=fault_mode)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("Portal test server did not start")
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield _start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


_CHROMIUM_OK: bool | None = None


def chromium_available() -> bool:
    global _CHROMIUM_OK
    if _CHROMIUM_OK is None:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                p.chromium.launch(headless=True).close()
            _CHROMIUM_OK = True
        except Exception:  # noqa: BLE001 - any failure means the browser tests must be skipped
            _CHROMIUM_OK = False
    return _CHROMIUM_OK


@pytest.fixture
def chromium() -> None:
    """Skip browser integration tests (with a clear reason) when Chromium is unavailable."""
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed: run 'python -m playwright install chromium'")
