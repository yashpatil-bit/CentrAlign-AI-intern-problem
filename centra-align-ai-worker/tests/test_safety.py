"""Safety boundaries: URLs, tool registry, approvals, structured errors."""

from __future__ import annotations

import pytest

from agent.errors import ToolError
from agent.safety import RunPolicy, is_consequential_action, redact_secrets, resolve_portal_url
from agent.tools import TOOL_SPECS, ToolRegistry, ToolSpec, _obj, validate_schema
from tests.conftest import make_context, make_settings

BASE = "http://127.0.0.1:8000"


@pytest.mark.parametrize("target, expected", [
    ("/", f"{BASE}/"),
    ("/invoices/new", f"{BASE}/invoices/new"),
    ("invoices/new", f"{BASE}/invoices/new"),
    (f"{BASE}/health", f"{BASE}/health"),
])
def test_portal_paths_are_allowed(target, expected):
    assert resolve_portal_url(BASE, target) == expected


@pytest.mark.parametrize("target", [
    "https://example.com/", "//evil.example/x", "javascript:alert(1)", "file:///etc/passwd",
    "http://127.0.0.1:9999/", "http://localhost:8000/", "data:text/html,hi", "ftp://127.0.0.1:8000/",
])
def test_navigation_outside_portal_is_rejected(target):
    with pytest.raises(ToolError) as exc:
        resolve_portal_url(BASE, target)
    assert exc.value.error_type == "navigation_not_allowed"


def _no_browser():
    raise AssertionError("browser must not be started for a rejected action")


def test_browser_navigate_rejects_external_url_without_starting_browser(tmp_path):
    ctx = make_context(make_settings(tmp_path))
    ctx.browser_factory = _no_browser
    result = ToolRegistry().execute(ctx, "browser_navigate", {"path": "https://www.google.com"})
    assert result["ok"] is False and result["error"]["type"] == "navigation_not_allowed"


def test_unregistered_tool_cannot_be_invoked(tmp_path):
    ctx = make_context(make_settings(tmp_path))
    for name in ("run_shell", "execute_python", "sql_write", "delete_invoice"):
        result = ToolRegistry().execute(ctx, name, "{}")
        assert result["ok"] is False
        assert result["error"]["type"] == "unknown_tool"
        assert "list_files" in result["error"]["details"]["available_tools"]


def test_invalid_json_and_schema_violations_are_structured_errors(tmp_path):
    ctx = make_context(make_settings(tmp_path))
    reg = ToolRegistry()
    assert reg.execute(ctx, "read_file", "{not json")["error"]["type"] == "invalid_json"
    assert reg.execute(ctx, "read_file", {})["error"]["type"] == "invalid_arguments"
    assert reg.execute(ctx, "read_file", {"filename": "a.txt", "extra": 1})["error"]["type"] == "invalid_arguments"
    assert reg.execute(ctx, "browser_click", {"role": "checkbox", "name": "x"})["error"]["type"] == "invalid_arguments"


def test_tool_exception_becomes_structured_error_not_crash(tmp_path):
    def boom(ctx, args):
        raise RuntimeError("simulated failure")

    reg = ToolRegistry((ToolSpec("explode", "test", _obj({}), boom),))
    result = reg.execute(make_context(make_settings(tmp_path)), "explode", "{}")
    assert result == {"ok": False, "tool": "explode", "error": {
        "type": "tool_exception", "message": "RuntimeError: simulated failure",
        "hint": "Observe the current state before deciding what to do next."}}


@pytest.mark.parametrize("name", ["Mark as Paid record 1", "Pay Now", "Delete", "Remove invoice", "Approve payment"])
def test_consequential_actions_are_blocked_before_touching_browser(tmp_path, name):
    ctx = make_context(make_settings(tmp_path), RunPolicy(allow_writes=True))
    ctx.browser_factory = _no_browser
    result = ToolRegistry().execute(ctx, "browser_click", {"role": "button", "name": name})
    assert result["error"]["type"] == "consequential_action_blocked"
    assert ctx.memory.blocked_writes


def test_save_is_not_flagged_as_consequential():
    assert not is_consequential_action("Save Invoice")
    assert is_consequential_action("Mark as Paid record 3")


def test_run_policy_dry_run_overrides_authorization():
    assert RunPolicy(allow_writes=True, dry_run=False).can_write
    assert not RunPolicy(allow_writes=True, dry_run=True).can_write
    assert not RunPolicy(allow_writes=False).can_write


def test_secrets_are_redacted():
    assert "sk-abc" not in redact_secrets("key=sk-abcdefghijklmnop123 here")
    assert redact_secrets("token XYZ", ("XYZ",)) == "token ***REDACTED***"


def test_tool_definitions_are_strict_and_validate_their_own_schemas():
    for spec in TOOL_SPECS:
        d = spec.definition()
        assert d["strict"] is True and d["parameters"]["additionalProperties"] is False
        assert set(d["parameters"]["required"]) == set(d["parameters"]["properties"])
    fill = next(s for s in TOOL_SPECS if s.name == "browser_fill")
    assert validate_schema({"fields": []}, fill.parameters)  # minItems is still enforced locally
    assert validate_schema({"fields": [{"label": "Amount", "value": "1"}]}, fill.parameters) == []
