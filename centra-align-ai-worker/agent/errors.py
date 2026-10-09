"""Exception types shared across the agent package."""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """An expected, explainable tool failure that is returned to the LLM as data.

    ``error_type`` is a short machine-readable code (e.g. ``field_not_found``),
    ``hint`` tells the model what a sensible recovery looks like.
    """

    def __init__(self, error_type: str, message: str, hint: str = "", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.error_type, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        if self.details:
            out["details"] = self.details
        return out


class LLMError(Exception):
    """The LLM provider could not be reached or rejected the request."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
