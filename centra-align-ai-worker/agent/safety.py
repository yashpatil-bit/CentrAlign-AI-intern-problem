"""Deterministic safety rules. The LLM cannot override anything in this module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from agent.errors import ToolError

ALLOWED_INVOICE_SUFFIXES = {".txt", ".md"}

# Button/link names that represent consequential actions the agent must never take.
CONSEQUENTIAL_ACTION_RE = re.compile(
    r"\b(pay|paid|payment|delete|remove|void|refund|transfer|approve|wire)\b", re.IGNORECASE
)

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


@dataclass(frozen=True)
class RunPolicy:
    """Permissions granted by the human for one run."""

    allow_writes: bool = False
    dry_run: bool = False

    @property
    def can_write(self) -> bool:
        return self.allow_writes and not self.dry_run


def safe_invoice_path(invoice_dir: Path, filename: str) -> Path:
    """Resolve ``filename`` inside ``invoice_dir`` or raise ToolError.

    Only bare file names are accepted: no separators, no '..', no absolute paths.
    """
    name = (filename or "").strip()
    if not name:
        raise ToolError("invalid_path", "A filename is required.")
    if name in {".", ".."} or "/" in name or "\\" in name or ":" in name or name.startswith("~"):
        raise ToolError(
            "path_not_allowed",
            f"'{filename}' is not a plain file name inside the permitted invoice directory.",
            hint="Use a file name exactly as returned by list_files.",
        )
    base = Path(invoice_dir).resolve()
    candidate = (base / name).resolve()
    if candidate.parent != base:
        raise ToolError("path_not_allowed", f"'{filename}' resolves outside the permitted invoice directory.")
    if candidate.suffix.lower() not in ALLOWED_INVOICE_SUFFIXES:
        raise ToolError("file_type_not_allowed", f"Only {sorted(ALLOWED_INVOICE_SUFFIXES)} files may be read.")
    if not candidate.is_file():
        raise ToolError("file_not_found", f"No file named '{name}' in the invoice directory.",
                        hint="Call list_files to see the available file names.")
    return candidate


def _origin(url: str) -> tuple[str, str, int | None]:
    p = urlparse(url)
    port = p.port or {"http": 80, "https": 443}.get(p.scheme)
    return p.scheme, (p.hostname or "").lower(), port


def resolve_portal_url(base_url: str, target: str) -> str:
    """Turn a path or URL into an absolute URL on the portal origin, or raise ToolError."""
    raw = (target or "").strip()
    if not raw:
        raise ToolError("invalid_url", "A path such as '/' or '/invoices/new' is required.")
    if raw.startswith("//"):
        raise ToolError("navigation_not_allowed", f"Protocol-relative URL '{raw}' is not allowed.")
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise ToolError("navigation_not_allowed", f"URL scheme '{parsed.scheme}' is not allowed.")
    absolute = urljoin(base_url.rstrip("/") + "/", raw.lstrip("/")) if not parsed.scheme else raw
    if _origin(absolute) != _origin(base_url):
        raise ToolError(
            "navigation_not_allowed",
            f"Navigation to '{raw}' is outside the permitted portal {base_url}.",
            hint="Only paths on the local Invoice Register are allowed, e.g. '/' or '/invoices/new'.",
        )
    return absolute


def is_same_origin(base_url: str, url: str) -> bool:
    return _origin(base_url) == _origin(url)


def is_consequential_action(name: str) -> bool:
    return bool(CONSEQUENTIAL_ACTION_RE.search(name or ""))


def redact_secrets(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    out = _SECRET_RE.sub("sk-***REDACTED***", text)
    for secret in extra_secrets:
        if secret:
            out = out.replace(secret, "***REDACTED***")
    return out
