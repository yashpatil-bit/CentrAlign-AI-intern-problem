"""Working memory and structured execution state for a single agent run."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent.safety import RunPolicy, redact_secrets


class TaskStatus:
    READY = "Ready"
    RUNNING = "Running"
    NEEDS_CLARIFICATION = "Needs clarification"
    NEEDS_APPROVAL = "Needs approval"
    COMPLETED_VERIFIED = "Completed and verified"
    COMPLETED_NO_CHANGES = "Completed (no changes made)"
    DRY_RUN_COMPLETE = "Dry run complete (nothing saved)"
    PARTIAL = "Partially completed"
    FAILED = "Failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ActionRecord:
    step: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    error_type: str | None = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=_now)


@dataclass
class Fact:
    key: str
    value: str
    source: str
    step: int


@dataclass
class WorkingMemory:
    request: str
    policy: RunPolicy
    task_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    started_at: str = field(default_factory=_now)
    status: str = TaskStatus.READY
    current_step: int = 0
    budget_used: int = 0
    facts: dict[str, Fact] = field(default_factory=dict)
    actions: list[ActionRecord] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    retries: list[dict[str, Any]] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    form_submissions: list[dict[str, Any]] = field(default_factory=list)
    blocked_writes: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    verification_step: int | None = None
    screenshots: list[str] = field(default_factory=list)
    finished: bool = False
    finish_rejections: int = 0
    agent_outcome: str | None = None
    final_summary: str = ""
    clarification_question: str = ""
    stop_reason: str = ""
    fatal_error: str = ""
    mode: str = "llm"

    # ---- facts -------------------------------------------------------------
    def add_fact(self, key: str, value: str, source: str) -> None:
        key = " ".join(key.split())[:80]
        self.facts[key] = Fact(key, redact_secrets(str(value))[:400], source, self.current_step)

    # ---- actions -----------------------------------------------------------
    def record_action(self, tool: str, arguments: dict[str, Any], result: dict[str, Any], duration_ms: int) -> None:
        ok = bool(result.get("ok"))
        error = result.get("error") or {}
        self.actions.append(
            ActionRecord(self.current_step, tool, arguments, ok, summarize_result(result), error.get("type"), duration_ms)
        )
        self.tool_outputs.append({"step": self.current_step, "tool": tool, "result": result})
        if not ok:
            self.errors.append({"step": self.current_step, "tool": tool, **error})
            previous = [a for a in self.actions[:-1] if a.tool == tool and a.arguments == arguments and not a.ok]
            if previous:
                self.retries.append({"step": self.current_step, "tool": tool, "arguments": arguments,
                                     "attempt": len(previous) + 1})
        elif any(a.tool == tool and not a.ok for a in self.actions[:-1]):
            self.retries.append({"step": self.current_step, "tool": tool, "arguments": arguments,
                                 "recovered": True})

    def log_decision(self, text: str) -> None:
        self.decisions.append({"step": self.current_step, "text": redact_secrets(text)[:2000], "timestamp": _now()})

    # ---- derived state -------------------------------------------------------
    @property
    def last_submission_step(self) -> int | None:
        return self.form_submissions[-1]["step"] if self.form_submissions else None

    @property
    def verified_after_last_submission(self) -> bool:
        if not self.verification or not self.verification.get("verified"):
            return False
        last = self.last_submission_step
        return last is not None and self.verification_step is not None and self.verification_step >= last

    def unverified_submission_outcome_unknown(self) -> bool:
        """True if the last form POST may have saved data (not a 4xx rejection) and was not verified since."""
        if not self.form_submissions:
            return False
        last = self.form_submissions[-1]
        status = last.get("response_status")
        explicitly_rejected = status is not None and 400 <= status < 500
        verified_since = self.verification_step is not None and self.verification_step >= last["step"]
        return not explicitly_rejected and not verified_since

    def summary_for_llm(self, max_tool_calls: int) -> str:
        """Concise working-memory digest sent to the LLM every turn (bounded size)."""
        lines = [
            "WORKING MEMORY (maintained by the runtime; authoritative):",
            f"- Step {self.current_step}; action tool calls used {self.budget_used}/{max_tool_calls}.",
            f"- Permissions: write_authorized={self.policy.allow_writes}, dry_run={self.policy.dry_run}.",
            f"- Files read so far: {', '.join(self.files_read) or 'none'}.",
            f"- Form submissions this run: {len(self.form_submissions)}.",
        ]
        if self.verification is not None:
            failed = [c["name"] for c in self.verification.get("checks", []) if c.get("passed") is False]
            lines.append(
                f"- Last verification: verified={self.verification.get('verified')}; failed checks: {failed or 'none'}."
            )
        if self.blocked_writes:
            lines.append(f"- Blocked write attempts: {len(self.blocked_writes)} ({self.blocked_writes[-1]['reason']}).")
        if self.facts:
            lines.append("- Facts:")
            for fact in list(self.facts.values())[-30:]:
                lines.append(f"  * {fact.key}: {fact.value}")
        recent_errors = self.errors[-3:]
        if recent_errors:
            lines.append("- Recent errors:")
            for err in recent_errors:
                lines.append(f"  * step {err['step']} {err['tool']}: {err.get('type')} - {str(err.get('message'))[:160]}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = asdict(self.policy)
        return json.loads(redact_secrets(json.dumps(data, default=str)))


def summarize_result(result: dict[str, Any], limit: int = 220) -> str:
    """One-line description of a tool result for logs and compacted history."""
    if not result.get("ok"):
        err = result.get("error") or {}
        return f"ERROR {err.get('type')}: {str(err.get('message'))[:limit]}"
    data = result.get("data")
    if isinstance(data, dict) and "summary" in data:
        return f"OK: {str(data['summary'])[:limit]}"
    text = json.dumps(data, default=str)
    return "OK: " + (text[:limit] + "..." if len(text) > limit else text)
