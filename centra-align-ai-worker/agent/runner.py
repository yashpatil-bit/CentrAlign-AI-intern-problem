"""Generic observe -> decide -> act loop.

Nothing here is specific to invoices: the runner sends the task, the working
memory digest and the tool definitions to the LLM, executes whatever
registered tool it asks for, feeds the real result back, and repeats until
``finish_task`` or a stopping condition. The final status is computed from
evidence (verification results, blocked writes), never from the model's words.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.browser import BrowserController
from agent.errors import LLMError
from agent.llm import LLMClient, ToolCall
from agent.memory import TaskStatus, WorkingMemory, summarize_result
from agent.prompts import build_system_prompt
from agent.safety import RunPolicy, redact_secrets
from agent.tools import ToolContext, ToolRegistry, timed_execute
from config import Settings

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]
RECENT_EXCHANGES_IN_FULL = 6
MAX_REPEATS_PER_SIGNATURE = 3       # identical successful calls
MAX_FAILURES_PER_SIGNATURE = 2      # original attempt + one retry
MAX_CONSECUTIVE_BLOCKED = 3
NUDGE = ("You replied without calling a tool. Nothing happens unless you call a tool. If the task is done or "
         "blocked, call finish_task; otherwise call the next tool.")


@dataclass
class _Exchange:
    call: ToolCall
    result: dict[str, Any]
    assistant_text: str = ""


@dataclass
class _Message:
    role: str
    content: str


class LoopGuard:
    """Detects repeated, unproductive actions."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self.consecutive_blocked = 0

    @staticmethod
    def signature(name: str, raw_arguments: str) -> str:
        try:
            canonical = json.dumps(json.loads(raw_arguments or "{}"), sort_keys=True)
        except json.JSONDecodeError:
            canonical = raw_arguments
        return f"{name}:{canonical}"

    def check(self, name: str, raw_arguments: str) -> dict[str, Any] | None:
        if name in {"finish_task", "remember_fact"}:
            return None
        sig = self.signature(name, raw_arguments)
        limit = MAX_REPEATS_PER_SIGNATURE + (1 if name == "browser_observe" else 0)
        if self.failures.get(sig, 0) >= MAX_FAILURES_PER_SIGNATURE:
            reason = "This exact action already failed twice (original + one retry)."
        elif self.calls.get(sig, 0) >= limit:
            reason = f"This exact action was already performed {self.calls[sig]} times."
        else:
            return None
        self.consecutive_blocked += 1
        return {"ok": False, "tool": name, "error": {
            "type": "repeated_action_blocked", "message": reason,
            "hint": "Try a materially different approach, or call finish_task explaining what blocks you."}}

    def record(self, name: str, raw_arguments: str, ok: bool) -> None:
        sig = self.signature(name, raw_arguments)
        self.calls[sig] = self.calls.get(sig, 0) + 1
        if not ok:
            self.failures[sig] = self.failures.get(sig, 0) + 1
        self.consecutive_blocked = 0


@dataclass
class AgentRunner:
    llm: LLMClient
    settings: Settings
    policy: RunPolicy
    on_event: EventCallback | None = None
    invoice_dir: Path | None = None
    db_path: Path | None = None
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    mode: str = "llm"

    def run(self, task: str) -> WorkingMemory:
        memory = WorkingMemory(request=task.strip(), policy=self.policy, mode=self.mode)
        memory.status = TaskStatus.RUNNING
        artifacts = self.settings.artifacts_dir / memory.task_id
        ctx = ToolContext(
            settings=self.settings,
            policy=self.policy,
            memory=memory,
            invoice_dir=Path(self.invoice_dir or self.settings.invoice_dir),
            db_path=Path(self.db_path or self.settings.database_path),
            browser_factory=lambda: BrowserController(
                self.settings.portal_base_url, artifacts, self.settings.headless,
                self.settings.slow_mo_ms, self.settings.browser_timeout_ms),
        )
        self._emit("run_started", task_id=memory.task_id, task=memory.request, llm=self.llm.label,
                   policy={"allow_writes": self.policy.allow_writes, "dry_run": self.policy.dry_run})
        try:
            if not memory.request:
                memory.agent_outcome = "needs_clarification"
                memory.final_summary = "The task is empty. Please describe what you want done."
            else:
                self._loop(ctx)
        except LLMError as exc:
            logger.error("LLM error: %s | %s", exc.user_message, exc.detail)
            memory.fatal_error = exc.user_message
            memory.stop_reason = "llm_unavailable"
        except Exception as exc:  # noqa: BLE001 - report as failure, keep traceback in logs
            logger.exception("Agent run crashed")
            memory.fatal_error = f"Unexpected error: {type(exc).__name__}: {exc}"
            memory.stop_reason = "exception"
        finally:
            self._final_screenshot(ctx)
            ctx.close()
        memory.status = determine_final_status(memory)
        self._save_log(memory, artifacts)
        self._emit("run_finished", status=memory.status, stop_reason=memory.stop_reason,
                   summary=memory.final_summary or memory.fatal_error)
        return memory

    # ------------------------------------------------------------------ loop
    def _loop(self, ctx: ToolContext) -> None:
        memory = ctx.memory
        budget = self.settings.max_tool_calls
        instructions = build_system_prompt(self.settings.portal_base_url, budget)
        history: list[_Exchange | _Message] = []
        guard = LoopGuard()
        nudged = False
        tools = self.registry.definitions()

        for _turn in range(budget * 2 + 10):
            turn = self.llm.next_turn(instructions, self._build_input(memory, history, budget), tools)
            if turn.text:
                memory.log_decision(turn.text)
                self._emit("model_message", step=memory.current_step, text=redact_secrets(turn.text))
            if not turn.tool_calls:
                if not nudged:
                    nudged = True
                    if turn.text:
                        history.append(_Message("assistant", turn.text))
                    history.append(_Message("user", NUDGE))
                    continue
                memory.stop_reason = "model_stopped_without_finish_task"
                memory.final_summary = memory.final_summary or turn.text
                return

            for i, call in enumerate(turn.tool_calls):
                result = self._execute(ctx, guard, call)
                history.append(_Exchange(call, result, turn.text if i == 0 else ""))
                if memory.finished:
                    return
            if guard.consecutive_blocked >= MAX_CONSECUTIVE_BLOCKED:
                memory.stop_reason = "repeated_unproductive_actions"
                return
            if memory.budget_used >= budget:
                memory.stop_reason = "tool_call_limit"
                self._final_report_turn(ctx, instructions, history, budget)
                return
        memory.stop_reason = "max_turns"

    def _execute(self, ctx: ToolContext, guard: LoopGuard, call: ToolCall) -> dict[str, Any]:
        memory = ctx.memory
        memory.current_step += 1
        spec = self.registry.get(call.name)
        try:
            args_for_log = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            args_for_log = {"_raw": call.arguments}
        self._emit("tool_call", step=memory.current_step, tool=call.name, arguments=args_for_log)

        blocked = guard.check(call.name, call.arguments)
        if blocked is None and spec is not None and spec.budgeted and memory.budget_used >= self.settings.max_tool_calls:
            blocked = {"ok": False, "tool": call.name, "error": {
                "type": "tool_call_limit", "message": "The action tool-call budget is exhausted; not executed."}}
        if blocked is not None:
            result, duration = blocked, 0
        else:
            result, duration = timed_execute(self.registry, ctx, call.name, call.arguments)
            guard.record(call.name, call.arguments, bool(result.get("ok")))
            if spec is not None and spec.budgeted:
                memory.budget_used += 1
        memory.record_action(call.name, args_for_log if isinstance(args_for_log, dict) else {}, result, duration)
        self._emit("tool_result", step=memory.current_step, tool=call.name, ok=bool(result.get("ok")),
                   summary=summarize_result(result), result=result, duration_ms=duration)
        return result

    def _final_report_turn(self, ctx: ToolContext, instructions: str, history: list, budget: int) -> None:
        """After the budget is exhausted, allow one finish_task call so the model can summarise."""
        history.append(_Message("user", f"The action budget of {budget} tool calls is exhausted. No more actions "
                                        "are possible. Call finish_task now: state what was accomplished, what "
                                        "remains unfinished and the last known state."))
        turn = self.llm.next_turn(instructions, self._build_input(ctx.memory, history, budget),
                                  self.registry.definitions(only=["finish_task"]))
        for call in turn.tool_calls:
            if call.name == "finish_task":
                self._execute(ctx, LoopGuard(), call)
                break
        if not ctx.memory.final_summary:
            ctx.memory.final_summary = turn.text or "Stopped: tool-call limit reached."

    # ---------------------------------------------------------- context building
    def _build_input(self, memory: WorkingMemory, history: list, budget: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [{"role": "user", "content": f"TASK FROM USER:\n{memory.request}"}]
        exchange_idx = [i for i, h in enumerate(history) if isinstance(h, _Exchange)]
        recent = set(exchange_idx[-RECENT_EXCHANGES_IN_FULL:])
        for i, entry in enumerate(history):
            if isinstance(entry, _Message):
                items.append({"role": entry.role, "content": entry.content})
                continue
            if entry.assistant_text:
                items.append({"role": "assistant", "content": entry.assistant_text})
            items.append({"type": "function_call", "call_id": entry.call.call_id, "name": entry.call.name,
                          "arguments": entry.call.arguments})
            output = entry.result if i in recent else {"ok": entry.result.get("ok"), "compacted": True,
                                                       "summary": summarize_result(entry.result)}
            items.append({"type": "function_call_output", "call_id": entry.call.call_id,
                          "output": json.dumps(output, default=str)})
        items.append({"role": "developer", "content": memory.summary_for_llm(budget)})
        return items

    # ----------------------------------------------------------------- helpers
    def _final_screenshot(self, ctx: ToolContext) -> None:
        if not ctx.browser_started:
            return
        try:
            shot = ctx.browser().screenshot("final_state")
            ctx.memory.screenshots.append(shot["path"])
        except Exception as exc:  # noqa: BLE001 - evidence capture must not mask the run outcome
            logger.warning("Final screenshot failed: %s", exc)

    def _save_log(self, memory: WorkingMemory, artifacts: Path) -> None:
        try:
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "run_log.json").write_text(json.dumps(memory.to_dict(), indent=2, default=str),
                                                     encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write run log: %s", exc)

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"type": event_type, **payload})
        except Exception:  # noqa: BLE001 - UI callback problems must not break the agent
            logger.exception("Event callback failed")


def determine_final_status(memory: WorkingMemory) -> str:
    """Evidence-based final status. The LLM's own claim of success is never sufficient."""
    if memory.fatal_error:
        return TaskStatus.FAILED
    outcome = memory.agent_outcome
    refused = any(b["reason"].startswith("consequential") for b in memory.blocked_writes)
    if memory.verified_after_last_submission:
        if outcome in (None, "completed") and not refused:
            return TaskStatus.COMPLETED_VERIFIED
        # The write is verified, but part of the request (e.g. "and pay it") was not done.
        return TaskStatus.NEEDS_APPROVAL if outcome == "needs_approval" or refused else TaskStatus.PARTIAL
    if outcome == "needs_clarification":
        return TaskStatus.NEEDS_CLARIFICATION
    if memory.policy.dry_run:
        return TaskStatus.DRY_RUN_COMPLETE if not memory.form_submissions else TaskStatus.FAILED
    if outcome == "needs_approval" or any(b["reason"] == "write not authorized" for b in memory.blocked_writes):
        return TaskStatus.NEEDS_APPROVAL
    if memory.form_submissions or memory.verification is not None:
        return TaskStatus.PARTIAL
    if outcome == "completed":
        return TaskStatus.COMPLETED_NO_CHANGES
    if memory.stop_reason == "tool_call_limit" and memory.actions:
        return TaskStatus.PARTIAL
    return TaskStatus.FAILED
