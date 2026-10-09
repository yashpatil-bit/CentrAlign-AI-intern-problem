"""OFFLINE DETERMINISTIC DEMO MODE - this is NOT an LLM and NOT autonomous.

A hand-written state machine that drives the *same* runner, tools, browser and
verifier as the real agent, so the pipeline can be exercised without an API
key. It only understands the "register the latest invoice from <company>"
pattern. The UI and logs always label runs made with it as non-LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.invoice_parser import InvoiceParseError, ParsedInvoice, parse_invoice_text, select_latest_invoice
from agent.llm import LLMTurn, ToolCall

_COMPANY_RE = re.compile(r"\bfrom\s+(.+?)(?:,|\s+and\s|\s+extract|\.\s|$)", re.IGNORECASE)
_LABEL_KEYWORDS = [
    ("company_name", ("company", "vendor", "supplier")),
    ("invoice_number", ("invoice number", "reference", "invoice no")),
    ("issue_date", ("issue date", "invoice date")),
    ("amount", ("amount", "total")),
    ("currency", ("currency",)),
    ("due_date", ("due",)),
]


class OfflineDemoPlanner:
    label = "Offline deterministic demo planner (NOT an LLM)"

    def __init__(self) -> None:
        self.stage = "list"
        self.pending: list[str] = []
        self.contents: dict[str, str] = {}
        self.selected: str | None = None
        self.invoice: ParsedInvoice | None = None
        self._n = 0

    # ------------------------------------------------------------------ helpers
    def _call(self, name: str, args: dict[str, Any], note: str) -> LLMTurn:
        self._n += 1
        return LLMTurn(text=f"[offline planner, not an LLM] {note}",
                       tool_calls=[ToolCall(f"offline_{self._n}", name, json.dumps(args))])

    def _finish(self, outcome: str, summary: str, question: str = "") -> LLMTurn:
        self.stage = "done"
        return self._call("finish_task", {"outcome": outcome, "summary": summary, "question_for_user": question},
                          f"finishing: {outcome}")

    @staticmethod
    def _last_output(items: list[dict[str, Any]]) -> dict[str, Any]:
        for item in reversed(items):
            if item.get("type") == "function_call_output":
                return json.loads(item["output"])
        return {}

    def _values(self) -> dict[str, str]:
        inv = self.invoice
        assert inv is not None
        return {"company_name": inv.company_name, "invoice_number": inv.invoice_number,
                "issue_date": inv.issue_date.isoformat(), "amount": f"{inv.amount:.2f}",
                "currency": inv.currency, "due_date": inv.due_date.isoformat()}

    # --------------------------------------------------------------- main step
    def next_turn(self, instructions: str, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        last = self._last_output(input_items)
        if [t["name"] for t in tools] == ["finish_task"]:
            return self._finish("cannot_complete", "Offline planner stopped: tool budget exhausted.")
        task = str(input_items[0]["content"])

        if self.stage == "list":
            self.stage = "read"
            return self._call("list_files", {}, "listing invoice files")

        if self.stage == "read":
            data = last.get("data", {})
            if "files" in data:
                self.pending = [f["name"] for f in data["files"]]
            elif "content" in data:
                self.contents[data["filename"]] = data["content"]
            if self.pending:
                name = self.pending.pop(0)
                return self._call("read_file", {"filename": name}, f"reading {name}")
            return self._select(task)

        if self.stage == "fill":
            fields = last.get("data", {}).get("observation", {}).get("fields", [])
            values = self._values()
            to_fill = []
            for f in fields:
                label = f["label"].lower()
                for key, words in _LABEL_KEYWORDS:
                    if any(w in label for w in words):
                        to_fill.append({"label": f["label"], "value": values[key]})
                        break
            self.stage = "submit"
            return self._call("browser_fill", {"fields": to_fill}, "filling the form by observed labels")

        if self.stage == "submit":
            if not last.get("ok"):
                return self._finish("cannot_complete", f"Could not fill the form: {last.get('error')}")
            self.stage = "after_submit"
            return self._call("browser_click", {"role": "button", "name": "Save Invoice"}, "submitting")

        if self.stage == "after_submit":
            err = (last.get("error") or {}).get("type")
            if err == "dry_run_blocked":
                return self._finish("completed", f"Dry run: prepared {self._values()} from {self.selected}; "
                                                 "nothing was saved.")
            if err == "approval_required":
                return self._finish("needs_approval", f"Form filled with {self._values()} but not saved: "
                                                      "write authorization is required.")
            self.stage = "verify"
            return self._call("verify_invoice_record", {**self._values(), "source_file": self.selected},
                              "verifying independently")

        if self.stage == "verify":
            v = last.get("data", {})
            rec = v.get("db_record") or {}
            return self._finish("completed" if v.get("verified") else "cannot_complete",
                                f"Used {self.selected} (latest by issue date). Entered {self._values()}. "
                                f"Record #{rec.get('id')}. Verification: {v.get('summary')}.")

        return self._finish("cannot_complete", "Offline planner reached an unexpected state.")

    def _select(self, task: str) -> LLMTurn:
        m = _COMPANY_RE.search(task)
        if not m:
            return self._finish("needs_clarification", "Which company's invoice should be processed?",
                                "Which company's invoice should I process?")
        company = m.group(1).strip()
        parsed: dict[str, ParsedInvoice] = {}
        for name, text in self.contents.items():
            try:
                parsed[name] = parse_invoice_text(text)
            except InvoiceParseError:
                continue
        choice = select_latest_invoice(parsed, company)
        if choice["status"] == "not_found":
            return self._finish("cannot_complete", f"No invoice from '{company}' exists in the invoice folder.")
        if choice["status"] == "ambiguous":
            return self._finish("needs_clarification",
                                f"Several invoices share the latest issue date: {choice['candidates']}.",
                                f"Which of {choice['candidates']} should I register?")
        self.selected = choice["selected"]
        self.invoice = parsed[self.selected]
        self.stage = "fill"
        return self._call("browser_navigate", {"path": "/invoices/new"}, f"latest is {self.selected}; opening form")
