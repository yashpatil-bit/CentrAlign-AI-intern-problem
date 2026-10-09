"""Agent loop mechanics, tested with a SCRIPTED MOCK LLM.

These tests prove the runner executes tools, feeds real results back, enforces
limits and derives status from evidence. They are NOT evidence of real LLM
autonomy; that requires running the app with an OpenAI API key.
"""

from __future__ import annotations

import json

from agent.errors import LLMError
from agent.llm import LLMTurn
from agent.memory import TaskStatus, WorkingMemory
from agent.offline_planner import OfflineDemoPlanner
from agent.runner import AgentRunner, determine_final_status
from agent.safety import RunPolicy
from mock_portal import db
from tests.conftest import AMBIGUOUS_DIR, ScriptedLLM, call, make_settings

TASK = ("Find the latest invoice from Acme Components Pvt Ltd, extract the invoice number, amount, issue date, "
        "and due date, enter it into the internal Invoice Register, verify that the entry was saved correctly, "
        "and give me a summary.")


def _runner(tmp_path, llm, policy=RunPolicy(allow_writes=True), base="http://127.0.0.1:8000", **kw):
    return AgentRunner(llm=llm, settings=make_settings(tmp_path, base, **kw), policy=policy)


def _last_output(items):
    return json.loads(next(i for i in reversed(items) if i.get("type") == "function_call_output")["output"])


def test_tool_results_are_fed_back_to_the_model(tmp_path):
    seen = {}

    def inspect(items):
        seen["listing"] = _last_output(items)
        return call("finish_task", outcome="completed", summary="listed", question_for_user="")

    llm = ScriptedLLM([call("list_files"), inspect])
    memory = _runner(tmp_path, llm).run("List the invoice files")
    assert seen["listing"]["ok"] is True and seen["listing"]["data"]["count"] >= 3
    assert memory.status == TaskStatus.COMPLETED_NO_CHANGES  # no claim of verified completion
    assert "WORKING MEMORY" in llm.inputs[-1][-1]["content"]


def test_unregistered_tool_call_returns_error_and_loop_continues(tmp_path):
    seen = {}

    def inspect(items):
        seen["out"] = _last_output(items)
        return call("finish_task", outcome="cannot_complete", summary="no shell", question_for_user="")

    memory = _runner(tmp_path, ScriptedLLM([call("run_shell", cmd="rm -rf /"), inspect])).run("do something")
    assert seen["out"]["error"]["type"] == "unknown_tool"
    assert memory.status == TaskStatus.FAILED


def test_agent_stops_at_max_tool_calls(tmp_path):
    files = ["acme_components_invoice_1.txt", "acme_components_invoice_2.txt"]
    script = [call("list_files")] + [call("read_file", filename=f) for f in files]
    script.append(call("finish_task", outcome="cannot_complete", summary="ran out of budget", question_for_user=""))
    llm = ScriptedLLM(script)
    memory = _runner(tmp_path, llm, max_tool_calls=3).run(TASK)
    assert memory.budget_used == 3
    assert memory.stop_reason == "tool_call_limit"
    assert llm.tools_seen[-1] == ["finish_task"]  # final turn only allows reporting
    assert memory.status == TaskStatus.PARTIAL
    assert memory.final_summary == "ran out of budget"


def test_text_without_tool_call_is_not_treated_as_success(tmp_path):
    llm = ScriptedLLM([LLMTurn(text="All done! The invoice is saved."), LLMTurn(text="Really, it is done.")])
    memory = _runner(tmp_path, llm).run(TASK)
    assert memory.stop_reason == "model_stopped_without_finish_task"
    assert memory.status == TaskStatus.FAILED
    assert "You replied without calling a tool" in json.dumps(llm.inputs[1])


def test_repeated_failing_action_is_blocked(tmp_path):
    bad = call("read_file", filename="missing.txt")
    llm = ScriptedLLM([bad, bad, bad, bad, bad])
    memory = _runner(tmp_path, llm).run(TASK)
    types = [a.error_type for a in memory.actions]
    assert types[:2] == ["file_not_found", "file_not_found"]
    assert "repeated_action_blocked" in types
    assert memory.stop_reason == "repeated_unproductive_actions"


def test_llm_unavailable_produces_failed_status_with_explanation(tmp_path):
    class DownLLM:
        label = "down"

        def next_turn(self, *a):
            raise LLMError("Could not reach the OpenAI API.")

    memory = _runner(tmp_path, DownLLM()).run(TASK)
    assert memory.status == TaskStatus.FAILED and "OpenAI" in memory.fatal_error


def test_verified_write_with_refused_payment_is_not_reported_as_complete():
    memory = WorkingMemory(request="register and pay", policy=RunPolicy(allow_writes=True))
    memory.form_submissions = [{"path": "/invoices/new", "step": 3}]
    memory.verification, memory.verification_step = {"verified": True, "checks": []}, 4
    memory.agent_outcome = "completed"
    assert determine_final_status(memory) == TaskStatus.COMPLETED_VERIFIED
    memory.blocked_writes = [{"step": 5, "target": "Mark as Paid record 1",
                              "reason": "consequential action (pay/delete/approve...)"}]
    assert determine_final_status(memory) == TaskStatus.NEEDS_APPROVAL


# ----------------------------------------------- end-to-end with real browser
def test_scripted_end_to_end_run_is_verified(tmp_path, start_portal, chromium):
    base = start_portal()
    fields = [{"label": "Company Name", "value": "Acme Components Pvt Ltd"},
              {"label": "Invoice Number", "value": "ACM-INV-2026-0587"},
              {"label": "Issue Date", "value": "2026-09-18"}, {"label": "Amount", "value": "148750.00"},
              {"label": "Currency", "value": "INR"}, {"label": "Due Date", "value": "2026-10-18"}]
    expected = {"company_name": "Acme Components Pvt Ltd", "invoice_number": "ACM-INV-2026-0587",
                "amount": "148750.00", "currency": "INR", "issue_date": "2026-09-18", "due_date": "2026-10-18",
                "source_file": "acme_components_invoice_1.txt"}
    llm = ScriptedLLM([
        call("list_files"), call("read_file", filename="acme_components_invoice_1.txt"),
        call("read_file", filename="acme_components_invoice_2.txt"),
        call("browser_navigate", path="/invoices/new"), call("browser_fill", fields=fields),
        call("browser_click", role="button", name="Save Invoice"), call("verify_invoice_record", **expected),
        call("finish_task", outcome="completed", summary="saved", question_for_user=""),
    ])
    memory = _runner(tmp_path, llm, base=base).run(TASK)
    assert memory.status == TaskStatus.COMPLETED_VERIFIED, memory.verification
    assert memory.verification["verified"] is True
    assert all(c["passed"] is True for c in memory.verification["checks"])  # incl. dashboard check
    assert any(s.endswith("final_state.png") for s in memory.screenshots)
    assert (tmp_path / "artifacts" / memory.task_id / "run_log.json").exists()


def test_claimed_success_without_verification_is_partial(tmp_path, start_portal, chromium):
    base = start_portal()
    fields = [{"label": "Company Name", "value": "Acme Components Pvt Ltd"},
              {"label": "Invoice Number", "value": "X-1"}, {"label": "Issue Date", "value": "2026-09-18"},
              {"label": "Amount", "value": "1.00"}, {"label": "Due Date", "value": "2026-10-18"}]
    done = call("finish_task", outcome="completed", summary="Done!", question_for_user="")
    llm = ScriptedLLM([call("browser_navigate", path="/invoices/new"), call("browser_fill", fields=fields),
                       call("browser_click", role="button", name="Save Invoice"), done, done])
    memory = _runner(tmp_path, llm, base=base).run(TASK)
    assert memory.errors[-1]["type"] == "verification_required"  # first claim rejected
    assert memory.status == TaskStatus.PARTIAL                   # never "verified" on the model's word


def test_offline_planner_end_to_end_recovers_from_error_after_save(tmp_path, start_portal, chromium):
    """Scenario F: the portal saves but shows a 504. The run must verify, not resubmit."""
    base = start_portal(fault_mode="error_after_save")
    memory = _runner(tmp_path, OfflineDemoPlanner(), base=base).run(TASK)
    assert memory.status == TaskStatus.COMPLETED_VERIFIED
    assert [s["response_status"] for s in memory.form_submissions] == [504]
    assert db.count_invoices(tmp_path / "portal.db") == 1


def test_offline_planner_dry_run_saves_nothing(tmp_path, start_portal, chromium):
    base = start_portal()
    memory = _runner(tmp_path, OfflineDemoPlanner(), RunPolicy(allow_writes=True, dry_run=True), base=base).run(TASK)
    assert memory.status == TaskStatus.DRY_RUN_COMPLETE
    assert db.count_invoices(tmp_path / "portal.db") == 0


def test_offline_planner_without_approval_needs_approval(tmp_path, start_portal, chromium):
    base = start_portal()
    memory = _runner(tmp_path, OfflineDemoPlanner(), RunPolicy(allow_writes=False), base=base).run(TASK)
    assert memory.status == TaskStatus.NEEDS_APPROVAL
    assert db.count_invoices(tmp_path / "portal.db") == 0


def test_offline_planner_ambiguous_invoices_need_clarification(tmp_path):
    runner = _runner(tmp_path, OfflineDemoPlanner())
    runner.invoice_dir = AMBIGUOUS_DIR
    memory = runner.run(TASK)
    assert memory.status == TaskStatus.NEEDS_CLARIFICATION
    assert "acme_invoice_a.txt" in memory.clarification_question


def test_offline_planner_unknown_company_stops_without_fabricating(tmp_path):
    memory = _runner(tmp_path, OfflineDemoPlanner()).run(TASK.replace("Acme Components Pvt Ltd", "Globex Ltd"))
    assert memory.status == TaskStatus.FAILED
    assert "No invoice" in memory.final_summary and not memory.form_submissions
