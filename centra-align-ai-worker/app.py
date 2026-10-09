"""Streamlit UI for the CentrAlign AI Autonomous Task Worker.

Run from the project root (with the portal already running):
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from agent.errors import LLMError
from agent.llm import OpenAIResponsesClient
from agent.memory import TaskStatus
from agent.offline_planner import OfflineDemoPlanner
from agent.runner import AgentRunner
from agent.safety import RunPolicy
from config import PROJECT_ROOT, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEMO_TASK = (
    "Find the latest invoice from Acme Components Pvt Ltd, extract the invoice number, amount, issue date, "
    "and due date, enter it into the internal Invoice Register, verify that the entry was saved correctly, "
    "and give me a summary."
)
MODE_LLM = "AI agent - OpenAI LLM (real autonomy)"
MODE_OFFLINE = "Offline deterministic demo - NOT an LLM"
STATUS_STYLE = {
    TaskStatus.COMPLETED_VERIFIED: ("success", "✅"),
    TaskStatus.COMPLETED_NO_CHANGES: ("info", "ℹ️"),
    TaskStatus.DRY_RUN_COMPLETE: ("info", "🧪"),
    TaskStatus.NEEDS_APPROVAL: ("warning", "✋"),
    TaskStatus.NEEDS_CLARIFICATION: ("warning", "❓"),
    TaskStatus.PARTIAL: ("warning", "⚠️"),
    TaskStatus.FAILED: ("error", "❌"),
    TaskStatus.RUNNING: ("info", "⏳"),
    TaskStatus.READY: ("info", "🟢"),
}

st.set_page_config(page_title="CentrAlign AI - Autonomous Task Worker", page_icon="🤖", layout="wide")
settings = get_settings()


def portal_health(base_url: str) -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=2.0)
        return resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def invoice_folders() -> dict[str, Path]:
    data = PROJECT_ROOT / "data"
    folders = {"invoices (default)": settings.invoice_dir}
    scenarios = data / "scenarios"
    if scenarios.is_dir():
        for p in sorted(scenarios.iterdir()):
            if p.is_dir():
                folders[f"scenarios/{p.name}"] = p
    return folders


def show_status(status: str) -> None:
    kind, icon = STATUS_STYLE.get(status, ("info", "•"))
    getattr(st, kind)(f"**Task status: {status}**", icon=icon)


def describe_event(ev: dict[str, Any]) -> str | None:
    t = ev["type"]
    if t == "run_started":
        return f"🚀 Run `{ev['task_id']}` started with **{ev['llm']}** · permissions {ev['policy']}"
    if t == "model_message":
        return f"💭 _Agent reasoning:_ {ev['text'][:400]}"
    if t == "tool_call":
        return f"🔧 **Step {ev['step']}** → `{ev['tool']}` `{json.dumps(ev['arguments'])[:300]}`"
    if t == "tool_result":
        icon = "✅" if ev["ok"] else "⚠️"
        return f"{icon} {ev['summary'][:400]}  _({ev['duration_ms']} ms)_"
    if t == "run_finished":
        return f"🏁 Finished: **{ev['status']}** ({ev['stop_reason'] or 'finish_task'})"
    return None


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Environment")
    health = portal_health(settings.portal_base_url)
    if health:
        st.success(f"Portal online · {health['records']} record(s) · fault mode: {health['fault_mode']}")
    else:
        st.error(f"Portal not reachable at {settings.portal_base_url}. Start it with:\n\n"
                 "`uvicorn mock_portal.app:app --host 127.0.0.1 --port 8000`")
    st.markdown(f"[Open Invoice Register]({settings.portal_base_url})")

    st.header("Agent")
    mode = st.radio("Mode", [MODE_LLM, MODE_OFFLINE], index=0)
    if mode == MODE_LLM:
        st.caption(f"Model: `{settings.openai_model}` · API key configured: "
                   f"{'yes' if settings.has_api_key else '**no** - add it to .env'}")
    else:
        st.warning("Offline mode is a hard-coded state machine for testing the pipeline. It is NOT an LLM and "
                   "does not demonstrate autonomy.")
    folders = invoice_folders()
    folder_label = st.selectbox("Invoice folder", list(folders))
    max_calls = st.number_input("Max action tool calls", 3, 40, settings.max_tool_calls)
    show_browser = st.checkbox("Show browser window (for demo recording)", value=not settings.headless)
    slow_mo = st.slider("Browser slow-motion (ms)", 0, 1500, settings.slow_mo_ms, step=100)

# ---------------------------------------------------------------------------- main
st.title("🤖 CentrAlign AI — Autonomous Task Worker")
st.caption("Give a task in plain English. The agent decides which tools to use: it reads invoice files, operates "
           "the local Invoice Register in a real Chromium browser, and verifies the result independently. "
           "Restricted to the supplied invoices and the local simulated portal.")

task = st.text_area("Task", value=DEMO_TASK, height=110)
allow_writes = st.checkbox("✍️ Authorize the agent to save records in the local Invoice Register", value=False)
dry_run = st.checkbox("🧪 Dry run (inspect and prepare only, never save)", value=False)
run_clicked = st.button("▶ Run Task", type="primary")

if run_clicked:
    run_settings = replace(settings, max_tool_calls=int(max_calls), headless=not show_browser, slow_mo_ms=int(slow_mo))
    policy = RunPolicy(allow_writes=allow_writes, dry_run=dry_run)
    try:
        llm = OpenAIResponsesClient(settings.openai_api_key, settings.openai_model) if mode == MODE_LLM \
            else OfflineDemoPlanner()
    except LLMError as exc:
        show_status(TaskStatus.FAILED)
        st.error(exc.user_message)
        st.stop()
    show_status(TaskStatus.RUNNING)
    with st.status("Agent running…", expanded=True) as live:
        def on_event(ev: dict[str, Any]) -> None:
            line = describe_event(ev)
            if line:
                live.write(line)

        runner = AgentRunner(llm=llm, settings=run_settings, policy=policy, on_event=on_event,
                             invoice_dir=folders[folder_label], mode="llm" if mode == MODE_LLM else "offline-non-llm")
        memory = runner.run(task)
        ok = memory.status in (TaskStatus.COMPLETED_VERIFIED, TaskStatus.DRY_RUN_COMPLETE,
                               TaskStatus.COMPLETED_NO_CHANGES)
        live.update(label=f"Run finished: {memory.status}", state="complete" if ok else "error", expanded=False)
    st.session_state["last_run"] = memory.to_dict()
    st.rerun()

# ------------------------------------------------------------------------- results
run: dict[str, Any] | None = st.session_state.get("last_run")
if run is None:
    show_status(TaskStatus.READY)
    st.stop()

show_status(run["status"])
if run["mode"] != "llm":
    st.warning("This run used the OFFLINE DETERMINISTIC planner (not an LLM).")
if run["fatal_error"]:
    st.error(run["fatal_error"])
if run["clarification_question"]:
    st.info(f"**Question for you:** {run['clarification_question']}")
st.subheader("Agent summary")
st.write(run["final_summary"] or "_(no summary returned)_")
st.caption(f"Task ID `{run['task_id']}` · stop reason: {run['stop_reason'] or 'finish_task'} · "
           f"action tool calls used: {run['budget_used']} · log: artifacts/{run['task_id']}/run_log.json")

tabs = st.tabs(["Verification", "Actions", "Extracted facts", "Errors & recovery", "Screenshot", "Full trace"])
with tabs[0]:
    v = run["verification"]
    if not v:
        st.info("No independent verification was performed in this run.")
    else:
        (st.success if v["verified"] else st.error)(f"Independent verification: {v['summary']}")
        st.table([{"": {True: "✅", False: "❌", None: "⏭"}[c["passed"]], "check": c["name"], "detail": c["detail"]}
                  for c in v["checks"]])
        if v.get("db_record"):
            st.markdown("**Stored SQLite record (read-only query)**")
            st.json(v["db_record"])
            if not v.get("record_created_during_this_run"):
                st.warning("This record existed before this run (duplicate was prevented).")
with tabs[1]:
    st.table([{"step": a["step"], "tool": a["tool"], "ok": "✅" if a["ok"] else "⚠️",
               "arguments": json.dumps(a["arguments"])[:160], "result": a["summary"][:200], "ms": a["duration_ms"]}
              for a in run["actions"]] or [{"info": "no actions"}])
with tabs[2]:
    st.table([{"fact": f["key"], "value": f["value"], "source": f["source"], "step": f["step"]}
              for f in run["facts"].values()] or [{"info": "no facts recorded"}])
with tabs[3]:
    st.markdown("**Tool / browser errors returned to the agent**")
    st.table([{k: str(e.get(k, "")) for k in ("step", "tool", "type", "message")} for e in run["errors"]]
             or [{"info": "no errors"}])
    st.markdown("**Retries and recoveries**")
    st.table([{k: str(v) for k, v in r.items()} for r in run["retries"]] or [{"info": "none"}])
    st.markdown("**Blocked writes / consequential actions**")
    st.table([{k: str(v) for k, v in b.items()} for b in run["blocked_writes"]] or [{"info": "none"}])
with tabs[4]:
    shots = [p for p in run["screenshots"] if Path(p).exists()]
    if shots:
        for p in reversed(shots[-2:]):
            st.image(p, caption=str(Path(p).relative_to(PROJECT_ROOT)) if Path(p).is_relative_to(PROJECT_ROOT) else p)
    else:
        st.info("No screenshot captured (the browser was not used).")
with tabs[5]:
    st.markdown("**Agent reasoning messages**")
    for d in run["decisions"]:
        st.markdown(f"- step {d['step']}: {d['text']}")
    st.markdown("**Raw tool outputs**")
    st.json(run["tool_outputs"], expanded=False)
    st.download_button("Download run log (JSON)", json.dumps(run, indent=2, default=str),
                       file_name=f"run_{run['task_id']}.json", mime="application/json")
