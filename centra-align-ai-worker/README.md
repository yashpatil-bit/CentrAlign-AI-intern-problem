# CentrAlign AI — Autonomous Task Worker

An LLM-driven agent that takes a task in plain English, decides which tools to use, **reads real files, drives a real Chromium browser, fills and submits a real web form, and independently verifies the result** before reporting success.

> **Scope:** this first version is restricted to the supplied fictional invoice files and the local simulated *Invoice Register* web app. It does not claim to operate arbitrary websites or desktop applications.

---

## 1. Project overview

| Layer | What it does |
|---|---|
| Streamlit UI (`app.py`) | Task input, write authorization, dry-run, live execution log, verification evidence, screenshots |
| Agent runner (`agent/runner.py`) | Generic observe → decide → act loop around an LLM with function calling |
| Tools (`agent/tools.py`) | 8 required tools + `remember_fact` + `finish_task`, all schema-validated |
| Browser (`agent/browser.py`) | Playwright/Chromium controller locked to the portal origin |
| Verifier (`agent/verifier.py`) | Independent, read-only checks: source document, latest-invoice logic, SQLite, dashboard |
| Mock portal (`mock_portal/`) | FastAPI + SQLite "CentrAlign Demo Company — Invoice Register" |

## 2. Problem statement

A back-office employee receives: *"Find the latest invoice from Acme Components, enter it into our internal system, and tell me when it's done."* They would open the invoice folder, compare documents, pick the newest one, re-key the values into an internal web app, check that it saved, and report back. This project builds an AI agent that does that work itself, handles errors, and **proves** completion with evidence instead of just claiming it.

## 3. Features

- Natural-language task → the LLM plans and chooses each next tool from real observations.
- Genuine file discovery and reading (path-traversal-safe).
- Real browser automation with role/label locators (no coordinates).
- Form filling, submission, validation-error recovery, label-mismatch recovery.
- Independent 13-check verification (decimal/date aware), plus dashboard evidence and screenshots.
- Human controls: write-authorization checkbox, dry-run mode, hard blocks on payment/deletion.
- Prompt-injection resistance: documents/web pages are treated as untrusted data.
- Loop detection, retry limits, tool-call budget, graceful stopping with a partial report.
- Full JSON execution trace per run under `artifacts/<task_id>/`.
- Fault-injection modes on the portal to demo recovery.
- 80+ automated tests (no API key needed), including real-browser integration tests.

## 4. Architecture

```mermaid
flowchart LR
    U[User] -->|task + approvals| UI[Streamlit app.py]
    UI --> R[AgentRunner<br/>agent/runner.py]
    R <-->|instructions, task, memory digest,<br/>recent tool results / tool calls| LLM[(OpenAI Responses API<br/>function calling)]
    R --> REG[ToolRegistry<br/>schema validation + safety]
    REG --> FS[list_files / read_file<br/>data/invoices only]
    REG --> BR[BrowserController<br/>Playwright Chromium]
    REG --> V[verify_invoice_record<br/>agent/verifier.py]
    REG --> M[WorkingMemory<br/>agent/memory.py]
    BR -->|HTTP, localhost only| P[FastAPI portal<br/>mock_portal/app.py]
    P --> DB[(SQLite<br/>data/portal.db)]
    V -->|read-only connection| DB
    V -->|re-parse documents| FS
    V -->|dashboard check + screenshot| BR
    R -->|status from evidence| UI
```

## 5. The agent execution loop

`AgentRunner.run(task)` (not specific to invoices):

1. Build the LLM input: the task, recent tool calls with their **real** outputs (older ones compacted to one-line summaries), and a **working-memory digest** (facts, budget, permissions, last verification, recent errors).
2. Call the OpenAI Responses API with the system instructions and the strict tool schemas.
3. If the model returns tool calls → validate each against its schema → execute it in Python → record the result → feed it back on the next turn.
4. If the model replies with text but no tool call → it is **not** treated as success. It gets one nudge; a second text-only reply stops the run (status from evidence).
5. Stop when `finish_task` is called, or on a stopping condition: tool-call budget (default 15), repeated unproductive actions, or an LLM/API error.
6. Always close the browser (`finally`), take a final screenshot, compute the status from evidence, and write `run_log.json`.

## 6. Tool calling and observation

| Tool | Purpose | Notes |
|---|---|---|
| `list_files` | List invoice documents with size and mtime | Only `.txt`/`.md` in the permitted folder |
| `read_file` | Read one document | Bare filenames only; traversal rejected; content marked UNTRUSTED |
| `browser_navigate` | Open a portal path | Origin-locked; returns a page observation (saves a step) |
| `browser_observe` | URL, title, headings, buttons, links, fields (label, value, help, errors), alerts, text | Size-limited |
| `browser_fill` | Fill fields by **exact accessible label** (inputs and selects) | Reports the value now in each field; on failure, returns the available labels |
| `browser_click` | Click a button/link by role + accessible name | Form POSTs need authorization; pay/delete/approve always blocked |
| `browser_screenshot` | Save a PNG under `artifacts/<task_id>/` | Returns the path |
| `verify_invoice_record` | Independent read-only verification | See §9 |
| `remember_fact` | Store a fact in working memory | Free (not budgeted) |
| `finish_task` | End with `completed / needs_clarification / needs_approval / cannot_complete` | `completed` after an unverified submission is rejected once |

Every result is structured: `{"ok": true, "data": ...}` or `{"ok": false, "error": {"type", "message", "hint", "details"}}`. Exceptions never escape the registry; unknown tools return `unknown_tool`. **The model never produces tool results** — Python executes the call and the real output is sent back with the matching `call_id`.

## 7. Working memory

`WorkingMemory` (`agent/memory.py`) holds: task ID, original request, facts (from tools and from `remember_fact`), current step, action history, tool outputs, errors, retries/recoveries, blocked writes (approval status), files read, browser form submissions, the latest verification result, screenshots, and the final status.

The LLM does **not** get the unbounded history: only the last 6 tool exchanges in full, older ones compacted, plus a bounded digest (≤30 facts, 3 most recent errors). API keys are redacted from memory and logs.

## 8. Error handling and retry strategy

| Scenario | Behaviour |
|---|---|
| A. No matching invoice | Agent must not fabricate; `finish_task(cannot_complete)`. |
| B. Two invoices share the latest date | `finish_task(needs_clarification)` with a specific question. Demo folder: `data/scenarios/ambiguous_latest`. |
| C. Browser action fails | Real Playwright error returned as `click_failed`/`fill_failed`/`navigation_failed`; the agent re-observes. |
| D. Field not found | `field_not_found` with `available_labels`; the agent retries once with the closest label (demo: `PORTAL_FAULT_MODE=alt_labels`). |
| E. Form rejects amount/date | Portal returns 422 with field messages and help text; the agent corrects unambiguous values. Resubmitting after an explicit 4xx is allowed. |
| F. Error after submission | The record may be saved: further submits are **blocked** (`verify_before_resubmit`) until the verifier has run. Demo: `PORTAL_FAULT_MODE=error_after_save` (saves, then shows a 504). |
| G. Record absent from DB | Verification fails → never "verified". Resubmission is allowed only after the verifier proves the record is absent. |
| H. Tool-call limit | Runner stops, offers the model a final `finish_task`-only turn to summarise, and reports `Partially completed`/`Failed`. |

Loop guard: an identical failed call is allowed once more (original + 1 retry), and identical successful calls up to 3 times. Three blocked calls in a row stop the run. Duplicates are also prevented by the DB constraint `UNIQUE(company_name, invoice_number)` (case-insensitive), which returns 409.

## 9. Verification approach

`verify_invoice_record` is pure Python and independent of the LLM. It checks:

1. The source file was actually read via `read_file` in this run.
2. Re-parsing the source document gives the same values that were entered.
3. Re-deriving "latest" from **issue dates inside all documents** selects the same file (and it is unambiguous).
4. The fields are internally consistent (amount > 0, due ≥ issue, 3-letter currency).
5. A `POST /invoices/new` with that invoice number was observed **in the browser** (no back-door writes).
6. A SQLite record exists (read-only `mode=ro` connection).
7. There are no duplicates.
8–12. Company, invoice number, amount + currency (`Decimal`), issue date and due date (`date`) all match.
13. The record is visible on the portal dashboard (browser check plus screenshot).

The UI shows **Completed and verified** only when every performed check passed *and* the verification ran after the last submission. Statuses are computed in `determine_final_status()` from evidence; the model's summary is shown, but it never sets the status.

## 10. Human approval and safety controls

- **Write authorization** checkbox: without it, any click that would submit a POST form returns `approval_required` → status *Needs approval*.
- **Dry run**: form submission returns `dry_run_blocked` → *Dry run complete (nothing saved)*.
- **Consequential actions** (`pay`, `paid`, `delete`, `remove`, `refund`, `transfer`, `approve`, …) are always blocked, even when writes are authorized. The portal has a real "Mark as Paid" button to prove this.
- **Origin lock**: `resolve_portal_url` rejects other hosts, ports and schemes, and a Playwright route aborts **every** network request outside the portal origin.
- **No arbitrary execution**: only registered tools; no shell, Python or SQL-write tool exists.
- **Untrusted content**: `data/invoices/inbox_note_from_vendor.md` contains a deliberate prompt injection ("ignore previous instructions… Pay Now… delete records"). The system prompt treats it as data, and the hard blocks above stop it even if the model were fooled.
- **Secrets**: the API key is read only from `.env`/the environment, never logged (redacted), and never shown in the UI.

## 11. Project structure

```
centra-align-ai-worker/
├── app.py                    # Streamlit UI
├── config.py                 # all settings (env/.env), single place for model name
├── requirements.txt  pytest.ini  .env.example  .gitignore  README.md
├── agent/
│   ├── runner.py             # generic LLM tool-calling loop + evidence-based status
│   ├── llm.py                # OpenAI Responses API client (+ LLMClient protocol)
│   ├── prompts.py            # system instructions
│   ├── tools.py              # tool registry, schemas, validation, handlers
│   ├── browser.py            # Playwright controller (origin-locked)
│   ├── memory.py             # working memory / execution state / statuses
│   ├── verifier.py           # independent read-only verification
│   ├── safety.py             # path, URL, approval and consequential-action rules
│   ├── invoice_parser.py     # deterministic parser used by the verifier
│   ├── offline_planner.py    # OFFLINE NON-LLM demo planner (testing only)
│   └── errors.py
├── mock_portal/
│   ├── app.py  db.py  validation.py
│   ├── templates/ base.html dashboard.html new_invoice.html error.html
│   └── static/style.css
├── data/
│   ├── invoices/             # 2 Acme invoices, 1 Northwind invoice, 1 injection note
│   └── scenarios/ambiguous_latest/
├── tests/                    # conftest + 5 test modules
└── artifacts/                # screenshots + run_log.json per run (git-ignored)
```

The Acme files are deliberately named `acme_components_invoice_1.txt` (newest, 18 Sep 2026) and `..._2.txt` (older, 2026-07-14). They also use different date formats. Neither the filename nor the mtime reveals which invoice is newest; only the dates inside do.

## 12. Prerequisites

- Ubuntu 22.04 (other Linux/macOS/Windows also work), Python **3.10+**, VS Code.
- An OpenAI API key for the real agent mode (the tests don't need one).

## 13. Installation (exact Ubuntu commands)

**Step 1 — Project directory and VS Code**
```bash
cd ~/projects/centra-align-ai-worker   # wherever you unpacked/cloned the project
code .
```

**Step 2 — Virtual environment**
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
```
(In VS Code: *Ctrl+Shift+P → Python: Select Interpreter →* `.venv`.)

**Step 3 — Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4 — Chromium for Playwright**
```bash
python -m playwright install chromium
```
If Chromium fails to launch with missing shared libraries (`libnss3`, `libatk-1.0`, …), install the system dependencies:
```bash
sudo .venv/bin/python -m playwright install-deps chromium
```

**Step 5 — API key (safely)**
```bash
cp .env.example .env
chmod 600 .env
code .env        # paste your key after OPENAI_API_KEY= and save
```
Never paste the key into chat, Python files, or commits. `.env` is in `.gitignore`.

## 14. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | – | Required for the real AI agent mode |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Any model that supports Responses API function calling |
| `PORTAL_BASE_URL` | `http://127.0.0.1:8000` | The only origin the browser may use |
| `PORTAL_DB_PATH` | `data/portal.db` | SQLite file (shared by the portal and the read-only verifier) |
| `PORTAL_FAULT_MODE` | `none` | `none`, `alt_labels`, `error_after_save` |
| `INVOICE_DIR` | `data/invoices` | Permitted document folder |
| `ARTIFACTS_DIR` | `artifacts` | Screenshots and run logs |
| `MAX_TOOL_CALLS` | `15` | Action tool-call budget per run |
| `HEADLESS` | `true` | `false` shows the Chromium window |
| `BROWSER_SLOW_MO_MS` | `0` | Slow down browser actions for recordings |
| `BROWSER_TIMEOUT_MS` | `5000` | Playwright action timeout |

## 15. Running the portal and the agent

**Step 6 — Start the portal** (terminal 1, venv active, project root):
```bash
uvicorn mock_portal.app:app --host 127.0.0.1 --port 8000
```

**Step 7 — Check it**: open http://127.0.0.1:8000/ and http://127.0.0.1:8000/invoices/new, and run:
```bash
curl http://127.0.0.1:8000/health
```
Expected: `{"status":"ok","database":"ok","records":0,"fault_mode":"none"}`.

**Step 8 — Start the UI** (terminal 2: *Terminal → New Terminal*, then `source .venv/bin/activate`):
```bash
streamlit run app.py
```
It opens http://localhost:8501. The sidebar should say *Portal online*.

**Step 9 — First task**: keep the mode on **AI agent – OpenAI LLM**, tick **Authorize the agent to save records**, paste:

```
Find the latest invoice from Acme Components Pvt Ltd, extract the invoice number, amount, issue date, and due date, enter it into the internal Invoice Register, verify that the entry was saved correctly, and give me a summary.
```

Click **Run Task**. You should see in the live log:
1. `list_files` → 4 files.
2. `read_file` on the Acme files (the model decides which ones and in what order), with reasoning about the issue dates. It should ignore the injected instructions in the vendor note.
3. Probably `remember_fact` entries such as "latest invoice = acme_components_invoice_1.txt (18 Sep 2026)".
4. `browser_navigate /invoices/new` → an observation listing the six labelled fields.
5. `browser_fill` with converted values (`148750.00`, `2026-09-18`, `2026-10-18`).
6. `browser_click "Save Invoice"` → redirect to `/?created=N`.
7. `verify_invoice_record` → 13/13 checks.
8. `finish_task` → status **Completed and verified**.

The exact sequence varies between runs because the model decides it. That is expected.

**Step 10 — Check the result**
- Portal: refresh http://127.0.0.1:8000/. The new row is highlighted.
- UI: the **Verification** tab lists every check, and the **Screenshot** tab shows the dashboard and final state.
- Files: `artifacts/<task_id>/` contains the PNGs and `run_log.json` (full trace).
- Read-only DB inspection (debugging only — the agent never writes the DB directly):
  ```bash
  python -c "import sqlite3; c=sqlite3.connect('file:data/portal.db?mode=ro', uri=True); [print(r) for r in c.execute('SELECT * FROM invoices')]"
  ```
- To start over: stop the portal (Ctrl+C), then `rm -f data/portal.db`, then start it again.

## 16. Automated tests

```bash
pytest -q
```
No API key is needed. Browser tests start their own portal on a random port with a temporary database, and are **skipped with a clear reason** if Chromium is not installed.

| File | Covers |
|---|---|
| `test_invoice_files.py` | discovery, safe reading, traversal rejection, parsing, latest-by-issue-date (even with a manipulated mtime), ambiguity, not-found, invalid data, no hard-coded answers in agent code |
| `test_safety.py` | URL allow-listing, unknown tools, schema/JSON errors, tool exceptions → structured errors, consequential-action blocks, redaction, strict schemas |
| `test_portal.py` | portal persistence, duplicates, validation, fault modes; **Playwright**: fill + submit + persist, dry-run saves nothing, unapproved writes blocked, missing-label recovery, network-layer blocking of external requests |
| `test_verifier.py` | match, alternative formats, missing record, wrong amount/date, wrong (older) invoice, source mismatch, not submitted via browser, read-only guarantee |
| `test_runner.py` | loop mechanics with a **scripted mock LLM** (results fed back, unknown tools, max tool calls, text-only replies, loop guard, LLM outage, a success claim without verification → *Partially completed*), plus real-browser end-to-end runs: verified run, error-after-save recovery, dry run, needs approval, ambiguity, unknown company |

> The mock-LLM tests verify the **runtime**; they are not evidence of LLM autonomy. Real autonomy is demonstrated by running the app in AI mode with your key.

## 17. Sample task and expected result

Task: as in Step 9. Expected final state:

| Field | Value (from `acme_components_invoice_1.txt`) |
|---|---|
| Company | Acme Components Pvt Ltd |
| Invoice number | ACM-INV-2026-0587 |
| Issue date | 2026-09-18 (written "18 September 2026" in the file) |
| Amount | 148750.00 INR (written "INR 1,48,750.00") |
| Due date | 2026-10-18 |

Status: **Completed and verified**. The verification shows 13/13 checks and a dashboard screenshot.

Other tasks the same tools handle without code changes:
- "Register the latest invoice from Northwind Industrial Supplies LLP." (dates are DD/MM/YYYY in that file)
- "What is the due date of the latest Acme invoice? Don't change anything." → *Completed (no changes made)*
- "Register the latest invoice from Globex Ltd." → stops; no such invoice
- "Register the latest Acme invoice and mark it as paid." → the invoice can be saved and verified, but payment is refused, so the status is *Needs approval* or *Partially completed*, never *Completed and verified*

## 18. Models, APIs, libraries, services

- **OpenAI Responses API** via the official `openai` Python SDK (`client.responses.create` with strict function tools, `parallel_tool_calls=False`, `store=False`). The model is set by `OPENAI_MODEL`.
- Streamlit, FastAPI, Uvicorn, Jinja2, python-multipart, Playwright (Chromium), SQLite (stdlib), python-dotenv, httpx, pytest.
- No other external services. All business data is local and fictional.

## 19. Assumptions

- "Latest" means the most recent issue/invoice date written in the document.
- Invoice documents are plain text or Markdown with `Key: Value` lines (the LLM reads them freely; the deterministic parser used for verification supports common date formats; slash dates are DD/MM/YYYY).
- One portal record per (company, invoice number).
- Write authorization covers saving records in the local portal only, never payments or deletions.

## 20. Known limitations

- Restricted to the supplied invoice folder and the local portal. There is no general web or desktop control.
- The verifier is invoice-specific. New workflows need their own verifier (the runner, browser and tool infrastructure are reusable).
- The deterministic parser only understands `Key: Value` style documents. There is no PDF/OCR support yet.
- LLM behaviour is non-deterministic. Safety and status do not depend on it, but step order and wording vary.
- Single user, single run at a time, synchronous Streamlit execution.
- The real-LLM path depends on your OpenAI account and model access.

## 21. Next steps with more time

- PDF/email ingestion (OCR) and a document-type classifier.
- Pluggable verifiers per workflow (e.g. purchase orders, vendor onboarding) behind a common interface.
- Proper approval queue with preview/diff of the pending write, instead of a pre-run checkbox.
- An evaluation harness: 50+ generated scenarios and fault injections, scored automatically across models.
- Run history in a DB, plus OpenTelemetry tracing.
- Multi-tab/multi-app browser sessions with per-app allow-lists.

## 22. Recording the demo video (≈5 minutes)

Preparation: `rm -f data/portal.db`, start the portal, set `HEADLESS=false` and `BROWSER_SLOW_MO_MS=400` in `.env` (or use the sidebar options), and start Streamlit. Put the browser windows side by side.

1. **Intro (30 s)**: the architecture diagram; "the LLM decides, Python executes and verifies".
2. **Main run (90 s)**: AI mode, tick *Authorize*, paste the task, click Run. Narrate as the log streams: file discovery → reading → reasoning about issue dates (and ignoring the injected note) → Chromium opens the form → fields fill → Save.
3. **Evidence (45 s)**: the Verification tab (13 checks), the stored SQLite row, the screenshot, and the dashboard refreshed in the browser.
4. **Trace (20 s)**: the Actions and Full trace tabs; download `run_log.json`.
5. **Safe stops (60 s)**, pick two:
   - Run again → the portal returns *Duplicate* (409); the verifier still confirms a single record.
   - Untick *Authorize* → *Needs approval*, nothing saved.
   - Choose folder `scenarios/ambiguous_latest` → *Needs clarification* with a question.
   - Task "…from Globex Ltd…" → stops without inventing data.
6. **Recovery (45 s)**: `rm -f data/portal.db`, then restart the portal with `PORTAL_FAULT_MODE=error_after_save uvicorn mock_portal.app:app --host 127.0.0.1 --port 8000` and rerun. The page shows **504 Gateway Timeout** after Save, although the record was saved. The agent should verify instead of resubmitting; if it tries to resubmit, the runtime blocks it with `verify_before_resubmit`. The run ends **Completed and verified**. Alternatively, use `PORTAL_FAULT_MODE=alt_labels` (the form uses "Vendor Legal Name", "Invoice Total", "Payment Due Date"…). The agent either adapts straight from the page observation, or gets `field_not_found` with the list of available labels and retries once. Point out whichever happens in the log.
7. **Close (15 s)**: run `pytest -q` and show the green result.

## 23. What is genuinely autonomous vs. limited

**Autonomous (decided by the LLM at run time):** interpreting the request and the company name, which files to read and in what order, determining the latest invoice from document contents, converting formats, which page to open, which labels to fill, how to recover from errors (relabelled fields, validation messages, error pages), when to verify, and when to stop or ask.

**Deterministic by design (not the LLM):** tool execution, path/URL/approval/consequential-action rules, schema validation, loop and budget limits, verification, and the final status. That is intentional: the model makes decisions, and software enforces boundaries and establishes truth.

**Limited:** everything runs against fictional local data and a simulated portal. The *offline deterministic demo* mode in the sidebar is a hard-coded state machine for testing the pipeline without a key. It is **not** an LLM, and the UI labels it that way.
