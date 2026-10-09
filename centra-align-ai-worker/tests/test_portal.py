"""The simulated portal, plus Playwright browser integration tests against it."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent.safety import RunPolicy
from agent.tools import ToolRegistry
from mock_portal import db
from mock_portal.app import create_app
from tests.conftest import make_context, make_settings

VALID = {"company_name": "Acme Components Pvt Ltd", "invoice_number": "TEST-001", "issue_date": "2026-09-18",
         "amount": "1500.50", "currency": "INR", "due_date": "2026-10-18"}
FORM = [{"label": "Company Name", "value": "Acme Components Pvt Ltd"},
        {"label": "Invoice Number", "value": "TEST-777"},
        {"label": "Issue Date", "value": "2026-09-18"},
        {"label": "Amount", "value": "2500.00"},
        {"label": "Currency", "value": "INR"},
        {"label": "Due Date", "value": "2026-10-18"}]


def client(tmp_path, fault_mode="none"):
    return TestClient(create_app(db_path=tmp_path / "portal.db", fault_mode=fault_mode))


# ---------------------------------------------------------------- HTTP level
def test_health_and_pages(tmp_path):
    c = client(tmp_path)
    assert c.get("/health").json()["status"] == "ok"
    page = c.get("/").text
    assert "CentrAlign Demo Company" in page and "Invoice Register" in page
    form = c.get("/invoices/new").text
    for label in ("Company Name", "Invoice Number", "Issue Date", "Amount", "Currency", "Due Date", "Save Invoice"):
        assert label in form


def test_form_submission_creates_and_persists_record(tmp_path):
    c = client(tmp_path)
    r = c.post("/invoices/new", data=VALID, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?created=1"
    rows = db.list_invoices(tmp_path / "portal.db")
    assert len(rows) == 1 and rows[0]["amount"] == "1500.50" and rows[0]["issue_date"] == "2026-09-18"
    assert "TEST-001" in c.get("/").text


def test_duplicate_invoice_is_rejected_without_creating_a_second_row(tmp_path):
    c = client(tmp_path)
    c.post("/invoices/new", data=VALID)
    dup = c.post("/invoices/new", data={**VALID, "invoice_number": "test-001", "amount": "9.99"})
    assert dup.status_code == 409 and "Duplicate invoice" in dup.text
    assert db.count_invoices(tmp_path / "portal.db") == 1


def test_invalid_amount_and_dates_are_rejected_with_messages(tmp_path):
    c = client(tmp_path)
    r = c.post("/invoices/new", data={**VALID, "amount": "INR 1,500.50", "issue_date": "18/09/2026"})
    assert r.status_code == 422
    assert "Invalid amount" in r.text and "Invalid issue date" in r.text
    r2 = c.post("/invoices/new", data={**VALID, "due_date": "2026-01-01"})
    assert r2.status_code == 422 and "earlier than the issue date" in r2.text
    assert db.count_invoices(tmp_path / "portal.db") == 0


def test_fault_mode_alt_labels_changes_form_labels(tmp_path):
    form = client(tmp_path, "alt_labels").get("/invoices/new").text
    assert "Invoice Total" in form and "Payment Due Date" in form and ">Amount<" not in form


def test_fault_mode_error_after_save_saves_but_reports_error(tmp_path):
    r = client(tmp_path, "error_after_save").post("/invoices/new", data=VALID)
    assert r.status_code == 504
    assert db.count_invoices(tmp_path / "portal.db") == 1


# ------------------------------------------------------- browser integration
def test_browser_fills_and_submits_form_and_record_persists(tmp_path, start_portal, chromium):
    base = start_portal()
    ctx = make_context(make_settings(tmp_path, base), RunPolicy(allow_writes=True))
    reg = ToolRegistry()
    try:
        nav = reg.execute(ctx, "browser_navigate", {"path": "/invoices/new"})
        assert nav["ok"], nav
        labels = [f["label"] for f in nav["data"]["observation"]["fields"]]
        assert labels == ["Company Name", "Invoice Number", "Issue Date", "Amount", "Currency", "Due Date"]
        fill = reg.execute(ctx, "browser_fill", {"fields": FORM})
        assert fill["ok"], fill
        click = reg.execute(ctx, "browser_click", {"role": "button", "name": "Save Invoice"})
        assert click["ok"], click
        assert click["data"]["form_submitted"] is True and "created=1" in click["data"]["url_after"]
        shot = reg.execute(ctx, "browser_screenshot", {"label": "after save"})
        assert shot["ok"] and shot["data"]["path"].endswith(".png")
    finally:
        ctx.close()
    rows = db.list_invoices(tmp_path / "portal.db")
    assert [r["invoice_number"] for r in rows] == ["TEST-777"]
    assert ctx.memory.form_submissions[0]["path"] == "/invoices/new"


def test_dry_run_does_not_save(tmp_path, start_portal, chromium):
    base = start_portal()
    ctx = make_context(make_settings(tmp_path, base), RunPolicy(allow_writes=True, dry_run=True))
    reg = ToolRegistry()
    try:
        reg.execute(ctx, "browser_navigate", {"path": "/invoices/new"})
        assert reg.execute(ctx, "browser_fill", {"fields": FORM})["ok"]
        click = reg.execute(ctx, "browser_click", {"role": "button", "name": "Save Invoice"})
    finally:
        ctx.close()
    assert click["error"]["type"] == "dry_run_blocked"
    assert db.count_invoices(tmp_path / "portal.db") == 0


def test_unapproved_write_is_blocked(tmp_path, start_portal, chromium):
    base = start_portal()
    ctx = make_context(make_settings(tmp_path, base), RunPolicy(allow_writes=False))
    reg = ToolRegistry()
    try:
        reg.execute(ctx, "browser_navigate", {"path": "/invoices/new"})
        reg.execute(ctx, "browser_fill", {"fields": FORM})
        click = reg.execute(ctx, "browser_click", {"role": "button", "name": "Save Invoice"})
        # Non-writing navigation is still permitted.
        assert reg.execute(ctx, "browser_click", {"role": "link", "name": "Dashboard"})["ok"]
    finally:
        ctx.close()
    assert click["error"]["type"] == "approval_required"
    assert db.count_invoices(tmp_path / "portal.db") == 0


def test_missing_field_label_returns_structured_error_with_alternatives(tmp_path, start_portal, chromium):
    base = start_portal(fault_mode="alt_labels")
    ctx = make_context(make_settings(tmp_path, base), RunPolicy(allow_writes=True))
    reg = ToolRegistry()
    try:
        reg.execute(ctx, "browser_navigate", {"path": "/invoices/new"})
        result = reg.execute(ctx, "browser_fill", {"fields": [{"label": "Amount", "value": "10.00"}]})
        retry = reg.execute(ctx, "browser_fill", {"fields": [{"label": "Invoice Total", "value": "10.00"}]})
    finally:
        ctx.close()
    assert result["ok"] is False and result["error"]["type"] == "field_not_found"
    assert "Invoice Total" in result["error"]["details"]["available_labels"]
    assert retry["ok"] is True


def test_external_requests_are_blocked_at_network_layer(tmp_path, start_portal, chromium):
    base = start_portal()
    ctx = make_context(make_settings(tmp_path, base))
    try:
        browser = ctx.browser()
        browser.navigate("/")
        outcome = browser.page.evaluate(
            "() => fetch('https://example.com/').then(() => 'reached').catch(() => 'blocked')")
    finally:
        ctx.close()
    assert outcome == "blocked"
    assert any("example.com" in u for u in browser.blocked_requests)
