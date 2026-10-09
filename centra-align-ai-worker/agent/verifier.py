"""Independent, read-only verification of an invoice entry.

This is deliberately separate from the LLM: it re-parses the source document,
re-derives which invoice is the latest, reads SQLite in read-only mode and
(optionally) checks the dashboard in the browser. The final task status is
derived from these checks, never from the model's own claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent.browser import BrowserController
from agent.errors import ToolError
from agent.invoice_parser import (
    InvoiceParseError,
    normalize_company,
    parse_amount,
    parse_date,
    parse_invoice_dir,
    parse_invoice_text,
    select_latest_invoice,
)
from agent.safety import safe_invoice_path
from mock_portal import db


@dataclass
class Check:
    name: str
    passed: bool | None  # None = skipped (not performed)
    detail: str


@dataclass(frozen=True)
class ExpectedInvoice:
    company_name: str
    invoice_number: str
    amount: Decimal
    currency: str
    issue_date: date
    due_date: date


def normalize_expected(raw: dict[str, Any]) -> ExpectedInvoice:
    try:
        amount, amount_currency = parse_amount(str(raw["amount"]))
        currency = str(raw["currency"]).strip().upper()
        if amount_currency and amount_currency != currency:
            raise InvoiceParseError(f"Amount currency {amount_currency} differs from currency {currency}.")
        return ExpectedInvoice(
            company_name=" ".join(str(raw["company_name"]).split()),
            invoice_number=str(raw["invoice_number"]).strip(),
            amount=amount,
            currency=currency,
            issue_date=parse_date(str(raw["issue_date"])),
            due_date=parse_date(str(raw["due_date"])),
        )
    except (KeyError, InvoiceParseError) as exc:
        raise ToolError("invalid_expected_values", f"Expected invoice values could not be parsed: {exc}") from exc


def _compare_fields(expected: ExpectedInvoice, company: str, number: str, amount: Decimal, currency: str,
                    issue: date, due: date) -> list[str]:
    mismatches = []
    if normalize_company(company) != normalize_company(expected.company_name):
        mismatches.append(f"company {company!r} != {expected.company_name!r}")
    if number.strip().lower() != expected.invoice_number.lower():
        mismatches.append(f"invoice_number {number!r} != {expected.invoice_number!r}")
    if amount != expected.amount:
        mismatches.append(f"amount {amount} != {expected.amount}")
    if currency.upper() != expected.currency:
        mismatches.append(f"currency {currency} != {expected.currency}")
    if issue != expected.issue_date:
        mismatches.append(f"issue_date {issue} != {expected.issue_date}")
    if due != expected.due_date:
        mismatches.append(f"due_date {due} != {expected.due_date}")
    return mismatches


def verify_invoice(
    *,
    expected_raw: dict[str, Any],
    source_file: str,
    invoice_dir: Path,
    db_path: Path,
    files_read: list[str],
    form_submissions: list[dict[str, Any]],
    run_started_at: str | None = None,
    browser: BrowserController | None = None,
) -> dict[str, Any]:
    expected = normalize_expected(expected_raw)
    checks: list[Check] = []

    # 1-4: document-side checks -------------------------------------------------
    checks.append(Check("source_file_was_read", source_file in files_read,
                        f"'{source_file}' {'was' if source_file in files_read else 'was NOT'} read via read_file."))
    try:
        source = parse_invoice_text(safe_invoice_path(invoice_dir, source_file).read_text(encoding="utf-8"))
        mism = _compare_fields(expected, source.company_name, source.invoice_number, source.amount,
                               source.currency, source.issue_date, source.due_date)
        checks.append(Check("source_document_matches_entered_values", not mism,
                            "All fields match the source document." if not mism else "; ".join(mism)))
    except (ToolError, InvoiceParseError) as exc:
        checks.append(Check("source_document_matches_entered_values", False, f"Source unreadable: {exc}"))

    parsed, _ = parse_invoice_dir(invoice_dir)
    latest = select_latest_invoice(parsed, expected.company_name)
    is_latest = latest["status"] == "found" and latest["selected"] == source_file
    checks.append(Check("selected_invoice_is_latest_by_issue_date", is_latest,
                        f"Independent selection: status={latest['status']}, selected={latest['selected']}, "
                        f"candidates={latest['candidates']}"))

    consistent = expected.amount > 0 and expected.due_date >= expected.issue_date and len(expected.currency) == 3
    checks.append(Check("extracted_fields_internally_consistent", consistent,
                        f"amount>0, due_date>=issue_date, 3-letter currency: {consistent}"))

    # 5: submission went through the browser form ----------------------------
    via_form = [s for s in form_submissions if s.get("path") == "/invoices/new"
                and str(s.get("fields", {}).get("invoice_number", "")).strip().lower() == expected.invoice_number.lower()]
    checks.append(Check("submitted_through_browser_form", bool(via_form),
                        f"{len(via_form)} matching POST /invoices/new submission(s) observed in the browser."))

    # 6-12: database checks (read-only connection) -------------------------
    record: dict[str, Any] | None = None
    try:
        rows = db.find_invoices(db_path, expected.company_name, expected.invoice_number)
    except FileNotFoundError as exc:
        rows = []
        checks.append(Check("db_record_exists", False, str(exc)))
    else:
        checks.append(Check("db_record_exists", bool(rows), f"{len(rows)} matching row(s) in SQLite."))
    checks.append(Check("db_no_duplicates", len(rows) <= 1, f"{len(rows)} row(s) for this company + invoice number."))
    if rows:
        record = rows[0]
        r_amount = Decimal(record["amount"])
        r_issue, r_due = date.fromisoformat(record["issue_date"]), date.fromisoformat(record["due_date"])
        pairs = [
            ("db_company_matches", normalize_company(record["company_name"]) == normalize_company(expected.company_name),
             record["company_name"]),
            ("db_invoice_number_matches", record["invoice_number"].lower() == expected.invoice_number.lower(),
             record["invoice_number"]),
            ("db_amount_and_currency_match", r_amount == expected.amount and record["currency"] == expected.currency,
             f"{record['currency']} {r_amount}"),
            ("db_issue_date_matches", r_issue == expected.issue_date, record["issue_date"]),
            ("db_due_date_matches", r_due == expected.due_date, record["due_date"]),
        ]
        for name, ok, stored in pairs:
            checks.append(Check(name, ok, f"stored={stored}"))

    # 13: dashboard evidence -------------------------------------------------------
    screenshot = None
    if browser is not None and browser.is_started:
        try:
            browser.navigate("/")
            row_texts = browser.table_rows_containing(expected.invoice_number)
            visible = any(expected.invoice_number in t and f"{expected.amount:.2f}" in t for t in row_texts)
            checks.append(Check("visible_on_portal_dashboard", visible,
                                row_texts[0] if row_texts else "No dashboard row contains the invoice number."))
            screenshot = browser.screenshot("verification_dashboard")["path"]
        except ToolError as exc:
            checks.append(Check("visible_on_portal_dashboard", False, exc.message))
    else:
        checks.append(Check("visible_on_portal_dashboard", None, "Skipped: browser not running."))

    performed = [c for c in checks if c.passed is not None]
    passed = [c for c in performed if c.passed]
    verified = bool(record) and len(passed) == len(performed)
    created_during_run = bool(record and run_started_at and record["created_at"] >= run_started_at)
    return {
        "verified": verified,
        "summary": f"{len(passed)}/{len(performed)} checks passed" + ("" if verified else " - NOT verified"),
        "checks": [asdict(c) for c in checks],
        "db_record": record,
        "record_created_during_this_run": created_during_run,
        "dashboard_screenshot": screenshot,
    }
