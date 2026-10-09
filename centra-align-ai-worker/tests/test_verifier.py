"""Independent verifier: detects matching, missing and incorrect records."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date
from decimal import Decimal

import pytest

from agent.verifier import verify_invoice
from mock_portal import db
from mock_portal.validation import InvoiceInput
from tests.conftest import INVOICE_DIR

LATEST_FILE = "acme_components_invoice_1.txt"
EXPECTED = {"company_name": "Acme Components Pvt Ltd", "invoice_number": "ACM-INV-2026-0587",
            "amount": "148750.00", "currency": "INR", "issue_date": "2026-09-18", "due_date": "2026-10-18"}
SUBMISSION = [{"path": "/invoices/new", "fields": {"invoice_number": "ACM-INV-2026-0587"}, "step": 5}]


def _insert(db_path, **overrides):
    values = dict(company_name="Acme Components Pvt Ltd", invoice_number="ACM-INV-2026-0587",
                  issue_date=date(2026, 9, 18), amount=Decimal("148750.00"), currency="INR",
                  due_date=date(2026, 10, 18))
    values.update(overrides)
    db.init_db(db_path)
    return db.insert_invoice(db_path, InvoiceInput(**values))


def _verify(db_path, expected=EXPECTED, source=LATEST_FILE, files_read=(LATEST_FILE,), submissions=SUBMISSION):
    return verify_invoice(expected_raw=dict(expected), source_file=source, invoice_dir=INVOICE_DIR, db_path=db_path,
                          files_read=list(files_read), form_submissions=list(submissions))


def _check(result, name):
    return next(c for c in result["checks"] if c["name"] == name)


def test_matching_record_is_verified(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path)
    result = _verify(path)
    assert result["verified"] is True, result["checks"]
    assert _check(result, "visible_on_portal_dashboard")["passed"] is None  # skipped without a browser
    assert result["db_record"]["invoice_number"] == "ACM-INV-2026-0587"


def test_equivalent_formats_are_accepted(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path)
    result = _verify(path, {**EXPECTED, "amount": "INR 1,48,750", "issue_date": "18 September 2026"})
    assert result["verified"] is True


def test_missing_record_is_detected(tmp_path):
    path = tmp_path / "portal.db"
    db.init_db(path)
    result = _verify(path)
    assert result["verified"] is False
    assert _check(result, "db_record_exists")["passed"] is False


def test_incorrect_amount_in_database_is_detected(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path, amount=Decimal("148750.01"))
    result = _verify(path)
    assert result["verified"] is False
    assert _check(result, "db_amount_and_currency_match")["passed"] is False


def test_incorrect_due_date_in_database_is_detected(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path, due_date=date(2026, 10, 19))
    result = _verify(path)
    assert _check(result, "db_due_date_matches")["passed"] is False and not result["verified"]


def test_wrong_invoice_selection_is_detected(tmp_path):
    """Entering the OLDER Acme invoice (correctly copied) still fails: it is not the latest."""
    path = tmp_path / "portal.db"
    _insert(path, invoice_number="ACM-INV-2026-0412", issue_date=date(2026, 7, 14), amount=Decimal("112400.00"),
            due_date=date(2026, 8, 13))
    older = {**EXPECTED, "invoice_number": "ACM-INV-2026-0412", "amount": "112400.00",
             "issue_date": "2026-07-14", "due_date": "2026-08-13"}
    result = _verify(path, older, source="acme_components_invoice_2.txt", files_read=["acme_components_invoice_2.txt"],
                     submissions=[{"path": "/invoices/new", "fields": {"invoice_number": "ACM-INV-2026-0412"}}])
    assert _check(result, "selected_invoice_is_latest_by_issue_date")["passed"] is False
    assert result["verified"] is False


def test_values_not_matching_source_document_are_detected(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path, amount=Decimal("99.00"))
    result = _verify(path, {**EXPECTED, "amount": "99.00"})
    assert _check(result, "source_document_matches_entered_values")["passed"] is False


def test_source_not_read_or_not_submitted_via_browser_fails(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path)
    assert _check(_verify(path, files_read=()), "source_file_was_read")["passed"] is False
    no_form = _verify(path, submissions=())
    assert _check(no_form, "submitted_through_browser_form")["passed"] is False and not no_form["verified"]


def test_verifier_is_read_only(tmp_path):
    path = tmp_path / "portal.db"
    _insert(path)
    before = db.list_invoices(path)
    _verify(path)
    assert db.list_invoices(path) == before
    with closing(db.connect_readonly(path)) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM invoices")
