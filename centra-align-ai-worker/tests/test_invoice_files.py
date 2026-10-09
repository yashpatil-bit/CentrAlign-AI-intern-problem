"""Invoice discovery, safe reading, parsing and latest-invoice selection."""

from __future__ import annotations

import os
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from agent.invoice_parser import InvoiceParseError, parse_invoice_dir, parse_invoice_text, select_latest_invoice
from agent.tools import ToolRegistry
from tests.conftest import AMBIGUOUS_DIR, INVOICE_DIR, make_context, make_settings

ACME = "Acme Components Pvt Ltd"


def test_list_files_discovers_all_invoice_documents(tmp_path):
    ctx = make_context(make_settings(tmp_path))
    result = ToolRegistry().execute(ctx, "list_files", "{}")
    assert result["ok"] is True
    names = {f["name"] for f in result["data"]["files"]}
    assert {"acme_components_invoice_1.txt", "acme_components_invoice_2.txt",
            "northwind_supplies_invoice.txt", "inbox_note_from_vendor.md"} <= names
    assert all("size_bytes" in f and "modified_utc" in f for f in result["data"]["files"])


def test_read_allowed_file_returns_content_and_untrusted_notice(tmp_path):
    ctx = make_context(make_settings(tmp_path))
    result = ToolRegistry().execute(ctx, "read_file", {"filename": "acme_components_invoice_1.txt"})
    assert result["ok"] is True
    assert "ACM-INV-2026-0587" in result["data"]["content"]
    assert "UNTRUSTED" in result["data"]["notice"]
    assert ctx.memory.files_read == ["acme_components_invoice_1.txt"]


@pytest.mark.parametrize("bad", ["../config.py", "..\\config.py", "/etc/passwd", "C:\\Windows\\win.ini",
                                 "scenarios/acme_invoice_a.txt", "..", "~/.bashrc", "../../.env"])
def test_paths_outside_permitted_directory_are_rejected(tmp_path, bad):
    ctx = make_context(make_settings(tmp_path))
    result = ToolRegistry().execute(ctx, "read_file", {"filename": bad})
    assert result["ok"] is False
    assert result["error"]["type"] in {"path_not_allowed", "invalid_path"}
    assert ctx.memory.files_read == []


def test_non_document_file_types_are_rejected(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "script.py").write_text("print('hi')")
    ctx = make_context(make_settings(tmp_path), invoice_dir=folder)
    result = ToolRegistry().execute(ctx, "read_file", {"filename": "script.py"})
    assert result["error"]["type"] == "file_type_not_allowed"


def test_parser_extracts_fields_from_mixed_date_formats():
    latest = parse_invoice_text((INVOICE_DIR / "acme_components_invoice_1.txt").read_text(encoding="utf-8"))
    assert latest.invoice_number == "ACM-INV-2026-0587"
    assert latest.amount == Decimal("148750.00") and latest.currency == "INR"
    assert latest.issue_date == date(2026, 9, 18) and latest.due_date == date(2026, 10, 18)
    northwind = parse_invoice_text((INVOICE_DIR / "northwind_supplies_invoice.txt").read_text(encoding="utf-8"))
    assert northwind.issue_date == date(2026, 9, 25)  # DD/MM/YYYY


def test_latest_invoice_selected_by_issue_date_not_filename_or_mtime(tmp_path):
    folder = tmp_path / "inv"
    shutil.copytree(INVOICE_DIR, folder)
    # Make the OLDER invoice look newest by filesystem time; selection must ignore this.
    older = folder / "acme_components_invoice_2.txt"
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime + 10_000_000))
    parsed, errors = parse_invoice_dir(folder)
    choice = select_latest_invoice(parsed, ACME)
    assert choice["status"] == "found"
    assert choice["selected"] == "acme_components_invoice_1.txt"
    assert choice["candidates"] == ["acme_components_invoice_1.txt", "acme_components_invoice_2.txt"]
    assert "inbox_note_from_vendor.md" in errors  # the injection note is not an invoice


def test_company_name_matching_is_tolerant_of_punctuation():
    parsed, _ = parse_invoice_dir(INVOICE_DIR)
    assert select_latest_invoice(parsed, "acme components pvt. ltd.")["status"] == "found"


def test_ambiguous_latest_invoice_is_detected():
    parsed, _ = parse_invoice_dir(AMBIGUOUS_DIR)
    choice = select_latest_invoice(parsed, ACME)
    assert choice["status"] == "ambiguous"
    assert set(choice["candidates"]) == {"acme_invoice_a.txt", "acme_invoice_b.txt"}


def test_missing_company_returns_not_found():
    parsed, _ = parse_invoice_dir(INVOICE_DIR)
    assert select_latest_invoice(parsed, "Globex Corporation")["status"] == "not_found"


@pytest.mark.parametrize("text, reason", [
    ("Supplier: X Ltd\nInvoice Number: A-1\nIssue Date: 2026-01-01\nCurrency: INR", "missing"),
    ("Supplier: X Ltd\nInvoice Number: A-1\nIssue Date: 2026-01-01\nDue Date: 2026-02-01\n"
     "Currency: INR\nTotal: INR abc", "amount"),
    ("Supplier: X Ltd\nInvoice Number: A-1\nIssue Date: 2026-03-01\nDue Date: 2026-02-01\n"
     "Currency: INR\nTotal: INR 10.00", "earlier"),
    ("Supplier: X Ltd\nInvoice Number: A-1\nIssue Date: 31/02/2026\nDue Date: 2026-04-01\n"
     "Currency: INR\nTotal: INR 10.00", "date"),
    ("Supplier: X Ltd\nInvoice Number: A-1\nIssue Date: 2026-01-01\nDue Date: 2026-02-01\n"
     "Currency: USD\nTotal: INR 10.00", "mismatch"),
])
def test_invalid_invoice_data_is_rejected(text, reason):
    with pytest.raises(InvoiceParseError):
        parse_invoice_text(text)


def test_invoice_files_contain_no_hardcoded_answers_in_agent_code():
    """The expected invoice values must be discovered from files, not embedded in agent logic."""
    agent_dir = Path(__file__).resolve().parent.parent / "agent"
    for source in agent_dir.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        for value in ("ACM-INV-2026-0587", "148750", "1,48,750"):
            assert value not in text, f"{value} hard-coded in {source.name}"
