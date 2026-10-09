"""Server-side validation for invoice submissions to the Invoice Register."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

ALLOWED_CURRENCIES = ("INR", "USD", "EUR", "GBP")

_INVOICE_NUMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/_.]{1,49}$")
_PLAIN_AMOUNT_RE = re.compile(r"^\d{1,12}(\.\d{1,2})?$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

AMOUNT_HINT = "Enter a plain number with up to 2 decimals, e.g. 23500.75 (no currency symbols or thousands separators)."
DATE_HINT = "Use the format YYYY-MM-DD, e.g. 2026-09-18."


@dataclass(frozen=True)
class InvoiceInput:
    company_name: str
    invoice_number: str
    issue_date: date
    amount: Decimal
    currency: str
    due_date: date


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _parse_iso_date(raw: str) -> date | None:
    if not _ISO_DATE_RE.match(raw):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def validate_invoice_form(form: dict[str, object]) -> tuple[InvoiceInput | None, dict[str, str]]:
    """Validate raw form values. Returns (InvoiceInput, {}) or (None, field_errors)."""
    errors: dict[str, str] = {}
    company = _clean(form.get("company_name"))
    number = _clean(form.get("invoice_number"))
    issue_raw = _clean(form.get("issue_date"))
    amount_raw = _clean(form.get("amount"))
    currency = _clean(form.get("currency")).upper()
    due_raw = _clean(form.get("due_date"))

    if not (2 <= len(company) <= 200):
        errors["company_name"] = "Company name is required (2-200 characters)."
    if not _INVOICE_NUMBER_RE.match(number):
        errors["invoice_number"] = "Invoice number is required: 2-50 letters, digits, '-', '/', '_' or '.'."

    issue = _parse_iso_date(issue_raw)
    if issue is None:
        errors["issue_date"] = f"Invalid issue date. {DATE_HINT}"
    due = _parse_iso_date(due_raw)
    if due is None:
        errors["due_date"] = f"Invalid due date. {DATE_HINT}"
    if issue and due and due < issue:
        errors["due_date"] = "Due date cannot be earlier than the issue date."

    amount: Decimal | None = None
    if not _PLAIN_AMOUNT_RE.match(amount_raw):
        errors["amount"] = f"Invalid amount. {AMOUNT_HINT}"
    else:
        try:
            amount = Decimal(amount_raw).quantize(Decimal("0.01"))
        except InvalidOperation:
            errors["amount"] = f"Invalid amount. {AMOUNT_HINT}"
        if amount is not None and amount <= 0:
            errors["amount"] = "Amount must be greater than zero."

    if currency not in ALLOWED_CURRENCIES:
        errors["currency"] = f"Currency must be one of: {', '.join(ALLOWED_CURRENCIES)}."

    if errors:
        return None, errors
    assert issue and due and amount is not None  # for type checkers; guaranteed above
    return InvoiceInput(company, number, issue, amount, currency, due), {}
