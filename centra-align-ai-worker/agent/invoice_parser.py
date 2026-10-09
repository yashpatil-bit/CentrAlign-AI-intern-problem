"""Deterministic invoice parsing used by the *verifier* (and the offline demo mode).

The LLM agent reads and interprets invoice files itself. This module exists so
that Python can independently check the agent's conclusions: which file is the
latest for a company, and whether the values it entered match the document.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from agent.safety import ALLOWED_INVOICE_SUFFIXES

_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "company_name": ("supplier", "company", "company name", "vendor", "seller"),
    "invoice_number": ("invoice number", "invoice no", "invoice no.", "invoice #"),
    "issue_date": ("issue date", "invoice date", "date of issue"),
    "due_date": ("due date", "payment due date"),
    "currency": ("currency",),
    "total": ("total amount due", "total amount", "amount due", "total"),
}
_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .#]*?)\s*:\s*(.+?)\s*$")
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%d-%b-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y")
_CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "RS": "INR", "RS.": "INR"}


class InvoiceParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedInvoice:
    company_name: str
    invoice_number: str
    issue_date: date
    due_date: date
    amount: Decimal
    currency: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["issue_date"] = self.issue_date.isoformat()
        d["due_date"] = self.due_date.isoformat()
        d["amount"] = f"{self.amount:.2f}"
        return d


def parse_date(value: str) -> date:
    text = " ".join(str(value).replace(",", ", ").split()).replace(" ,", ",")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise InvoiceParseError(f"Unrecognised date: {value!r}")


def parse_amount(value: str) -> tuple[Decimal, str | None]:
    """Parse 'INR 2,35,000.50' / '235000.5' / '₹1,000.5' -> (Decimal('235000.50'), 'INR')."""
    text = str(value).strip()
    currency = None
    m = re.match(r"^([A-Za-z]{3}|Rs\.?|[₹$€£])\s*", text, re.IGNORECASE)
    if m:
        token = m.group(1).upper()
        currency = _CURRENCY_SYMBOLS.get(token, token)
        text = text[m.end():]
    text = text.replace(",", "").replace(" ", "")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        raise InvoiceParseError(f"Unrecognised amount: {value!r}")
    try:
        return Decimal(text).quantize(Decimal("0.01")), currency
    except InvalidOperation as exc:
        raise InvoiceParseError(f"Unrecognised amount: {value!r}") from exc


def normalize_company(name: str) -> str:
    return " ".join(re.sub(r"[.,]", " ", name.lower()).split())


def parse_invoice_text(text: str) -> ParsedInvoice:
    raw: dict[str, str] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        key = " ".join(m.group(1).lower().split())
        for field, aliases in _KEY_ALIASES.items():
            if key in aliases and field not in raw:
                raw[field] = m.group(2).strip()
    missing = [f for f in ("company_name", "invoice_number", "issue_date", "due_date", "total") if f not in raw]
    if missing:
        raise InvoiceParseError(f"Invoice is missing required fields: {', '.join(missing)}")
    amount, amount_currency = parse_amount(raw["total"])
    currency = (raw.get("currency") or amount_currency or "").upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise InvoiceParseError("Invoice currency is missing or invalid.")
    if amount_currency and amount_currency != currency:
        raise InvoiceParseError(f"Currency mismatch: {amount_currency} vs {currency}.")
    issue, due = parse_date(raw["issue_date"]), parse_date(raw["due_date"])
    if amount <= 0:
        raise InvoiceParseError("Invoice amount must be positive.")
    if due < issue:
        raise InvoiceParseError("Due date is earlier than issue date.")
    return ParsedInvoice(" ".join(raw["company_name"].split()), raw["invoice_number"].strip(), issue, due, amount, currency)


def parse_invoice_dir(invoice_dir: Path) -> tuple[dict[str, ParsedInvoice], dict[str, str]]:
    """Parse every permitted file. Returns ({filename: invoice}, {filename: parse_error})."""
    parsed: dict[str, ParsedInvoice] = {}
    errors: dict[str, str] = {}
    for path in sorted(Path(invoice_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_INVOICE_SUFFIXES:
            continue
        try:
            parsed[path.name] = parse_invoice_text(path.read_text(encoding="utf-8"))
        except InvoiceParseError as exc:
            errors[path.name] = str(exc)
    return parsed, errors


def select_latest_invoice(invoices: dict[str, ParsedInvoice], company_name: str) -> dict[str, Any]:
    """Pick the newest invoice for a company by the issue date *inside* the documents."""
    target = normalize_company(company_name)
    matches = {f: inv for f, inv in invoices.items() if normalize_company(inv.company_name) == target}
    if not matches:
        return {"status": "not_found", "selected": None, "candidates": []}
    newest = max(inv.issue_date for inv in matches.values())
    top = sorted(f for f, inv in matches.items() if inv.issue_date == newest)
    candidates = sorted(matches, key=lambda f: matches[f].issue_date, reverse=True)
    if len(top) > 1:
        return {"status": "ambiguous", "selected": None, "candidates": top}
    return {"status": "found", "selected": top[0], "candidates": candidates}
