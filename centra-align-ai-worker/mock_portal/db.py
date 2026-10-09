"""SQLite persistence for the Invoice Register.

The portal writes through ``insert_invoice``. The agent's verifier only ever
uses ``connect_readonly`` / ``find_invoices`` so it cannot modify records.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mock_portal.validation import InvoiceInput

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name   TEXT NOT NULL COLLATE NOCASE,
    invoice_number TEXT NOT NULL COLLATE NOCASE,
    issue_date     TEXT NOT NULL,          -- ISO 8601 date
    amount         TEXT NOT NULL,          -- exact decimal string, e.g. '148750.00'
    currency       TEXT NOT NULL,
    due_date       TEXT NOT NULL,          -- ISO 8601 date
    status         TEXT NOT NULL DEFAULT 'Recorded',
    created_at     TEXT NOT NULL,
    UNIQUE (company_name, invoice_number)
);
"""


class DuplicateInvoiceError(Exception):
    def __init__(self, existing_id: int | None) -> None:
        super().__init__(f"Invoice already exists (record #{existing_id}).")
        self.existing_id = existing_id


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the database in SQLite read-only mode (writes raise an error)."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Portal database not found at {db_path}")
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def insert_invoice(db_path: Path, invoice: InvoiceInput) -> int:
    """Insert a validated invoice. Raises DuplicateInvoiceError on company+number clash."""
    with closing(_connect(db_path)) as conn:
        try:
            with conn:
                cur = conn.execute(
                    """INSERT INTO invoices
                       (company_name, invoice_number, issue_date, amount, currency, due_date, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        invoice.company_name,
                        invoice.invoice_number,
                        invoice.issue_date.isoformat(),
                        f"{invoice.amount:.2f}",
                        invoice.currency,
                        invoice.due_date.isoformat(),
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            row = conn.execute(
                "SELECT id FROM invoices WHERE company_name = ? AND invoice_number = ?",
                (invoice.company_name, invoice.invoice_number),
            ).fetchone()
            raise DuplicateInvoiceError(row["id"] if row else None) from exc


def mark_paid(db_path: Path, record_id: int) -> bool:
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute("UPDATE invoices SET status = 'Paid' WHERE id = ?", (record_id,))
        return cur.rowcount == 1


def list_invoices(db_path: Path) -> list[dict[str, Any]]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM invoices ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def count_invoices(db_path: Path) -> int:
    with closing(_connect(db_path)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0])


def find_invoices(db_path: Path, company_name: str, invoice_number: str) -> list[dict[str, Any]]:
    """Read-only lookup by company + invoice number (case-insensitive)."""
    with closing(connect_readonly(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM invoices WHERE company_name = ? AND invoice_number = ?",
            (" ".join(company_name.split()), invoice_number.strip()),
        ).fetchall()
    return [dict(r) for r in rows]
