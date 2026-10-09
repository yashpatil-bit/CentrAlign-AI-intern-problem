"""FastAPI app for the simulated internal Invoice Register.

Run from the project root:
    uvicorn mock_portal.app:app --host 127.0.0.1 --port 8000

Fault-injection modes (PORTAL_FAULT_MODE) exist only to demonstrate the
agent's recovery behaviour:
    none             normal behaviour
    alt_labels       the form uses different field labels (agent must re-observe)
    error_after_save the record IS saved, but the browser receives a 504 error page
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Allow `uvicorn mock_portal.app:app` and direct imports from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_settings  # noqa: E402
from mock_portal import db  # noqa: E402
from mock_portal.validation import ALLOWED_CURRENCIES, AMOUNT_HINT, DATE_HINT, validate_invoice_form  # noqa: E402

_HERE = Path(__file__).resolve().parent
FAULT_MODES = {"none", "alt_labels", "error_after_save"}

STANDARD_LABELS = {
    "company_name": "Company Name",
    "invoice_number": "Invoice Number",
    "issue_date": "Issue Date",
    "amount": "Amount",
    "currency": "Currency",
    "due_date": "Due Date",
}
ALTERNATE_LABELS = {
    "company_name": "Vendor Legal Name",
    "invoice_number": "Invoice Reference No.",
    "issue_date": "Invoice Date",
    "amount": "Invoice Total",
    "currency": "Currency",
    "due_date": "Payment Due Date",
}
FIELD_ORDER = ["company_name", "invoice_number", "issue_date", "amount", "currency", "due_date"]
HELP_TEXT = {
    "issue_date": DATE_HINT,
    "due_date": DATE_HINT,
    "amount": AMOUNT_HINT,
    "currency": "Three-letter ISO currency code.",
}


def create_app(db_path: Path | None = None, fault_mode: str | None = None) -> FastAPI:
    settings = get_settings()
    database = Path(db_path) if db_path else settings.database_path
    mode = (fault_mode or settings.portal_fault_mode or "none").lower()
    if mode not in FAULT_MODES:
        raise ValueError(f"Unknown PORTAL_FAULT_MODE {mode!r}; expected one of {sorted(FAULT_MODES)}")
    db.init_db(database)

    app = FastAPI(title="CentrAlign Demo Company - Invoice Register")
    app.state.db_path = database
    app.state.fault_mode = mode
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    templates = Jinja2Templates(directory=_HERE / "templates")
    labels = ALTERNATE_LABELS if mode == "alt_labels" else STANDARD_LABELS

    def render_form(request: Request, values: dict, errors: dict, status_code: int, banner: str | None = None):
        fields = [
            {"name": n, "label": labels[n], "value": values.get(n, ""), "error": errors.get(n), "help": HELP_TEXT.get(n)}
            for n in FIELD_ORDER
        ]
        return templates.TemplateResponse(
            request,
            "new_invoice.html",
            {"fields": fields, "errors": errors, "currencies": ALLOWED_CURRENCIES, "banner": banner},
            status_code=status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, created: int | None = None):
        invoices = db.list_invoices(database)
        return templates.TemplateResponse(
            request, "dashboard.html", {"invoices": invoices, "created": created, "fault_mode": mode}
        )

    @app.get("/invoices/new", response_class=HTMLResponse)
    def new_invoice_form(request: Request):
        return render_form(request, {"currency": "INR"}, {}, 200)

    @app.post("/invoices/new", response_class=HTMLResponse)
    async def create_invoice(request: Request):
        form = await request.form()
        values = {k: str(form.get(k, "")) for k in FIELD_ORDER}
        invoice, errors = validate_invoice_form(values)
        if invoice is None:
            return render_form(request, values, errors, 422, "Please correct the highlighted fields.")
        try:
            record_id = db.insert_invoice(database, invoice)
        except db.DuplicateInvoiceError as exc:
            msg = (
                f"Duplicate invoice: {invoice.company_name} / {invoice.invoice_number} "
                f"already exists as record #{exc.existing_id}. Nothing was saved."
            )
            return render_form(request, values, {"invoice_number": msg}, 409, msg)
        if mode == "error_after_save":
            # Simulates a flaky gateway: the write succeeded but the user sees an error.
            return templates.TemplateResponse(request, "error.html", {}, status_code=504)
        return RedirectResponse(url=f"/?created={record_id}", status_code=303)

    @app.post("/invoices/{record_id}/mark-paid")
    def mark_paid(record_id: int):
        db.mark_paid(database, record_id)
        return RedirectResponse(url="/", status_code=303)

    @app.get("/health")
    def health():
        try:
            records = db.count_invoices(database)
        except Exception as exc:  # report, don't hide, database problems
            return JSONResponse({"status": "error", "database": str(exc)}, status_code=503)
        return {"status": "ok", "database": "ok", "records": records, "fault_mode": mode}

    return app


app = create_app()
