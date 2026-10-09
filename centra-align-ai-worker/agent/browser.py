"""Playwright (Chromium) browser controller restricted to the local portal.

All locators are role/label based; nothing relies on screen coordinates.
Every request leaving the portal origin is aborted at the network layer, so
even a link click cannot take the agent to another site.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, Route, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from agent.errors import ToolError
from agent.safety import is_same_origin, resolve_portal_url

logger = logging.getLogger(__name__)

_OBSERVE_JS = r"""
() => {
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
                      return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const txt = el => (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
  const headings = [...document.querySelectorAll('h1,h2,h3')].filter(vis).map(h => h.tagName + ': ' + txt(h));
  const buttons = [...document.querySelectorAll('button, input[type=submit], input[type=button], [role=button]')]
    .filter(vis).map(b => ({name: (b.getAttribute('aria-label') || txt(b) || b.value || '').trim(),
                            submits_form: !!b.closest('form') && (b.getAttribute('type') || 'submit') === 'submit'}));
  const links = [...document.querySelectorAll('a[href]')].filter(vis)
    .map(a => ({text: txt(a), href: a.getAttribute('href')}));
  const fields = [...document.querySelectorAll('input:not([type=hidden]), select, textarea')].filter(vis).map(el => {
    const label = el.labels && el.labels.length ? txt(el.labels[0]) : (el.getAttribute('aria-label') || '');
    const help = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean)
      .map(id => document.getElementById(id)).filter(Boolean).map(txt).join(' | ');
    const f = {label, tag: el.tagName.toLowerCase(), value: el.value, required: el.required,
               invalid: el.getAttribute('aria-invalid') === 'true'};
    if (el.placeholder) f.placeholder = el.placeholder;
    if (help) f.help_or_error = help;
    if (el.tagName === 'SELECT') f.options = [...el.options].map(o => o.value);
    return f;
  });
  const alerts = [...document.querySelectorAll('[role=alert], [role=status], .field-error')].filter(vis).map(txt);
  const main = document.querySelector('main') || document.body;
  return {headings, buttons, links, fields, alerts, text: txt(main)};
}
"""

_CLICK_TARGET_JS = r"""
el => {
  const form = el.closest('form');
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  const submits = !!form && ((tag === 'button' && (type === '' || type === 'submit')) ||
                             (tag === 'input' && (type === 'submit' || type === 'image')));
  return {tag, submits_form: submits,
          form_method: form ? (form.getAttribute('method') || 'get').toLowerCase() : null,
          form_action: form ? form.action : null,
          href: tag === 'a' ? el.href : null};
}
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "page"


class BrowserController:
    def __init__(
        self,
        base_url: str,
        artifacts_dir: Path,
        headless: bool = True,
        slow_mo_ms: int = 0,
        timeout_ms: int = 5000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.artifacts_dir = Path(artifacts_dir)
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self.timeout_ms = timeout_ms
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._shot_counter = 0
        self.form_submissions: list[dict[str, Any]] = []
        self.blocked_requests: list[str] = []
        self.last_document_status: int | None = None

    # ---- lifecycle -----------------------------------------------------------
    def start(self) -> "BrowserController":
        if self._page is not None:
            return self
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch(headless=self.headless, slow_mo=self.slow_mo_ms)
            self._context = self._browser.new_context(viewport={"width": 1280, "height": 900})
            self._context.route("**/*", self._guard_route)
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)
            self._page.on("request", self._on_request)
            self._page.on("response", self._on_response)
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        for name, closer in (("context", self._context), ("browser", self._browser)):
            if closer is not None:
                try:
                    closer.close()
                except PlaywrightError as exc:
                    logger.warning("Error closing browser %s: %s", name, exc)
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception as exc:  # noqa: BLE001 - log and continue shutting down
                logger.warning("Error stopping Playwright: %s", exc)
        self._pw = self._browser = self._context = self._page = None

    def __enter__(self) -> "BrowserController":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_started(self) -> bool:
        return self._page is not None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise ToolError("browser_not_started", "The browser is not running.")
        return self._page

    # ---- network hooks --------------------------------------------------------
    def _guard_route(self, route: Route) -> None:
        url = route.request.url
        if is_same_origin(self.base_url, url) or url.startswith(("data:", "about:")):
            route.continue_()
        else:
            self.blocked_requests.append(url)
            route.abort("blockedbyclient")

    def _on_request(self, request: Any) -> None:
        if request.method == "POST" and request.is_navigation_request():
            fields = {k: v[0] for k, v in parse_qs(request.post_data or "").items()}
            self.form_submissions.append({"url": request.url, "path": urlparse(request.url).path,
                                          "fields": fields, "time": time.time()})

    def _on_response(self, response: Any) -> None:
        if response.request.is_navigation_request():
            self.last_document_status = response.status

    # ---- actions ---------------------------------------------------------------
    def navigate(self, target: str) -> dict[str, Any]:
        url = resolve_portal_url(self.base_url, target)
        try:
            response = self.page.goto(url, wait_until="load")
        except PlaywrightError as exc:
            raise ToolError("navigation_failed", f"Could not load {url}: {str(exc).splitlines()[0]}",
                            hint="Check that the portal is running (GET /health).") from exc
        return {"url": self.page.url, "title": self.page.title(),
                "http_status": response.status if response else None}

    def observe(self, max_text_chars: int = 2500) -> dict[str, Any]:
        snap = self.page.evaluate(_OBSERVE_JS)
        text = snap.pop("text", "")
        return {
            "url": self.page.url,
            "title": self.page.title(),
            "http_status": self.last_document_status,
            **{k: v[:25] if isinstance(v, list) else v for k, v in snap.items()},
            "page_text": text[:max_text_chars] + (" ...[truncated]" if len(text) > max_text_chars else ""),
        }

    def available_labels(self) -> list[str]:
        return [f["label"] for f in self.page.evaluate(_OBSERVE_JS)["fields"] if f["label"]]

    def fill(self, label: str, value: str) -> dict[str, Any]:
        locator = self.page.get_by_label(label, exact=True)
        count = locator.count()
        if count == 0:
            raise ToolError("field_not_found", f"No form field labelled exactly '{label}' on {self.page.url}.",
                            hint="Use one of available_labels (the form may use different wording).",
                            details={"available_labels": self.available_labels()})
        if count > 1:
            raise ToolError("ambiguous_field", f"{count} elements are labelled '{label}'.")
        element = locator.first
        tag = element.evaluate("e => e.tagName.toLowerCase()")
        try:
            if tag == "select":
                options = element.evaluate("e => [...e.options].map(o => o.value)")
                match = next((o for o in options if o.lower() == value.strip().lower()), None)
                if match is None:
                    raise ToolError("invalid_option", f"'{value}' is not an option for '{label}'.",
                                    details={"options": options})
                element.select_option(value=match)
            else:
                element.fill(value)
            actual = element.input_value()
        except PlaywrightError as exc:
            raise ToolError("fill_failed", f"Could not fill '{label}': {str(exc).splitlines()[0]}") from exc
        return {"label": label, "value_in_field": actual, "matches_requested": actual.strip() == value.strip()}

    def inspect_click_target(self, role: str, name: str) -> dict[str, Any]:
        locator = self.page.get_by_role(role, name=name, exact=True)  # type: ignore[arg-type]
        count = locator.count()
        if count == 0:
            snap = self.page.evaluate(_OBSERVE_JS)
            raise ToolError("element_not_found", f"No visible {role} named exactly '{name}'.",
                            hint="Call browser_observe and use an exact button/link name.",
                            details={"buttons": [b["name"] for b in snap["buttons"]][:15],
                                     "links": [a["text"] for a in snap["links"]][:15]})
        if count > 1:
            raise ToolError("ambiguous_element", f"{count} {role}s are named '{name}'.")
        info = locator.first.evaluate(_CLICK_TARGET_JS)
        info["is_write"] = bool(info["submits_form"] and info["form_method"] == "post")
        return info

    def click(self, role: str, name: str) -> dict[str, Any]:
        url_before = self.page.url
        submissions_before = len(self.form_submissions)
        self.last_document_status = None
        locator = self.page.get_by_role(role, name=name, exact=True)  # type: ignore[arg-type]
        try:
            locator.first.click()
            self.page.wait_for_load_state("load")
        except PlaywrightError as exc:
            raise ToolError("click_failed", f"Clicking '{name}' failed: {str(exc).splitlines()[0]}",
                            hint="Observe the page; do not resubmit a form before verifying.") from exc
        snap = self.page.evaluate(_OBSERVE_JS)
        return {
            "clicked": name,
            "url_before": url_before,
            "url_after": self.page.url,
            "title": self.page.title(),
            "http_status": self.last_document_status,
            "form_submitted": len(self.form_submissions) > submissions_before,
            "headings": snap["headings"][:6],
            "alerts": snap["alerts"][:10],
        }

    def screenshot(self, label: str) -> dict[str, Any]:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._shot_counter += 1
        path = self.artifacts_dir / f"{self._shot_counter:02d}_{_slug(label)}.png"
        self.page.screenshot(path=str(path), full_page=True)
        return {"path": str(path), "url": self.page.url}

    def table_rows_containing(self, text: str) -> list[str]:
        rows = self.page.locator("tr").filter(has_text=text)
        return [" ".join(t.split()) for t in rows.all_inner_texts()]
