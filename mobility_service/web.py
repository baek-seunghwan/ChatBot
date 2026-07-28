from __future__ import annotations

from pathlib import Path

from .site_header import SITE_HEADER_CSS, site_header


def _page(name: str, *, active: str = "bundle") -> str:
    return (
        Path(__file__)
        .with_name(name)
        .read_text(encoding="utf-8")
        .replace("{{SITE_HEADER_CSS}}", SITE_HEADER_CSS)
        .replace("{{SITE_HEADER}}", site_header(active))
    )


INDEX_HTML = _page("index.html", active="bundle")
ADMIN_HTML = Path(__file__).with_name("admin.html").read_text(encoding="utf-8")
FEATURES_HTML = _page("features.html", active="bundle")
