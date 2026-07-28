from __future__ import annotations

from pathlib import Path

INDEX_HTML = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
BUNDLE_HTML = Path(__file__).with_name("bundle.html").read_text(encoding="utf-8")
ADMIN_HTML = Path(__file__).with_name("admin.html").read_text(encoding="utf-8")
FEATURES_HTML = Path(__file__).with_name("features.html").read_text(encoding="utf-8")
