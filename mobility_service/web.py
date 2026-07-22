from __future__ import annotations

from pathlib import Path

INDEX_HTML = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
TAXI_HTML = Path(__file__).with_name("taxi.html").read_text(encoding="utf-8")
ADMIN_HTML = Path(__file__).with_name("admin.html").read_text(encoding="utf-8")
FEATURES_HTML = (
    Path(__file__).parent.parent / "pool-feature-diagram.html"
).read_text(encoding="utf-8")
