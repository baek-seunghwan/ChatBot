from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx


VISIBLE_TAGS = {"h1", "h2", "h3", "p", "li", "summary"}
IGNORED_TAGS = {"script", "style", "svg", "template", "noscript"}


class VisibleTextParser(HTMLParser):
    """홈페이지의 사용자 안내 문구만 안전하게 추출한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._capture_tag: str | None = None
        self._buffer: list[str] = []
        self.sections: list[tuple[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if not self._ignored_depth and tag in VISIBLE_TAGS:
            self._capture_tag = tag
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth or tag != self._capture_tag:
            return
        text = " ".join("".join(self._buffer).split())
        if text:
            self.sections.append((tag, text))
        self._capture_tag = None
        self._buffer = []

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and self._capture_tag:
            self._buffer.append(data)


@dataclass(frozen=True)
class CrawledPage:
    url: str
    sections: tuple[tuple[str, str], ...]


def extract_visible_sections(html: str) -> tuple[tuple[str, str], ...]:
    parser = VisibleTextParser()
    parser.feed(html)
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for section in parser.sections:
        if section in seen:
            continue
        seen.add(section)
        unique.append(section)
    return tuple(unique)


def crawl_site(
    base_url: str,
    *,
    paths: tuple[str, ...] = ("/", "/features"),
    timeout_seconds: float = 15.0,
) -> list[CrawledPage]:
    """같은 MOVB 호스트의 공개 안내 페이지만 제한적으로 수집한다."""

    parsed_base = urlparse(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
        raise ValueError("base_url은 http 또는 https 홈페이지 주소여야 합니다.")

    pages: list[CrawledPage] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={"User-Agent": "MOVB-Knowledge-Crawler/1.0"},
    ) as client:
        for path in paths:
            url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
            if urlparse(url).netloc != parsed_base.netloc:
                raise ValueError("같은 홈페이지 호스트만 수집할 수 있습니다.")
            response = client.get(url)
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise ValueError(f"페이지가 너무 큽니다: {url}")
            pages.append(
                CrawledPage(
                    url=str(response.url),
                    sections=extract_visible_sections(response.text),
                )
            )
    return pages


def render_knowledge_snapshot(
    pages: list[CrawledPage],
    *,
    crawled_at: datetime | None = None,
) -> str:
    timestamp = (crawled_at or datetime.now(timezone.utc)).isoformat()
    lines = [
        "# MOVB 홈페이지 안내",
        "",
        f"> 홈페이지 공개 문구 수집 시각: {timestamp}",
        "",
    ]
    for page in pages:
        lines.extend([f"## 페이지 {page.url}", ""])
        for tag, text in page.sections:
            prefix = "### " if tag in {"h1", "h2", "h3", "summary"} else "- "
            lines.append(prefix + text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
