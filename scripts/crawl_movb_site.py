from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobility_service.site_crawler import crawl_site, render_knowledge_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOVB 공개 홈페이지 문구를 챗봇 지식 문서로 저장합니다."
    )
    parser.add_argument(
        "--base-url",
        default="https://movb.onrender.com",
        help="수집할 MOVB 홈페이지 주소",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mobility_service/knowledge/06-homepage-crawl.md"),
        help="저장할 Markdown 파일",
    )
    args = parser.parse_args()

    pages = crawl_site(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_knowledge_snapshot(pages),
        encoding="utf-8",
    )
    print(f"{len(pages)}개 페이지를 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
