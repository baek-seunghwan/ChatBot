from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


KNOWLEDGE_DIR = Path(__file__).with_name("knowledge")

# 기존 프롬프트와 로컬 Ollama 모드가 사용하는 짧은 서비스 개요입니다.
# 상세 답변은 아래 MobilityKnowledgeBase가 mobility_service/knowledge/*.md에서
# 관련 근거를 검색해 생성합니다.
SERVICE_FACTS = """[MOVB 서비스 정보]

MOVB(모브)는 Kakao Mobility 퀵·도보배송 Sandbox API를 연동한 AI 모빌리티
운영 서비스다. 일반 퀵, 퀵 이코노미, 퀵 급송, 도보 배송의 견적과 주문을
지원하고, 여러 목적지를 묶는 묶음배송과 서로 다른 사용자의 배송을 같은
방향으로 매칭하는 합승 배송 기능을 제공한다.

채팅 주문은 출발지·도착지·물품·연락처를 수집하고 주소를 좌표로 변환한 뒤
오토바이·다마스·라보·1톤 차량 선택, 실도로 거리와 예약 ETA, 배송 견적을
먼저 보여준다. 사용자가 명시적으로 확인해야 주문을 생성한다. 접수 후에는
출발지·경유지·목적지 Step 상태 조회와 취소가 가능하다.

현재는 Sandbox 환경이므로 실제 기사 배정이나 결제가 발생하지 않는다.
정보에 없는 실제 운영 정책이나 가격은 추측하지 말고 견적 API 또는 공식
안내를 확인하도록 설명해야 한다.
"""

_TOKEN_PATTERN = re.compile(r"[가-힣]{2,}|[a-z0-9_]{2,}", re.IGNORECASE)
_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_SPACE_PATTERN = re.compile(r"\s+")
_KOREAN_SUFFIXES = (
    "에서는", "으로는", "이라고", "이라는", "이랑", "에서", "으로",
    "하고", "처럼", "보다", "까지", "부터", "에도", "에게",
    "은", "는", "이", "가", "을", "를", "와", "과", "랑", "로", "에", "의",
)

_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "합승": ("풀링", "매칭", "같이", "분담"),
    "풀링": ("합승", "매칭"),
    "묶음": ("여러", "경유지", "배송"),
    "카풀": ("택시", "동승", "요금", "분담"),
    "동승": ("카풀", "택시", "분담"),
    "취소": ("주문", "접수", "철회"),
    "상태": ("조회", "배송", "현황"),
    "차량": ("오토바이", "다마스", "라보", "1톤", "fleet"),
    "거리": ("실도로", "길찾기", "경로", "카카오내비"),
    "예약": ("미래", "eta", "교통량", "wishTime"),
    "스텝": ("step", "출발지", "경유지", "목적지", "상태"),
    "단계": ("step", "출발지", "경유지", "목적지", "상태"),
    "가격": ("요금", "견적", "비용"),
    "요금": ("가격", "견적", "비용"),
    "종류": ("퀵", "도보", "상품"),
    "안전": ("확인", "중복", "견적", "sandbox"),
    "샌드박스": ("sandbox", "테스트", "결제"),
}


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        tokens.append(token)
        if re.fullmatch(r"[가-힣]+", token):
            for suffix in _KOREAN_SUFFIXES:
                if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                    tokens.append(token[: -len(suffix)])
                    break
    return tokens


def _normalized_chars(text: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def _bigrams(text: str) -> set[str]:
    normalized = _normalized_chars(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _expanded_query_tokens(query: str) -> list[str]:
    tokens = _tokens(query)
    expanded = list(tokens)
    normalized = _normalized_chars(query)
    for keyword, synonyms in _QUERY_EXPANSIONS.items():
        if keyword in normalized:
            expanded.extend(synonyms)
    return expanded


def _slug(text: str) -> str:
    slug = re.sub(r"[^가-힣a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    title: str
    source: str
    content: str


@dataclass(frozen=True)
class KnowledgeSearchResult:
    chunk_id: str
    title: str
    source: str
    content: str
    score: float

    def to_source(self) -> dict[str, str | float]:
        return {
            "id": self.chunk_id,
            "title": self.title,
            "source": self.source,
            "score": round(self.score, 4),
        }


def _parse_markdown(path: Path, root: Path) -> list[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    document_title = path.stem.replace("-", " ")
    sections: list[tuple[str, list[str]]] = []
    current_title = document_title
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        match = _HEADING_PATTERN.match(raw_line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1:
                document_title = heading
                current_title = heading
                continue
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = f"{document_title} › {heading}"
            current_lines = []
            continue
        current_lines.append(raw_line)

    if current_lines:
        sections.append((current_title, current_lines))

    source = path.relative_to(root.parent).as_posix()
    chunks: list[KnowledgeChunk] = []
    for index, (title, lines) in enumerate(sections, start=1):
        content = "\n".join(lines).strip()
        if not content:
            continue
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{path.stem}#{_slug(title)}-{index}",
                title=title,
                source=source,
                content=content,
            )
        )
    return chunks


class MobilityKnowledgeBase:
    """외부 Vector DB 없이 동작하는 작은 모빌리티 도메인 검색기.

    문서가 작을 때는 별도 인프라보다 재현 가능한 BM25 + 문자 n-gram 검색이
    운영과 평가에 유리합니다. 문서가 커지면 같은 search 인터페이스를 임베딩
    검색으로 교체할 수 있습니다.
    """

    def __init__(self, directory: Path = KNOWLEDGE_DIR) -> None:
        self.directory = directory
        paths = sorted(directory.glob("*.md")) if directory.exists() else []
        self.chunks = [
            chunk
            for path in paths
            for chunk in _parse_markdown(path, directory)
        ]
        self._token_counts = [Counter(_tokens(chunk.title + "\n" + chunk.content)) for chunk in self.chunks]
        self._document_frequency: Counter[str] = Counter()
        for counts in self._token_counts:
            self._document_frequency.update(counts.keys())
        self._average_length = (
            sum(sum(counts.values()) for counts in self._token_counts) / len(self._token_counts)
            if self._token_counts
            else 1.0
        )

    def _bm25(self, query_tokens: list[str], index: int) -> float:
        counts = self._token_counts[index]
        document_length = max(sum(counts.values()), 1)
        document_count = max(len(self.chunks), 1)
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequency.get(token, 0)
            inverse_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * document_length / self._average_length
            )
            score += inverse_frequency * (frequency * 2.5) / denominator
        return score

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        min_score: float = 0.55,
    ) -> list[KnowledgeSearchResult]:
        if not query.strip() or not self.chunks:
            return []

        query_tokens = _expanded_query_tokens(query)
        query_bigrams = _bigrams(query)
        results: list[KnowledgeSearchResult] = []
        for index, chunk in enumerate(self.chunks):
            searchable = chunk.title + "\n" + chunk.content
            candidate_bigrams = _bigrams(searchable)
            character_recall = (
                len(query_bigrams & candidate_bigrams) / len(query_bigrams)
                if query_bigrams
                else 0.0
            )
            title = _normalized_chars(chunk.title)
            title_bonus = sum(
                0.35 for token in set(query_tokens) if len(token) >= 2 and token in title
            )
            score = self._bm25(query_tokens, index) + character_recall * 2.5 + title_bonus
            if score < min_score:
                continue
            results.append(
                KnowledgeSearchResult(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    source=chunk.source,
                    content=chunk.content,
                    score=score,
                )
            )

        results.sort(key=lambda result: (-result.score, result.chunk_id))
        return results[:limit]

    @staticmethod
    def context(results: list[KnowledgeSearchResult]) -> str:
        return "\n\n".join(
            f"[근거 {index}: {result.title}]\n{result.content}"
            for index, result in enumerate(results, start=1)
        )

    @staticmethod
    def fallback_answer(results: list[KnowledgeSearchResult]) -> str:
        if not results:
            return (
                "MOVB 지식 문서에서 관련 근거를 찾지 못했어요. "
                "배송 종류, 주문 방법, 묶음배송, 합승, 카풀 또는 Sandbox 이용 방법처럼 "
                "조금 더 구체적으로 물어봐 주세요."
            )

        top = results[0]
        paragraphs = [
            _SPACE_PATTERN.sub(" ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", top.content)
            if paragraph.strip()
        ]
        answer = "\n".join(paragraphs[:2])
        if len(answer) > 700:
            answer = answer[:697].rstrip() + "..."
        sources = ", ".join(f"[{result.title}]" for result in results[:2])
        return f"{answer}\n\n근거 문서: {sources}"


@lru_cache(maxsize=1)
def default_knowledge_base() -> MobilityKnowledgeBase:
    return MobilityKnowledgeBase()
