from __future__ import annotations

import re
from typing import Any


_PHONE_PATTERN = re.compile(r"(?<!\d)(01[016789])[-.\s]?(\d{3,4})[-.\s]?(\d{4})(?!\d)")
_LABELED_NAME_PATTERN = re.compile(
    r"(?:받는\s*(?:분|사람)|수령인|수취인|이름|성함)\s*[:：\-]?\s*"
    r"([가-힣A-Za-z][가-힣A-Za-z \t]{1,19})",
    re.IGNORECASE,
)
_NAME_BEFORE_PHONE_PATTERN = re.compile(
    r"(?:^|\n)\s*([가-힣A-Za-z]{2,20})\s+"
    r"(?=01[016789][\-.\s]?\d{3,4}[\-.\s]?\d{4})",
    re.IGNORECASE,
)
_ADDRESS_LABEL_PATTERN = re.compile(
    r"^(?:받는\s*곳|배송지|도착지|주소|수령\s*주소)\s*[:：\-]?\s*",
    re.IGNORECASE,
)
_ADDRESS_REGION_PATTERN = re.compile(
    r"(서울(?:특별시)?|부산(?:광역시)?|대구(?:광역시)?|인천(?:광역시)?|"
    r"광주(?:광역시)?|대전(?:광역시)?|울산(?:광역시)?|세종(?:특별자치시)?|"
    r"경기(?:도)?|강원(?:특별자치도|도)?|충북|충남|전북|전남|경북|경남|"
    r"제주(?:특별자치도|도)?)"
)
_ROAD_ADDRESS_PATTERN = re.compile(r"(?:로|길)\s*\d+(?:-\d+)?")
_LOT_ADDRESS_PATTERN = re.compile(r"(?:읍|면|동|리)\s*\d+(?:-\d+)?")
_DETAIL_PATTERN = re.compile(
    r"((?:지하\s*)?\d+\s*층|(?:제?\s*)?\d+\s*동\s*\d+\s*호|"
    r"\d+\s*호|[가-힣A-Za-z0-9·\-]+\s*(?:빌딩|타워|아파트|오피스텔))"
)
_PRODUCT_PATTERN = re.compile(
    r"(?:물품|품목|보낼\s*것|배송\s*물품)\s*[:：\-]?\s*(.{1,80})",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    line = re.sub(r"^[>\-•·*\s]+", "", line.strip())
    return re.sub(r"\s+", " ", line).strip()


def _address_score(line: str) -> int:
    score = 0
    if _ADDRESS_REGION_PATTERN.search(line):
        score += 4
    if _ROAD_ADDRESS_PATTERN.search(line):
        score += 4
    if _LOT_ADDRESS_PATTERN.search(line):
        score += 3
    if _ADDRESS_LABEL_PATTERN.match(line):
        score += 3
    if re.search(r"(시|군|구)\s", line):
        score += 2
    if any(word in line for word in ("연락처", "전화", "물품", "이름", "성함")):
        score -= 4
    if len(line) > 120:
        score -= 2
    return score


def extract_dropoff_slots(text: str) -> dict[str, Any]:
    """Extract recipient fields from a copied message or OCR text.

    This deliberately prefers precision over guessing. The extracted values are
    safe to use as draft form values, while address geocoding remains the final
    validation step before quoting or ordering.
    """

    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return {"slots": {}, "detected": [], "confidence": 0.0}

    slots: dict[str, Any] = {}
    detected: list[str] = []

    phone_match = _PHONE_PATTERN.search(normalized)
    if phone_match:
        slots["dropoffPhone"] = "-".join(phone_match.groups())
        detected.append("연락처")

    name_match = (
        _LABELED_NAME_PATTERN.search(normalized)
        or _NAME_BEFORE_PHONE_PATTERN.search(normalized)
    )
    if name_match:
        name = re.split(
            r"\s*(?:연락처|전화|주소|배송지|도착지)\s*[:：]?",
            name_match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,./\n\t")
        if 2 <= len(name) <= 20:
            slots["dropoffName"] = name
            detected.append("받는 사람")

    lines = [_clean_line(line) for line in normalized.splitlines()]
    lines = [line for line in lines if line]
    candidates = sorted(
        ((_address_score(line), index, line) for index, line in enumerate(lines)),
        key=lambda item: (-item[0], item[1]),
    )
    if candidates and candidates[0][0] >= 4:
        address_line = _ADDRESS_LABEL_PATTERN.sub("", candidates[0][2]).strip()
        address_line = _PHONE_PATTERN.sub("", address_line)
        address_line = re.sub(
            r"\s*(?:받는\s*(?:분|사람)|수령인|이름|성함)\s*[:：]\s*"
            r"[가-힣A-Za-z][가-힣A-Za-z\s]{1,19}$",
            "",
            address_line,
            flags=re.IGNORECASE,
        ).strip(" ,/")

        detail_match = _DETAIL_PATTERN.search(address_line)
        if detail_match:
            detail = address_line[detail_match.start() :].strip(" ,")
            basic = address_line[: detail_match.start()].strip(" ,")
            if basic:
                slots["dropoffAddress"] = basic
                slots["dropoffDetailAddress"] = detail
                detected.extend(["도착지 주소", "상세주소"])
        else:
            slots["dropoffAddress"] = address_line
            detected.append("도착지 주소")

    product_match = _PRODUCT_PATTERN.search(normalized)
    if product_match:
        product = product_match.group(1).splitlines()[0].strip(" ,.")
        if product:
            slots["productName"] = product
            detected.append("물품")

    confidence_parts = {
        "dropoffAddress": 0.45,
        "dropoffName": 0.2,
        "dropoffPhone": 0.25,
        "productName": 0.1,
    }
    confidence = min(
        1.0,
        sum(weight for key, weight in confidence_parts.items() if slots.get(key)),
    )
    return {
        "slots": slots,
        "detected": detected,
        "confidence": round(confidence, 2),
    }
