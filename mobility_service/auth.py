from __future__ import annotations

import base64
import hashlib
import secrets
import time


def build_authorization(
    api_key: str,
    *,
    timestamp_ms: int | None = None,
    nonce: int | None = None,
) -> str:
    """카카오 물류 API 문서의 SHA-512 서명 형식으로 Authorization을 만든다."""

    timestamp = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    nonce_value = nonce if nonce is not None else secrets.randbelow(900_000) + 100_000
    plain_text = f"{timestamp}{nonce_value}{api_key}"
    sign_key = hashlib.sha512(plain_text.encode("utf-8")).hexdigest()
    raw_authorization = f"{timestamp}$${nonce_value}$${sign_key}"
    return base64.b64encode(raw_authorization.encode("utf-8")).decode("ascii")

