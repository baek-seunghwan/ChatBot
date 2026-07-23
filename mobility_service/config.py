from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_fallback(dotenv_path: Path) -> None:
    """Load KEY=VALUE pairs from .env when python-dotenv is unavailable.

    Local development expects .env to be the source of truth, so parsed values
    intentionally override existing process env vars.
    """
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
except ImportError:
    _load_dotenv_fallback(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    vendor_id: str
    base_url: str
    callback_base_url: str
    database_path: Path
    kakao_javascript_key: str = ""
    kakao_rest_api_key: str = ""
    directions_base_url: str = "https://apis-navi.kakaomobility.com"
    admin_username: str = ""
    admin_password: str = ""
    request_timeout_seconds: float = 10.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip() and self.vendor_id.strip())

    @property
    def map_configured(self) -> bool:
        return bool(self.kakao_javascript_key.strip())

    @property
    def geocoding_configured(self) -> bool:
        return bool(self.kakao_rest_api_key.strip())

    @property
    def directions_configured(self) -> bool:
        return bool(self.kakao_rest_api_key.strip())

    @property
    def admin_configured(self) -> bool:
        return bool(self.admin_username.strip() and self.admin_password)

    @classmethod
    def from_env(cls) -> "Settings":
        # 사용자가 이미 만든 키 이름과 일반적인 대문자 이름을 모두 지원한다.
        api_key = os.getenv("KAKAO_MOBILITY_API_KEY") or os.getenv(
            "KakaoMobility_API", ""
        )
        vendor_id = os.getenv("KAKAO_MOBILITY_VENDOR_ID") or os.getenv(
            "Vendor_ID", ""
        )
        database_path = Path(
            os.getenv(
                "KAKAO_MOBILITY_DATABASE_PATH",
                REPO_ROOT / "mobility_service" / "data" / "mobility.db",
            )
        )
        return cls(
            api_key=api_key.strip(),
            vendor_id=vendor_id.strip(),
            base_url=os.getenv(
                "KAKAO_MOBILITY_BASE_URL",
                "https://open-api-logistics.kakaomobility.com",
            ).rstrip("/"),
            callback_base_url=os.getenv(
                "KAKAO_MOBILITY_CALLBACK_BASE_URL", ""
            ).rstrip("/"),
            database_path=database_path,
            kakao_javascript_key=(
                os.getenv("KAKAO_JAVASCRIPT_KEY")
                or os.getenv("KAKAO_MAP_KEY", "")
            ).strip(),
            kakao_rest_api_key=os.getenv("KAKAO_REST_API_KEY", "").strip(),
            directions_base_url=os.getenv(
                "KAKAO_DIRECTIONS_BASE_URL",
                "https://apis-navi.kakaomobility.com",
            ).rstrip("/"),
            admin_username=os.getenv("MOVB_ADMIN_USERNAME", "").strip(),
            admin_password=os.getenv("MOVB_ADMIN_PASSWORD", ""),
            request_timeout_seconds=float(
                os.getenv("KAKAO_MOBILITY_TIMEOUT_SECONDS", "10")
            ),
        )
