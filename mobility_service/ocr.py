from __future__ import annotations

import base64
import binascii
import shutil
import subprocess
import tempfile
from pathlib import Path


MAX_OCR_BYTES = 5 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class OcrUnavailableError(RuntimeError):
    pass


def extract_text_from_image(image_base64: str, content_type: str) -> str:
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("JPG, PNG 또는 WEBP 이미지만 읽을 수 있습니다.")
    if not shutil.which("tesseract"):
        raise OcrUnavailableError("이 서버에는 OCR 엔진이 설치되어 있지 않습니다.")

    encoded = image_base64.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("이미지 데이터가 올바르지 않습니다.") from exc
    if not image_bytes:
        raise ValueError("빈 이미지입니다.")
    if len(image_bytes) > MAX_OCR_BYTES:
        raise ValueError("이미지는 5MB 이하로 올려주세요.")

    suffix = SUPPORTED_IMAGE_TYPES[content_type]
    with tempfile.TemporaryDirectory(prefix="movb-ocr-") as directory:
        image_path = Path(directory) / f"upload{suffix}"
        image_path.write_bytes(image_bytes)
        try:
            result = subprocess.run(
                [
                    "tesseract",
                    str(image_path),
                    "stdout",
                    "-l",
                    "kor+eng",
                    "--psm",
                    "6",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=25,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("사진을 읽는 데 시간이 너무 오래 걸렸습니다.") from exc

    if result.returncode != 0:
        raise RuntimeError("사진에서 글자를 읽지 못했습니다.")
    text = result.stdout.strip()
    if not text:
        raise ValueError("사진에서 읽을 수 있는 글자를 찾지 못했습니다.")
    return text[:5000]
