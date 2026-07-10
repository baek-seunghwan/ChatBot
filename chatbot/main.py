"""기존 실행 명령과의 호환성을 위한 단일 앱 진입점.

`chatbot.main:app`과 `chatbot.local_chat.app:app`은 모두 같은 로그인형
Leon's ChatBot 화면을 실행한다. 별도의 파란색 데모 화면은 제공하지 않는다.
"""

from .local_chat.app import app

__all__ = ["app"]
