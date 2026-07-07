# ChatBot

로그인 후 질문을 입력하면 하나의 AI 답변만 보여주는 FastAPI 챗봇입니다.
내부 응답 경로는 Claude를 우선 호출하고, 실패하면 Gemini로 전환하며, 외부 API를 사용할 수 없을 때는 로컬 학습 모델을 마지막 폴백으로 사용합니다.

## 실행

```bash
cd ~/Documents/GitHub/ChatBot
uv sync
uv run uvicorn chatbot.local_chat.app:app --reload --port 8001
```

- 웹 화면: `http://127.0.0.1:8001`
- API 문서: `http://127.0.0.1:8001/docs`
- 회원가입: `POST /api/auth/signup`
- 로그인: `POST /api/auth/login`
- 챗봇 질문: `POST /api/chat/ask`
- 대화 기록: `GET /api/chat/history`

API 키는 프로젝트 루트의 `.env`에 둡니다.

```bash
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite
```

## Colab 학습

VS Code에서 Colab 커널에 연결한 뒤 [colab_train.ipynb](/Users/samrobert/Documents/GitHub/ChatBot/colab_train.ipynb)의 셀 하나를 실행합니다.
학습 번들은 노트북 안에 들어 있으므로 파일 업로드 창을 쓰지 않습니다.

현재 설정은 GPU 기준으로 6 epoch 전체 학습을 실행합니다.
`embedding_dim=384`, `num_layers=8`, `num_heads=8`, `block_size=256`으로 기본 로컬 모델보다 크게 학습합니다.

학습 결과는 Colab 런타임의 `/content/chatbot.pt`에 저장됩니다.
가져온 파일은 아래 파일과 교체합니다.

```bash
~/Documents/GitHub/ChatBot/artifacts/chatbot.pt
```

## 주요 파일

- `chatbot/local_chat/`: 로그인, DB, 웹 화면, 통합 챗봇 FastAPI 라우터
- `chatbot/model.py`: 로컬 학습 모델 로딩, 생성, 마지막 폴백 코드
- `chatbot/providers.py`: Claude 우선 호출, Gemini 폴백 호출
- `chatbot/train.py`: 말뭉치 기반 로컬 모델 학습 코드
- `chatbot/autocomplete_corpus_nikl.txt`: 학습 말뭉치
- `artifacts/chatbot.pt`: 현재 앱에서 쓰는 로컬 모델 체크포인트
- `colab_train.ipynb`: VS Code Colab 커널용 학습 노트북
