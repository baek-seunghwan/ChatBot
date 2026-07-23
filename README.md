# 모브 (MOVB) 백엔드

이 프로젝트는 Kakao Mobility Sandbox API 연동을 위한 FastAPI 백엔드입니다.
현재 `main` 브랜치는 모브 (MOVB) 백엔드 코드만 남긴 상태입니다.

## Features

- Kakao auth check
- Delivery estimate and price lookup
- Sandbox order create/query/cancel
- Callback ingestion for order/step events
- Local SQLite persistence for orders, callbacks, and agent sessions
- Delivery assistant chat endpoint (`/api/agent/chat`)

## Run locally

```bash
uv sync
uv run uvicorn mobility_service.app:app --reload --port 8002
```

- Web: http://127.0.0.1:8002
- Docs: http://127.0.0.1:8002/docs
- Health: http://127.0.0.1:8002/health

## Edit the project

화면별 수정 파일, HTML·CSS·JavaScript 수정 방법, 로컬 확인, 테스트,
커밋·푸시와 Render 반영 과정은 [MOVB 직접 수정 가이드](EDITING_GUIDE.md)를
참고하세요.

## Environment

Use `.env.example` as a template.

Required for Kakao mobility sandbox:

- `KAKAO_MOBILITY_API_KEY` (or legacy `KakaoMobility_API`)
- `KAKAO_MOBILITY_VENDOR_ID` (or legacy `Vendor_ID`)

Required for map UI:

- `KAKAO_JAVASCRIPT_KEY`

Optional for address geocoding in agent flow:

- `KAKAO_REST_API_KEY`

Optional for LLM assistant mode:

- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

## Local chat and deployment

`내 로컬 채팅`의 Ollama mode connects from the FastAPI server to
`OLLAMA_BASE_URL`. With the default `http://localhost:11434`, it works only
when FastAPI and Ollama run on the same computer. A Render service cannot reach
Ollama running on a developer laptop. The web UI checks
`/api/local-chat/status` and automatically uses the built-in QA mode when
Ollama is unavailable.

To use Ollama from Render, host Ollama on a separately secured HTTPS model
server and set that reachable address as `OLLAMA_BASE_URL` in Render.

## Branch strategy

- `main`: 택시·퀵 관제 백엔드 전용
- `legacy-chatbot-full`: previous full repository snapshot (chatbot + RAG + training assets)
