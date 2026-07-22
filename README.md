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

## Branch strategy

- `main`: 택시·퀵 관제 백엔드 전용
- `legacy-chatbot-full`: previous full repository snapshot (chatbot + RAG + training assets)
