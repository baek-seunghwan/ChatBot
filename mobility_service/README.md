# mobility_service

이 문서는 MOVB 서비스 패키지의 구현·연동 기준을 정리합니다.
프로젝트 개요, 설치, Docker 실행은 루트 [README.md](../README.md)를 보고,
여기서는 실제 백엔드가 어떤 외부 API와 상태 흐름을 가지는지 확인하면 됩니다.

## 범위

`mobility_service` 패키지는 다음 책임을 가집니다.

- FastAPI 라우트와 의존성 조립
- Kakao Mobility Sandbox 견적, 주문, 조회, 취소, Step 조회
- Kakao Local 주소 변환과 Kakao Navi 경로 요약
- LangGraph Agent 기반 자연어 주문 보조
- BM25 + 문자 n-gram Knowledge RAG
- 스마트 딜리버리 비교 견적과 주문 변환
- KakaoPay 개발 결제 준비·승인 후 배송 주문 연결
- SQLite 기반 주문, 콜백, 대화, 결제 상태 저장

## 주요 모듈

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI 앱 생성, 라우트 등록, 예외 처리 |
| `client.py` | Kakao Mobility Sandbox HTTP 클라이언트 |
| `directions.py` | 실도로 길찾기와 경로 요약 |
| `agent.py` | LangGraph Agent와 슬롯 수집 흐름 |
| `bundle.py` | 스마트 딜리버리 견적 계산과 주문 변환 |
| `kakaopay.py` | KakaoPay 개발 결제 준비·승인 클라이언트 |
| `store.py` | 주문·콜백·매칭·결제 SQLite 저장소 |
| `conversation_store.py` | 채팅 세션과 슬롯 상태 저장 |
| `knowledge.py` | 지식 문서 색인과 검색 |
| `local_responder.py` | 로컬 QA, Ollama, vLLM 응답 경로 |

## 실행과 점검

프로젝트 루트에서 서버를 실행합니다.

```bash
uv run uvicorn mobility_service.app:app --reload --port 8002
```

개발 중 가장 먼저 확인할 엔드포인트는 아래와 같습니다.

| 경로 | 목적 |
|---|---|
| `/health` | 프로세스 상태 확인 |
| `/api/config` | 지도, 관리자, 결제, LLM 설정 노출 여부 확인 |
| `/api/kakao/auth-check` | Kakao Mobility 인증 확인 |
| `/api/local-chat/status` | Ollama 연결 상태 확인 |
| `/api/vllm/status` | OpenAI 호환 vLLM 연결 상태 확인 |
| `/docs` | 전체 OpenAPI 확인 |

## 환경변수 매핑

실제 키 목록은 [../.env.example](../.env.example)에 있습니다. 여기서는 어떤 기능이 어떤 값을 요구하는지만 정리합니다.

| 기능 | 필수 값 |
|---|---|
| Kakao Mobility 주문·조회 | `KAKAO_MOBILITY_API_KEY`, `KAKAO_MOBILITY_VENDOR_ID` |
| Kakao Mobility 콜백 | `KAKAO_MOBILITY_CALLBACK_BASE_URL` |
| 지도 SDK | `KAKAO_JAVASCRIPT_KEY` |
| 주소 변환·길찾기 | `KAKAO_REST_API_KEY` |
| 관리자 계정 | `MOVB_ADMIN_USERNAME`, `MOVB_ADMIN_PASSWORD` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| KakaoPay 개발 결제 | `KAKAOPAY_SECRET_KEY_DEV`, `KAKAOPAY_CID` |
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| vLLM | `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY` |

### 호환 키 이름

Kakao Mobility는 과거 환경파일과 호환되도록 아래 별칭도 읽습니다.

- `KakaoMobility_API` -> `KAKAO_MOBILITY_API_KEY`
- `Vendor_ID` -> `KAKAO_MOBILITY_VENDOR_ID`
- `KAKAO_MAP_KEY` -> `KAKAO_JAVASCRIPT_KEY`

## 핵심 API 흐름

### 일반 배송 주문

1. `POST /api/deliveries/estimate`로 ETA와 경로 요약을 확인합니다.
2. `POST /api/deliveries/price`로 배송비를 확인합니다.
3. `POST /api/orders`로 주문을 생성합니다.
4. `GET /api/orders/{partnerOrderId}`와 `GET /api/orders/{partnerOrderId}/steps`로 상태를 조회합니다.
5. 필요하면 `PATCH /api/orders/{partnerOrderId}/cancel`로 취소합니다.

`Idempotency-Key` 헤더 또는 요청의 `partner_order_id`로 동일 주문 중복 생성을 방지합니다.

### 스마트 딜리버리

1. `POST /api/smart-delivery/quote`로 다중 픽업·배송 비교 견적을 조회합니다.
2. 사용자가 최종 경로와 금액에 동의하면 `POST /api/smart-delivery/orders`를 호출합니다.
3. 서버는 주소, 경유 순서, 금액, 동의 여부를 다시 검증한 뒤 일반 Kakao 주문 포맷으로 변환합니다.

기존 `/api/bundle/*` 경로는 하위 호환 별칭입니다.

### 자연어 채팅

- `POST /api/agent/chat`: 기본 AI 채팅 진입점입니다.
- `mode=local`: 외부 LLM 대신 로컬 QA 또는 로컬 모델 경로를 사용합니다.
- `GET /api/knowledge/search`: RAG 검색 결과만 따로 점검할 때 사용합니다.

채팅은 의도 분류 후 질문 응답과 실행형 Workflow를 분리합니다. 주문 생성처럼 위험한 동작은
대화 응답만으로 끝나지 않고 서버 검증 단계를 반드시 거칩니다.

### KakaoPay 개발 결제

1. 로그인된 사용자가 `POST /api/payments/kakaopay/ready`를 호출합니다.
2. 서버가 최신 배송 견적에서 결제 금액을 계산하고 KakaoPay 준비 API를 호출합니다.
3. 사용자가 KakaoPay Redirect를 거쳐 `/success`, `/cancel`, `/fail` 엔드포인트로 돌아옵니다.
4. 성공 시 결제 승인 후에만 실제 배송 주문을 생성합니다.

자동 매칭 스마트 딜리버리는 금액이 매칭 뒤에 확정되므로 순서가 다릅니다.

1. 두 사용자의 주문을 `WAITING` 상태로 등록합니다.
2. 경로와 배송 조건이 맞으면 묶음 경로·총액·참가자별 금액을 확정하고
   `MATCHED_AWAITING_PAYMENT`로 전환합니다.
3. 각 사용자가 자신의 확정 금액을 KakaoPay로 결제합니다.
4. 두 결제가 모두 승인된 경우에만 묶음 배송 주문을 한 번 생성합니다.

이 흐름에서는 Secret key가 브라우저로 전달되지 않습니다.

## 콜백 처리

Kakao Mobility 콘솔에는 공개 HTTPS 기준 주소를 등록해야 합니다.

```text
https://api.example.com
```

서비스가 받는 콜백 경로는 다음과 같습니다.

```text
PUT /api/v1/callback/orders/{partnerOrderId}/{event}
PUT /api/v1/callback/orders/{orderId}/steps/{stepId}
```

운영 규칙:

- 중복 콜백은 저장소에서 식별해 무시합니다.
- 늦게 도착한 과거 상태가 현재 주문 상태를 되돌리지 못하게 막습니다.
- `localhost`는 외부에서 접근할 수 없으므로 콜백 시험에는 공개 HTTPS 주소가 필요합니다.

## 인증과 운영 화면

- `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`
- `GET /api/auth/me`
- `GET /api/admin/summary`, `GET /api/admin/users`, `GET /api/admin/orders`
- `PATCH /api/admin/orders/{partnerOrderId}/sandbox-status`

세션은 HttpOnly 쿠키로 관리하며, 관리자 엔드포인트는 `ADMIN` 역할이 필요합니다.

## 문제 파악 순서

연동 이슈가 있을 때는 아래 순서로 확인하는 편이 가장 빠릅니다.

1. `/api/config`에서 어떤 기능이 설정되었는지 확인합니다.
2. `/api/kakao/auth-check`로 Kakao Mobility 인증부터 점검합니다.
3. 주소·경로 이슈는 `KAKAO_REST_API_KEY`와 `/api/routes/summary`를 확인합니다.
4. 채팅 이슈는 `/api/knowledge/search`, `/api/local-chat/status`, `/api/vllm/status`를 순서대로 봅니다.
5. 주문 상태 불일치는 `/api/orders/{partnerOrderId}`와 콜백 저장 상태를 함께 확인합니다.

## 검증 명령

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/evaluate_mobility_knowledge.py --threshold 0.90
uv run python scripts/crawl_movb_site.py --base-url https://movb.onrender.com
```

첫 두 명령은 CI에서 실행됩니다. 사이트 크롤링은 지식 문서 갱신이 필요할 때만 수동으로 사용합니다.

## 참고 문서

- [프로젝트 개요](../README.md)
- [Kakao Mobility 길찾기 API](https://developers.kakaomobility.com/guide/navi-api/start)
- [Kakao Mobility 배송 API](https://logistics-developers.kakaomobility.com/document/post-orders)
