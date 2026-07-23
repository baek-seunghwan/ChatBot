# MOVB — AI 모빌리티 운영 서비스

LangGraph Agent와 근거 기반 Knowledge RAG가 Kakao Mobility 퀵·도보 배송
Sandbox 업무를 연결하는 FastAPI 웹 서비스입니다.

## 제공 기능

- API 키 인증 상태 확인
- 이메일 회원가입·로그인·로그아웃과 30일 세션 유지
- PBKDF2 비밀번호 해시 및 HttpOnly 세션 쿠키
- 환경변수 기반 최초 관리자 계정과 역할 기반 접근 제어
- 관리자 전용 회원·주문·합승 현황 대시보드
- 배송 예상 시간 조회 & 배송 가격 조회
- 자동차·다중 경유지 길찾기의 실도로 거리 및 예상 시간
- 미래 운행 정보로 예약 배송 ETA 보강
- 챗봇·접수 화면의 오토바이·다마스·라보·1톤 선택
- 카카오 지도 출발지·도착지 표시
- 주소 검색 및 지도 클릭 좌표 입력
- Sandbox 주문 생성
- 주문 조회 및 카카오 상태 동기화
- 배송원 정보 조회 API
- 주문 취소
- 주문·스텝 상태 콜백 수신
- 출발지·경유지·목적지 Step 상태 단건 조회
- 관리자 Sandbox 배송 생명주기 시연
- SQLite 주문 및 콜백 이력 보존
- 같은 `partnerOrderId`의 중복 주문 방지
- 중복 콜백 제거 및 역순 상태 변경 방지
- 17개 노드의 LangGraph 배송 Agent
- 서비스 질문과 업무 요청을 분리하는 하이브리드 의도 분류
- 6개 MOVB 문서를 검색하는 BM25 + 문자 n-gram Knowledge RAG
- LLM 미설정 시 근거 문서를 이용한 추출형 답변
- Anthropic 호출 실패 시 Gemini 자동 폴백
- 묶음배송 비교 견적, 택시 동승 요금 분배, 사용자 간 퀵 합승

## 환경변수

프로젝트 루트 `.env`에서 기존 키 이름이나 표준 대문자 이름 중 하나를
사용할 수 있습니다.

```dotenv
# 기존 키 이름
KakaoMobility_API=발급받은_API_KEY
Vendor_ID=발급받은_VENDOR_ID

# 또는 표준 키 이름
KAKAO_MOBILITY_API_KEY=발급받은_API_KEY
KAKAO_MOBILITY_VENDOR_ID=발급받은_VENDOR_ID

KAKAO_MOBILITY_BASE_URL=https://open-api-logistics.kakaomobility.com
KAKAO_MOBILITY_CALLBACK_BASE_URL=https://api.example.com

# 카카오 지도에는 JavaScript 키를 사용합니다.
KAKAO_JAVASCRIPT_KEY=발급받은_JAVASCRIPT_KEY

# 주소 변환 + 카카오모빌리티 길찾기(자동차·다중 경유지·미래 운행)
KAKAO_REST_API_KEY=카카오디벨로퍼스_REST_API_KEY
KAKAO_DIRECTIONS_BASE_URL=https://apis-navi.kakaomobility.com

# 관리자 계정 — 비밀번호는 .env에만 저장합니다.
MOVB_ADMIN_USERNAME=관리자_아이디
MOVB_ADMIN_PASSWORD=8자_이상_비밀번호
```

`.env`는 Git에서 제외됩니다. 키나 실제 연락처를 저장소에 커밋하지 마세요.
지도 화면에는 웹 SDK 특성상 JavaScript 키만 전달되며, REST API 키와
Native App 키는 전달되지 않습니다.

카카오 디벨로퍼스의 JavaScript 키 설정에서 로컬 테스트 주소
`http://127.0.0.1:8002`를 JavaScript SDK 도메인으로 등록해야 지도가
표시됩니다. 운영할 때는 실제 HTTPS 서비스 주소도 함께 등록하세요.

길찾기는 [카카오모빌리티 길찾기 API 공식 문서](https://developers.kakaomobility.com/guide/navi-api/start),
배송 주문·차량·Step은 [퀵·도보 배송 API 공식 문서](https://logistics-developers.kakaomobility.com/document/post-orders)
형식을 따릅니다. 현재 주문 연동은 Sandbox 기준이며, 실제 운영 사용 전에는
카카오모빌리티의 운영 환경 연동 절차와 승인이 별도로 필요합니다.

## 실행

프로젝트 루트에서 실행합니다.

```bash
uv sync
uv run uvicorn mobility_service.app:app --reload --port 8002
```

- 웹 화면: <http://127.0.0.1:8002>
- API 문서: <http://127.0.0.1:8002/docs>
- 상태 확인: <http://127.0.0.1:8002/health>
- 카카오 인증 확인: <http://127.0.0.1:8002/api/kakao/auth-check>
- 관리자 화면: <http://127.0.0.1:8002/admin>
- 지식 검색: <http://127.0.0.1:8002/api/knowledge/search?q=합승%20요금>

직접 화면이나 API 코드를 수정하려면
[MOVB 직접 수정 가이드](../EDITING_GUIDE.md)를 먼저 확인하세요. 화면별 파일
위치부터 로컬 확인, 테스트, GitHub 푸시와 Render 반영까지 정리되어 있습니다.

화면에 기본으로 입력된 주소와 연락처는 Sandbox 테스트용 예시입니다.
실제 개인정보를 입력하지 마세요.

## 콜백

API 키 발급 화면에는 공개 서버의 기본 HTTPS 주소를 등록합니다.

```text
https://api.example.com
```

카카오모빌리티가 다음 엔드포인트로 상태 변경을 전달합니다.

```text
PUT /api/v1/callback/orders/{partnerOrderId}/{event}
PUT /api/v1/callback/orders/{orderId}/steps/{stepId}
```

`localhost`는 카카오모빌리티 서버에서 접근할 수 없습니다. 콜백까지
시험하려면 80 또는 443 포트로 접근 가능한 공개 서버가 필요합니다.

## 테스트

테스트는 실제 배송 주문을 생성하지 않고 Mock API를 사용합니다.

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/evaluate_mobility_knowledge.py
```

검증 항목:

- 문서에 명시된 SHA-512 Authorization 형식
- Vendor ID와 Authorization 헤더
- 가격 조회 요청 변환
- 동일 주문 ID의 멱등성
- 콜백 중복 제거
- 역순 콜백에 의한 상태 후퇴 방지
- 회원가입·로그인·로그아웃 세션 수명주기
- 중복 이메일 및 잘못된 비밀번호 거부
- 일반 사용자와 관리자의 접근 권한 분리
- 서비스 질문의 Knowledge RAG 경로와 출처 반환
- LLM이 없는 환경의 추출형 답변
- 질문과 합승 등록 요청의 의도 구분
- 차량 선택 버튼과 슬롯 저장
- 자동차·다중 경유지·미래 운행 요청 형식
- Step 상태 단건 조회와 관리자 Sandbox 상태 변경
- 26개 질문의 기대 근거 Source Hit@3
