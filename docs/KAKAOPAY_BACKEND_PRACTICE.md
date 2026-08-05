# 카카오페이 백엔드 구현 연습

이 문서는 MOVB에 추가된 카카오페이 단건결제를 실제 백엔드 개발자의 작업 순서대로
다시 만들어 보는 연습 자료입니다. 완성 코드를 먼저 복사하지 말고 각 단계의 TODO를
직접 구현한 뒤 실제 파일과 비교해 보세요.

## 0. 가장 먼저 지킬 보안 규칙

- 채팅, Git, HTML, JavaScript에 Secret key를 적지 않습니다.
- 노출된 키는 삭제만 하지 말고 개발자센터에서 **재발급**합니다.
- 브라우저는 우리 서버의 `/ready`만 호출합니다.
- 카카오페이 `ready`와 `approve` API는 항상 백엔드 서버가 호출합니다.
- 결제 금액은 브라우저가 보낸 숫자를 믿지 않고 서버가 다시 계산합니다.

MOVB에서 사용하는 값은 다음처럼 `.env`에만 둡니다.

```dotenv
KAKAOPAY_SECRET_KEY_DEV=재발급한_개발용_키
KAKAOPAY_CID=TC0ONETIME
KAKAOPAY_BASE_URL=https://open-api.kakaopay.com
KAKAOPAY_REDIRECT_BASE_URL=https://내-공개-개발주소.example
```

`Client ID`와 `Client Secret`은 카카오페이 로그인용이며, 이 단건결제 예제의
`ready/approve` 인증에는 사용하지 않습니다.

## 1. 전체 결제 흐름 그리기

```text
사용자 → MOVB /ready → 배송 견적 재조회 → 카카오페이 /ready
사용자 ← 카카오페이 결제창 URL
사용자 → 카카오페이 인증
카카오페이 → MOVB approval_url?pg_token=...
MOVB → 카카오페이 /approve
MOVB → 배송 Sandbox 주문 생성
사용자 ← 결제·접수 결과 화면
```

중요한 기준은 **approve 성공 전에는 배송 주문을 만들지 않는 것**입니다.

## 2. 설정 객체 만들기

연습 파일: `mobility_service/config.py`

TODO:

1. 개발용 Secret key, CID, API 기본 주소, Redirect 기본 주소 필드를 추가합니다.
2. `kakaopay_configured` 속성을 만들어 키와 CID가 모두 있을 때만 `True`를 반환합니다.
3. 공개 설정 API에서는 `kakaoPayConfigured`만 반환하고 Secret key는 반환하지 않습니다.

확인 질문: `/api/config` 응답에 Secret key가 포함되어 있지 않은가?

## 3. 카카오페이 API 클라이언트 만들기

연습 파일: `mobility_service/kakaopay.py`

`KakaoPayClient`에 다음 두 메서드를 직접 작성해 보세요.

```python
async def ready(self, *, partner_order_id, partner_user_id, item_name,
                quantity, total_amount, approval_url, cancel_url, fail_url):
    ...

async def approve(self, *, tid, partner_order_id, partner_user_id, pg_token):
    ...
```

요청 헤더는 서버 안에서만 생성합니다.

```python
{
    "Authorization": f"SECRET_KEY {secret_key}",
    "Content-Type": "application/json",
}
```

공식 API 경로:

- 준비: `POST /online/v1/payment/ready`
- 승인: `POST /online/v1/payment/approve`

에러를 그대로 사용자에게 던지지 말고, HTTP 상태와 카카오페이 오류 메시지를 정리한
전용 예외로 바꾸는 것도 백엔드의 역할입니다.

## 4. 결제 상태 저장하기

연습 파일: `mobility_service/store.py`

다음 상태를 저장할 SQLite 테이블을 설계합니다.

- 우리 결제 ID
- 카카오페이 TID
- 사용자 ID
- 결제 금액
- 배송 주문 원본 JSON
- 현재 상태: `CREATED`, `READY`, `APPROVED`, `COMPLETED`, `FAILED`
- ready/approve 응답과 오류

TID와 주문번호에는 `UNIQUE` 제약을 둡니다. 같은 성공 콜백이 두 번 와도 결제 승인과
배송 접수가 중복 실행되지 않도록 상태를 먼저 확인해야 합니다.

## 5. 백엔드 라우트 연결하기

연습 파일: `mobility_service/app.py`

### 결제 준비

`POST /api/payments/kakaopay/ready`

1. 로그인 사용자인지 확인합니다.
2. 전달받은 배송 주문으로 배송 견적 API를 다시 호출합니다.
3. 서버가 확인한 금액으로 결제 레코드를 만듭니다.
4. 카카오페이 ready API를 호출합니다.
5. TID를 저장하고 결제창 URL만 브라우저에 반환합니다.

### 결제 성공 콜백

`GET /api/payments/kakaopay/{payment_id}/success?pg_token=...`

1. 저장된 결제와 TID를 찾습니다.
2. pg_token과 저장된 값을 사용해 approve API를 호출합니다.
3. 승인이 성공하면 배송 주문을 생성합니다.
4. 같은 콜백이 다시 와도 기존 결과를 반환합니다.

결제 승인은 성공했지만 배송 접수가 실패할 수 있습니다. 운영 서비스라면 이 상태를
별도 기록하고 자동 결제 취소 또는 운영자 알림을 반드시 구현해야 합니다.

## 6. 프론트엔드 연결하기

연습 파일: `mobility_service/index.html`

프론트엔드는 Secret key를 전혀 알 필요가 없습니다.

```javascript
const result = await api("/api/payments/kakaopay/ready", {
  method: "POST",
  body: JSON.stringify({order: draft()}),
});
window.location.assign(result.data.nextRedirectPcUrl);
```

결제 완료 후 `/order?payment=success&orderId=...`로 돌아오면 결과 메시지를 보여줍니다.

## 7. 직접 풀어볼 테스트

연습 파일: `tests/test_mobility_service.py`

외부 결제 API에 실제 요청을 보내지 말고 가짜 클라이언트를 주입해 아래를 검증하세요.

1. 브라우저가 보낸 임의 금액이 아니라 배송 서버의 12,000원이 ready에 전달되는가?
2. ready 응답의 TID가 DB에 저장되는가?
3. pg_token이 approve 요청에 전달되는가?
4. approve 성공 후에만 배송 주문 생성 횟수가 1이 되는가?
5. 동일 성공 콜백을 두 번 호출해도 중복 결제되지 않는가?

실행:

```bash
uv run python -m unittest tests.test_mobility_service.KakaoPayFlowTests -v
```

## 8. 로컬에서 실제 테스트하기

1. 노출된 키를 개발자센터에서 재발급합니다.
2. 애플리케이션의 `사용 API`에서 온라인 결제를 활성화합니다.
3. 새 `Secret key(dev)`를 `.env`에 넣습니다.
4. PC에서만 시험하면 로컬 주소로도 가능하지만, 휴대폰 카카오톡에서 결제하려면
   ngrok 같은 도구로 공개 HTTPS 주소를 만든 뒤 Redirect 주소로 설정합니다.
5. 서버를 실행하고 단일 퀵 견적을 낸 뒤 결제 단계에서 카카오페이를 선택합니다.

테스트 CID `TC0ONETIME`과 개발용 키는 연습용입니다. 실제 운영 결제를 받으려면
사업자 등록, 비즈앱 전환, 온라인 가맹점 제휴와 운영 CID 권한이 별도로 필요합니다.
