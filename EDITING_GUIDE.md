# MOVB 직접 수정 가이드

이 문서는 MOVB 화면과 FastAPI 기능을 직접 수정하고, 로컬에서 확인한 뒤
GitHub와 Render에 반영하는 방법을 설명합니다.

## 1. 어떤 파일을 수정해야 하나요?

| 바꾸려는 항목 | 수정할 파일 |
| --- | --- |
| 퀵 접수, 지도, 이용 내역, 로그인, 채팅 화면 | `mobility_service/index.html` |
| 택시 합승 접수 화면 | `mobility_service/taxi.html` |
| 기능 소개 화면 | `pool-feature-diagram.html` |
| 관리자 화면 | `mobility_service/admin.html` |
| API 주소와 처리 로직 | `mobility_service/app.py` |
| 카카오모빌리티 API 요청 | `mobility_service/client.py` |
| 로그인과 회원 데이터 저장 | `mobility_service/user_store.py` |
| 묶음 퀵 계산 | `mobility_service/bundle.py` |
| 택시 합승 계산 | `mobility_service/rideshare.py` |
| 환경변수 읽기 | `mobility_service/config.py` |
| 자동 테스트 | `tests/test_mobility_service.py` |

현재 화면은 별도의 프런트엔드 프로젝트가 아닙니다. 각 HTML 파일 안에
화면 구조(HTML), 디자인(CSS), 동작(JavaScript)이 함께 들어 있습니다.

## 2. 편집기로 프로젝트 열기

VS Code가 설치되어 있고 `code` 명령을 사용할 수 있다면 프로젝트 폴더에서
다음을 실행합니다.

```bash
code .
```

명령이 동작하지 않으면 VS Code에서 **File → Open Folder**를 누르고
`ChatBot` 폴더를 선택하면 됩니다.

왼쪽 파일 목록에서 위 표에 있는 파일을 열고 수정한 뒤 저장합니다.
처음에는 `mobility_service/index.html`에서 화면 문구나 색상처럼 작은 부분부터
바꾸는 것이 좋습니다.

## 3. 로컬 서버 실행하기

프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
uv sync
uv run uvicorn mobility_service.app:app --reload --port 8002
```

브라우저에서 <http://127.0.0.1:8002>를 엽니다. `--reload`가 있으므로 파일을
저장하면 서버가 변경을 감지합니다. 화면이 그대로라면 브라우저에서
`Command + Shift + R`을 눌러 캐시 없이 새로고침합니다.

서버를 종료할 때는 실행 중인 터미널에서 `Control + C`를 누릅니다.

## 4. 화면을 수정하는 기본 방법

### 문구 바꾸기

`mobility_service/index.html`에서 바꾸려는 문구를 검색합니다. 예를 들어
`퀵 접수하기`를 검색한 뒤 태그 사이의 글자만 수정합니다.

```html
<h2>퀵 접수하기</h2>
```

태그의 `<`, `>`, `/`는 지우지 말고 글자만 바꾸는 것이 안전합니다.

### 색상 바꾸기

각 HTML 파일 위쪽의 `<style>` 안에 있는 `:root`가 주요 색상을 관리합니다.

```css
:root {
  --yellow: #fee500;
  --ink: #191919;
  --muted: #686868;
}
```

같은 색상을 여러 곳에서 바꾸려면 개별 요소보다 이 변수를 수정합니다.
글자와 배경의 대비가 너무 낮아지지 않도록 주의합니다.

### 여백과 크기 바꾸기

CSS에서 다음 속성을 주로 사용합니다.

- `padding`: 요소 안쪽 여백
- `margin`: 요소 바깥 여백
- `font-size`: 글자 크기
- `width`, `height`: 너비와 높이
- `border-radius`: 모서리 둥글기
- `background`, `color`: 배경색과 글자색

입력창과 버튼은 모바일 사용을 위해 높이 `44px` 이상을 유지하는 것이
좋습니다. 입력창 글자는 iPhone의 자동 확대를 피하기 위해 `16px`을
유지하세요.

### 동작 바꾸기

HTML 파일 아래쪽의 `<script>` 영역이 버튼 클릭, 지도, 접수 폼과 같은
브라우저 동작을 담당합니다. 특정 버튼의 `id`를 먼저 찾고, 같은 `id`를
사용하는 JavaScript 코드를 검색하면 연결된 동작을 찾을 수 있습니다.

```html
<button id="quoteButton">예상 요금 확인</button>
```

```javascript
document.getElementById("quoteButton")
```

HTML의 `id`를 바꾸면 JavaScript에서도 같은 이름을 모두 바꿔야 합니다.

## 5. 백엔드 API 수정하기

API 주소와 요청 처리는 주로 `mobility_service/app.py`에 있습니다.

```python
@application.get("/health")
async def health():
    return {"status": "ok"}
```

- `get`, `post`, `put`, `delete`는 요청 방식입니다.
- `"/health"`는 브라우저나 앱에서 호출할 주소입니다.
- 함수의 반환값이 JSON 응답으로 전달됩니다.

요청 데이터 형식을 변경할 때는 `mobility_service/models.py`도 함께 확인해야
합니다. 카카오모빌리티로 보내는 요청 자체를 바꾸려면
`mobility_service/client.py`를 확인합니다.

실제 주문 API는 Sandbox라도 외부 요청이 발생할 수 있으므로, 먼저 자동
테스트에서 Mock 요청으로 검증하세요.

## 6. 수정 후 확인하기

화면을 직접 확인한 뒤 자동 테스트를 실행합니다.

```bash
uv run python -m unittest discover -s tests -q
```

변경된 파일만 확인합니다.

```bash
git status
git diff
```

`.env`, API 키, 관리자 비밀번호, 실제 이름·전화번호가 변경 내용에 들어가면
커밋하지 마세요.

## 7. GitHub와 Render에 반영하기

수정한 파일을 지정해서 커밋합니다.

```bash
git add 수정한_파일
git commit -m "fix: 수정한 내용 요약"
git push origin main
```

예를 들어 메인 화면만 수정했다면 다음처럼 실행합니다.

```bash
git add mobility_service/index.html
git commit -m "fix: improve quick request screen"
git push origin main
```

Render가 GitHub의 `main` 브랜치와 연결되어 있으면 푸시 후 자동 배포가
시작됩니다. Render Dashboard의 **Events**에서 배포 성공 여부를 확인하고,
배포가 끝난 다음 운영 주소에서 `Command + Shift + R`로 새로고침합니다.

## 8. 실수했을 때

아직 커밋하지 않은 변경은 바로 삭제하지 말고 먼저 차이를 확인합니다.

```bash
git diff
```

이미 푸시한 코드를 되돌려야 한다면 이전 커밋을 강제로 삭제하지 말고
되돌림 커밋을 만듭니다.

```bash
git log --oneline -5
git revert 되돌릴_커밋_ID
git push origin main
```

`git reset --hard`는 저장하지 않은 작업까지 지울 수 있으므로 사용하지 않는
것이 좋습니다.

## 추천 수정 순서

1. 문구 또는 색상 한 곳만 수정
2. 파일 저장
3. 로컬 화면에서 데스크톱과 모바일 너비 확인
4. 자동 테스트 실행
5. `git diff`로 API 키와 개인정보가 없는지 확인
6. 커밋하고 푸시
7. Render 배포 완료 후 운영 화면 확인

