# MOVB 로컬 챗봇 vLLM·Docker·EC2 실행

## 지금 구현된 범위

MOVB의 로컬 채팅은 다음 순서로 엔진을 선택합니다.

1. vLLM 공개 가중치 모델
2. Ollama
3. 서버가 없어도 동작하는 자체 QA 검색

vLLM은 Apache-2.0 공개 가중치 모델인 `Qwen/Qwen3-4B-Instruct-2507`를
기본 모델로 사용하며 OpenAI 호환
`/v1/chat/completions` API로 연결됩니다. 모델 서버와 MOVB 앱은
`docker-compose.vllm.yml`로 함께 실행할 수 있습니다.

현재 개발 컴퓨터는 Apple Silicon Mac이므로 이 저장소의 NVIDIA GPU용 Compose
구성을 직접 실행할 수 없습니다. Mac에서는 vLLM-Metal을 별도로 설치하거나,
아래 구성대로 NVIDIA GPU가 있는 Linux·EC2에서 실행해야 합니다.

## NVIDIA GPU Linux에서 실행

`.env.example`을 참고해 `.env`에 최소한 다음 값을 설정합니다.

```dotenv
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
VLLM_MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
VLLM_API_KEY=충분히-긴-임의의-값
VLLM_MAX_MODEL_LEN=4096
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_DTYPE=half
```

그다음 앱과 vLLM을 함께 실행합니다.

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.vllm.yml \
  up -d
```

첫 실행은 공개 모델 가중치를 내려받으므로 오래 걸릴 수 있습니다.

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.vllm.yml \
  logs -f vllm
```

준비 상태는 MOVB 서버를 통해 확인합니다.

```bash
curl http://127.0.0.1:8002/api/vllm/status
```

응답의 `available`이 `true`가 되면 웹의 `로컬` 채팅이 vLLM을 자동 선택합니다.
vLLM 포트는 `127.0.0.1:8000`에만 바인딩되어 외부 인터넷에 직접 노출되지
않습니다.

## EC2 배포

권장 시작 사양은 다음과 같습니다.

- 인스턴스: `g4dn.xlarge`(NVIDIA T4 16 GiB) 이상
- AMI: NVIDIA 드라이버와 Docker GPU 런타임이 포함된 최신 AWS Deep Learning
  Base GPU AMI
- 디스크: 모델 캐시를 고려해 gp3 80~100 GiB 이상
- 보안 그룹: SSH는 관리자 IP만, 서비스 포트는 필요한 프록시·사용자만 허용
- 금지: vLLM의 8000 포트를 인터넷 전체에 공개

배포 순서는 다음과 같습니다.

1. GPU EC2와 Deep Learning Base GPU AMI를 선택합니다.
2. `nvidia-smi`와 `docker compose version`으로 GPU·Docker를 확인합니다.
3. 저장소를 복제하고 `.env`에 카카오 키와 `VLLM_API_KEY`를 설정합니다.
4. 위의 두 Compose 파일을 겹쳐 실행합니다.
5. `/api/vllm/status`와 웹의 로컬 채팅을 확인합니다.
6. 운영 환경에서는 8002 앞에 HTTPS 리버스 프록시나 로드 밸런서를 둡니다.

이 저장소는 EC2에서 바로 실행할 컨테이너 구성까지 포함하지만, 실제 인스턴스
생성은 AWS 계정·리전·결제 권한이 필요한 외부 작업이므로 자동 수행하지 않습니다.

## 프론트 정보와 “학습 데이터”의 차이

현재 로컬 챗봇은 프론트 HTML 전체를 매 요청마다 읽지 않습니다.

- 공개 서비스 설명: `mobility_service/knowledge/*.md`에서 관련 문단을 검색해
  vLLM/Ollama 프롬프트에 넣습니다.
- 홈페이지 공개 문구: `scripts/crawl_movb_site.py`가
  `mobility_service/knowledge/06-homepage-crawl.md` 스냅샷을 만듭니다.
- 현재 주문 화면 값: 주소·이름·연락처·물품 등 허용된 폼 필드만 요청과 함께
  전달해 로컬 모델의 현재 문맥으로 사용합니다.
- 자체 QA: `mobility_service/local_chat_qa.jsonl`을 문자 유사도로 검색해
  저장된 답을 반환합니다.

`VLLM_BASE_URL`이 다른 컴퓨터나 EC2를 가리키면 허용된 폼 값도 그 모델 서버로
전송됩니다. 본인이 관리하는 HTTPS 서버만 연결하고, 인터넷에 공개된 신뢰할 수
없는 vLLM 주소는 사용하지 마세요. 자체 QA 모드는 폼 값을 외부로 보내지 않습니다.

즉, `local_chat_qa.jsonl`과 `knowledge/*.md`는 모델 가중치를 바꾸는
파인튜닝 데이터가 아닙니다. 현재 방식은 공개 가중치 모델에 필요한 MOVB 정보를
요청 시 제공하는 RAG 방식입니다. 홈페이지 내용이 바뀌면 서버를 실행한 상태에서
다음 명령으로 지식 스냅샷을 갱신한 뒤 앱을 재시작합니다.

```bash
uv run python scripts/crawl_movb_site.py \
  --base-url http://127.0.0.1:8002
```

서비스 정책처럼 자주 바뀌고 정확한 근거가 필요한 정보는 파인튜닝보다 이 방식이
적합합니다. 말투·응답 형식을 고정할 만큼 충분한 검수 데이터가 쌓인 뒤에만
LoRA/SFT를 별도 단계로 추가하고, 생성된 LoRA 어댑터를 vLLM의
`--enable-lora --lora-modules` 옵션으로 서빙하는 것이 안전합니다.
