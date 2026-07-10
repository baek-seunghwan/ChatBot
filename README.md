# Leon's ChatBot: 로그인형 AI·로컬 LLM 챗봇

## 1. 프로젝트 목적

이 프로젝트는 사용자의 질문에 대해 사전에 구축한 문서 벡터DB에서 관련 문서를 검색하고,
검색된 문서를 LLM에 함께 전달하여 근거 기반 답변을 생성하는 RAG 챗봇이다.

핵심은 LLM을 새로 학습시키는 것이 아니라,
내 문서를 잘 검색하고 그 문서를 근거로 정확히 답변하게 만드는 것이다.

- 문서에 답이 있으면: 문서 근거로 답변하고 출처(sources)를 함께 반환한다.
- 문서에 답이 없으면: 추측하지 않고 "문서에서 확인할 수 없습니다"라고 답한다.

## 2. 사용 기술

| 구분 | 기술 |
|---|---|
| 언어 | Python 3.10+ |
| 웹 프레임워크 | FastAPI + uvicorn |
| 벡터DB | ChromaDB (로컬 저장) |
| 임베딩 모델 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| RAG 오케스트레이션 | LangGraph StateGraph |
| LLM | Claude (우선) → Gemini (폴백), `chatbot/providers.py`의 LLMRouter |
| 평가 | 자체 100점 점수표 + LLM 심사자(judge) |

## 3. 전체 구조도

```
[문서 준비 단계 - ingest.py]
rag_docs/ (txt/md/pdf)
    → 청크 분할 (CHUNK_SIZE=300, OVERLAP=50)
    → 임베딩 (sentence-transformers)
    → ChromaDB 저장 (artifacts/chroma_db/)

[문서 질문 단계 - rag_chain.py + local_chat/app.py]
사용자 질문 (POST /api/rag/ask)
    → LangGraph START
    → retrieve 노드: 질문 임베딩 + 벡터DB TOP_K 청크 검색
    → filter_context 노드: 유사도 < 0.35 청크 제거
    → 조건 분기
        → no_answer 노드: 근거 없으면 "문서에서 확인할 수 없습니다" (LLM 호출 안 함)
        → generate 노드: 근거 있으면 문서 + 질문을 LLM에 전달
    → LangGraph END
    → JSON 반환 (answer, sources, confidence, retrieved_chunks)
```

## 4. 폴더 구조

```
ChatBot/
├── chatbot/
│   ├── rag_docs/            # RAG에 넣을 문서 (txt/md/pdf)
│   ├── config.py            # chunk_size, top_k 등 설정값 관리
│   ├── ingest.py            # 문서 읽기 → 청크 → 임베딩 → 벡터DB 저장
│   ├── rag_chain.py         # LangGraph 기반 검색 → 필터 → 조건 분기 → 답변
│   ├── rag_langchain.py     # LangChain 컴포넌트 기반 RAG
│   ├── main.py              # 기존 명령 호환용 앱 진입점
│   ├── eval.py              # 테스트 질문 실행, 점수 계산
│   ├── eval_questions.jsonl # 평가 질문 30개
│   ├── providers.py         # Claude 우선 → Gemini 폴백 LLM 라우터
│   ├── local_chat/          # 메인 로그인형 웹 챗봇 앱
│   ├── model.py, train.py   # 직접 학습한 소형 챗봇 모델
│   └── chatbot.txt, nextword.txt  # 기본 챗봇 학습 데이터
├── artifacts/
│   ├── chroma_db/           # 벡터DB 저장 위치 (ingest 실행 시 생성)
│   └── chatbot.pt           # 직접 학습한 로컬 모델 체크포인트
├── Dockerfile               # 로그인형 웹 챗봇 컨테이너 배포
└── docker-compose.yml       # 로컬 Docker 실행 구성
```

## 5. 실행 방법

```bash
cd ~/Documents/GitHub/ChatBot
uv sync                                  # 의존성 설치

# 1) 문서를 임베딩해서 벡터DB에 저장 (문서를 바꾸면 다시 실행)
uv run python -m chatbot.ingest

# 2) 터미널에서 RAG 질문 테스트
uv run python -m chatbot.rag_chain "RAG가 뭐야?"

# 3) Leon's ChatBot 웹 서버 실행
uv run uvicorn chatbot.local_chat.app:app --reload --port 8001
# 웹 화면: http://127.0.0.1:8001
# API 문서: http://127.0.0.1:8001/docs

# 4) 답변 품질 평가 → eval_result.csv 생성
uv run python -m chatbot.eval
```

API 키는 프로젝트 루트의 `.env`에 둔다.

```bash
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

### LangSmith로 RAG 실행 추적하기 (선택)

LangSmith를 켜면 `rag-chat` trace 안에서 LangGraph 노드,
검색/필터/생성 경로, Claude→Gemini 폴백, 지연시간과 오류를 확인할 수 있다.

1. LangSmith에서 API 키를 만든다.
2. `.env`에 다음 값을 추가한 뒤 서버를 다시 시작한다.

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=leon-rag-chatbot
```

여러 workspace에 연결된 키는 `LANGSMITH_WORKSPACE_ID`를 추가한다. APAC 리전
계정은 `LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com`도 설정한다.
`GET /health`의 `langsmith` 항목에서 활성화 여부를 확인할 수 있다.

주의: 추적을 켜면 사용자 질문, 검색된 문서 청크, LLM 프롬프트와 답변이
LangSmith로 전송될 수 있다. 개인정보나 비공개 문서를 다룰 때는 추적을 끄거나
별도의 마스킹 정책을 먼저 적용한다.

현재 `eval/dataset.jsonl` 20문항을 LangSmith Dataset & Experiments에서 평가:

```bash
uv run python -m scripts.evaluate_rag_langsmith
```

이 평가는 검색 정밀도/재현율, 답변 가능 여부 판단, 규칙 기반 정답 포함률과
문서 기반성을 기록한다. 기존의 같은 이름 LangSmith 데이터셋은 예전 프로젝트
문서 기준이므로, 현재 평가에는 `chatbot-rag-eval-v2-20` 데이터셋을 사용한다.

## 6. 웹 앱과 API

- 웹 화면: `http://127.0.0.1:8001`
- 회원가입: `POST /api/auth/signup`
- 로그인: `POST /api/auth/login`
- 일반 채팅: `POST /api/chat/ask`
- 문서 기반 RAG: `POST /api/rag/ask`
- 상태 확인: `GET /health`

채팅 화면에는 `AI가 답변`과 `로컬 LLM이 답변` 두 모드만 제공한다.

## 6.5 평가 + 개선 + 재평가 자동화 파이프라인

목표 점수에 도달할 때까지 "평가 → 진단 → 설정 변경 → 재평가"를 반복하는 시스템이다.

```bash
# 1) 평가 실행 → 지표 계산 + pass/fail + 개선 진단
uv run python -m scripts.evaluate_rag
uv run python -m scripts.evaluate_rag --no-llm-judge     # API 없이 규칙 기반 채점
uv run python -m scripts.evaluate_rag --top-k 5 --reranker --prompt-version v1  # 다른 설정 재평가

# 2) 설정 조합 실험 → 최적 설정 추천
uv run python -m scripts.run_rag_experiments             # 검색 지표만 (LLM 비용 0)
uv run python -m scripts.run_rag_experiments --full      # 상위 설정은 답변 품질까지 평가

# 3) 인덱스 재생성 (설정 적용 후)
uv run python -m scripts.build_rag_index
```

생성 파일: `eval/results/latest.json`, `latest.csv`, `summary.md`, `experiments.csv`, `best_config.json`

지표 (전부 0~1, `eval/targets.json`에서 목표치 수정 가능):

| 지표 | 의미 | 기본 목표 |
|---|---|---|
| faithfulness | 답변이 검색 문서에 근거하는가 | ≥ 0.85 |
| answer_relevancy | 질문에 직접적으로 답하는가 | ≥ 0.80 |
| context_precision | 검색 청크 중 관련 문서 비율 | ≥ 0.75 |
| context_recall | 필요한 근거 문서를 찾아왔는가 | ≥ 0.75 |
| answer_correctness | 기준 정답과 맞는가 | ≥ 0.75 |
| overall_avg | 위 5개 평균 | ≥ 0.80 |
| critical_hallucination_rate | 치명적 환각 비율 | ≤ 0.10 |

평가셋은 `eval/dataset.jsonl`에 한 줄씩 추가하면 된다
(id, question, ground_truth, expected_sources, category, difficulty).
목표 미달 지표가 있으면 `summary.md`에 원인별 개선 제안이 자동으로 붙는다.

## 6.6 로컬 LLM (직접 학습 모델) 답변 개선

로컬 모델은 원래 "문장 이어쓰기(자동완성)"용이라 질문에 답하지 못했다.
이를 "질문 → 답변" 모델로 바꾸는 파이프라인:

```bash
# 1) QA 학습 데이터 생성 (chatbot/qa_pairs.jsonl → qa_corpus.txt)
uv run python -m scripts.build_qa_corpus

# 2) QA 형식으로 재학습 (GPU면 Colab 권장, CPU도 가능하지만 느림)
uv run python -m chatbot.train --corpus chatbot/qa_corpus.txt \
    --epochs 60 --max-steps 0 --block-size 256 --embedding-dim 256 --num-heads 8 --num-layers 6
```

답변 순서: ① 학습한 질문과 유사도 매칭 → 즉시 정답 반환(재학습 없이도 동작)
② QA 형식(`질문: X 답변:`)으로 모델 생성 ③ 실패 시 솔직한 안내 메시지.
새 질문을 가르치려면 `chatbot/qa_pairs.jsonl`에 추가하면 된다 (매칭은 즉시 반영, 생성은 재학습 필요).

## 7. 평가 방법

`chatbot/eval_questions.jsonl`의 30개 질문으로 평가한다.
(문서에 답이 있는 질문 10개, 표현을 바꾼 질문 10개, 복합 질문 5개, 문서에 없는 질문 5개)

자체 점수표 (100점 만점, LLM 심사자가 채점):

| 항목 | 배점 | 내용 |
|---|---|---|
| 관련성 | 30 | 질문에 맞게 답했는가 |
| 정확성 | 30 | 답변 내용이 맞는가 |
| 문서 기반성 | 20 | 검색된 문서에 근거했는가 |
| 자연스러움 | 10 | 한국어 문장이 자연스러운가 |
| 완성도 | 10 | 충분히 설명했는가 |

**오차율 = 100 − 최종 점수**

추가 지표: 검색 성공률@3 (기대 문서가 검색 결과에 포함된 비율),
환각률 (문서에 없는 질문에 지어내서 답한 비율), 응답속도 P95.

## 8. 평가 결과

`uv run python -m chatbot.eval` 실행 후 아래 표를 채운다. (결과: `eval_result.csv`)

| 지표 | 목표 | 결과 |
|---|---|---|
| 평균 RAG 점수 | 80점 이상 | (실행 후 기입) |
| 평균 오차율 | 20% 이하 | (실행 후 기입) |
| 환각률 | 10% 이하 | (실행 후 기입) |
| 검색 성공률@3 | 85% 이상 | (실행 후 기입) |
| 응답속도 P95 | 5초 이하 | (실행 후 기입) |

## 9. 한계점

1. 문서가 5개(학습 노트 수준)로 적어 다양한 질문에 대응하기 어렵다.
2. 청크 분할이 글자 수 기준이라 문장 중간에서 잘릴 수 있다.
3. 유사도 임계값(0.35) 하나로 답변 가능 여부를 판단하므로 경계 질문에서 오판할 수 있다.
4. 기본 `/chat` RAG는 직접 구현 체인이고, LangChain 버전은 별도 모듈(`rag_langchain.py`)로 분리되어 있다.
5. LLM 심사자 채점은 실행할 때마다 점수가 약간 달라질 수 있다.

## 10. 남은 보완 방향

1. Hybrid Search 적용
2. Parent-Child Chunking 적용
3. RAGAS 기반 자동 평가 고도화
4. 임계값과 TOP_K를 별도 테스트셋으로 검증
5. Docker 이미지 빌드와 웹·API 헬스체크 자동화

---

## 메인 웹 앱: 로그인형 챗봇 + 직접 학습 모델

로그인 후 하나의 AI 답변을 보여주는 FastAPI 앱. RAG 엔드포인트(`POST /api/rag/ask`)도 포함한다.

- 웹 화면: `http://127.0.0.1:8001` / API 문서: `/docs`
- 회원가입 `POST /api/auth/signup`, 로그인 `POST /api/auth/login`
- 일반 챗봇 `POST /api/chat/ask`, RAG 챗봇 `POST /api/rag/ask`

직접 학습한 소형 모델은 [colab_train.ipynb](colab_train.ipynb)로 학습한다.
`embedding_dim=384`, `num_layers=8`, `num_heads=8`, `block_size=256` 설정이며,
학습 결과(`/content/chatbot.pt`)는 `artifacts/chatbot.pt`와 교체한다.

---

## 11. 선택 기능 (LangChain / 배포)

### 11-1. LangChain 기반 RAG (`chatbot/rag_langchain.py`)

직접 구현 RAG와 동일한 흐름을 LangChain 표준 컴포넌트로 재구성한 버전.
TextLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings → Chroma → LCEL 체인.

```bash
uv run python -m chatbot.rag_langchain --ingest   # 인덱스 생성 (컬렉션: rag_docs_langchain)
uv run python -m chatbot.rag_langchain "RAG가 뭐야?"
```

### 11-2. Docker 배포

```bash
docker compose up --build      # http://localhost:8001
```

API 키는 `.env`에서 주입되고, 벡터DB(`artifacts/chroma_db`)는 볼륨으로 연결된다.

### 11-3. 의존성 설치

```bash
uv sync
```
