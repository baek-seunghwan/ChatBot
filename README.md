# FastAPI와 ChromaDB를 활용한 문서 기반 RAG 챗봇

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
| LLM | Claude (우선) → Gemini (폴백), `chatbot/providers.py`의 LLMRouter |
| 평가 | 자체 100점 점수표 + LLM 심사자(judge) |

## 3. 전체 구조도

```
[문서 준비 단계 - ingest.py]
rag_docs/ (txt/md/pdf)
    → 청크 분할 (CHUNK_SIZE=300, OVERLAP=50)
    → 임베딩 (sentence-transformers)
    → ChromaDB 저장 (artifacts/chroma_db/)

[질문 응답 단계 - rag_chain.py + main.py]
사용자 질문 (POST /chat)
    → 질문 임베딩
    → 벡터DB에서 TOP_K=3 청크 검색
    → 유사도 < 0.35 청크 제거
    → 근거 없으면: "문서에서 확인할 수 없습니다" (LLM 호출 안 함)
    → 근거 있으면: 문서 + 질문을 LLM에 전달 → 근거 기반 답변
    → JSON 반환 (answer, sources, confidence, retrieved_chunks)
```

## 4. 폴더 구조

```
ChatBot/
├── chatbot/
│   ├── rag_docs/            # RAG에 넣을 문서 (txt/md/pdf)
│   ├── config.py            # chunk_size, top_k 등 설정값 관리
│   ├── ingest.py            # 문서 읽기 → 청크 → 임베딩 → 벡터DB 저장
│   ├── rag_chain.py         # 검색 → 프롬프트 구성 → LLM 호출 → 답변
│   ├── main.py              # RAG 전용 FastAPI 서버 (/chat)
│   ├── eval.py              # 테스트 질문 실행, 점수 계산
│   ├── eval_questions.jsonl # 평가 질문 30개
│   ├── providers.py         # Claude 우선 → Gemini 폴백 LLM 라우터
│   ├── local_chat/          # (서브) 로그인형 통합 챗봇 앱 (/api/rag/ask 포함)
│   ├── model.py, train.py   # (서브) 직접 학습한 소형 챗봇 모델
│   └── chatbot.txt, nextword.txt  # (서브) 기본 챗봇 학습 데이터
├── artifacts/
│   ├── chroma_db/           # 벡터DB 저장 위치 (ingest 실행 시 생성)
│   └── chatbot.pt           # 직접 학습한 로컬 모델 체크포인트
├── eval_result.csv          # 평가 결과 (eval 실행 시 생성)
└── REPORT.md                # 프로젝트 보고서
```

## 5. 실행 방법

```bash
cd ~/Documents/GitHub/ChatBot
uv sync                                  # 의존성 설치

# 1) 문서를 임베딩해서 벡터DB에 저장 (문서를 바꾸면 다시 실행)
uv run python -m chatbot.ingest

# 2) 터미널에서 RAG 질문 테스트
uv run python -m chatbot.rag_chain "RAG가 뭐야?"

# 3) RAG API 서버 실행
uv run uvicorn chatbot.main:app --reload --port 8002
# API 문서: http://127.0.0.1:8002/docs

# 4) 답변 품질 평가 → eval_result.csv 생성
uv run python -m chatbot.eval
```

API 키는 프로젝트 루트의 `.env`에 둔다.

```bash
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

## 6. API 사용 예시

```bash
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG가 뭐야?"}'
```

문서에 답이 있는 경우:

```json
{
  "question": "RAG가 뭐야?",
  "answer": "RAG는 Retrieval-Augmented Generation의 약자로, 질문과 관련된 문서를 먼저 검색한 뒤 검색된 내용을 근거로 답변을 생성하는 방식입니다. (참고: rag_basic.txt)",
  "sources": ["rag_basic.txt"],
  "confidence": 0.87,
  "retrieved_chunks": 3,
  "message": "문서 근거 기반으로 답변했습니다."
}
```

문서에 답이 없는 경우:

```json
{
  "question": "알렉스 나이가 몇 살이야?",
  "answer": "문서에서 확인할 수 없습니다.",
  "sources": [],
  "confidence": 0.0,
  "retrieved_chunks": 0,
  "message": "관련 문서를 찾지 못했습니다."
}
```

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
4. LangChain 없이 직접 구현하여 학습에는 좋지만, 컴포넌트 교체 유연성은 떨어진다.
5. LLM 심사자 채점은 실행할 때마다 점수가 약간 달라질 수 있다.

## 10. 개선 방향

1. LangGraph를 활용한 Agentic RAG 적용
2. 질문 재작성 Query Rewriting 적용
3. 검색 결과 관련성 평가 단계 추가
4. Hybrid Search 적용
5. Parent-Child Chunking 적용
6. LangSmith를 통한 실행 과정 추적
7. RAGAS 기반 자동 평가 고도화
8. Qwen/Gemma 기반 로컬 LLM 실험
9. LoRA/QLoRA를 활용한 답변 형식 Fine-Tuning
10. GGUF 변환 후 llama.cpp 추론 최적화

---

## 서브 프로젝트: 로그인형 챗봇 + 직접 학습 모델

로그인 후 하나의 AI 답변을 보여주는 FastAPI 앱. RAG 엔드포인트(`POST /api/rag/ask`)도 포함한다.

```bash
uv run uvicorn chatbot.local_chat.app:app --reload --port 8001
```

- 웹 화면: `http://127.0.0.1:8001` / API 문서: `/docs`
- 회원가입 `POST /api/auth/signup`, 로그인 `POST /api/auth/login`
- 일반 챗봇 `POST /api/chat/ask`, RAG 챗봇 `POST /api/rag/ask`

직접 학습한 소형 모델은 [colab_train.ipynb](colab_train.ipynb)로 학습한다.
`embedding_dim=384`, `num_layers=8`, `num_heads=8`, `block_size=256` 설정이며,
학습 결과(`/content/chatbot.pt`)는 `artifacts/chatbot.pt`와 교체한다.
