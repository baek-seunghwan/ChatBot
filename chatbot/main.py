# 📝 main: RAG 챗봇 전용 FastAPI 서버, /chat API 제공
# 📝 실행: uv run uvicorn chatbot.main:app --reload --port 8002
# 📝 테스트: http://127.0.0.1:8002/docs 에서 /chat 호출
#
# 로그인이 필요한 통합 앱은 chatbot/local_chat/app.py(/api/rag/ask)를 사용하고,
# 이 파일은 과제 제출용 최소 RAG API 서버다.
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent_graph import RagAgent
from .config import langsmith_status
from .model import generate_text, load_checkpoint
from .providers import LLMRouter
from .qa_match import best_match
from .rag_chain import RagChain

app = FastAPI(
    title="문서 기반 RAG 챗봇",
    description="질문과 관련된 문서를 검색하고, 검색된 문서를 근거로 답변을 생성합니다.",
    version="2.0.0",
)

# 📝 첫 질문이 들어올 때 임베딩 모델과 벡터DB를 로드한다(lazy loading).
_chain = RagChain()
_agent = RagAgent()
_router = LLMRouter()

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv("CHATBOT_MODEL_PATH", REPO_ROOT / "artifacts" / "chatbot.pt")
)

ResponderMode = Literal["ai", "local", "qwen", "agent"]

_SYSTEM_PROMPT = (
    "당신은 친절한 개인 비서 챗봇입니다. 사용자의 질문에 한국어로 명확하고 "
    "간결하게 답하세요. 모르는 내용은 추측하지 말고 모른다고 답하세요."
)

_QA_PROMPT_FORMAT = "질문: {q} 답변:"
_LOCAL_FALLBACK_ANSWER = (
    "아직 배우지 못한 질문이에요. 저는 학습한 범위 안에서만 답할 수 있는 작은 모델이라, "
    "이 질문은 AI나 Qwen 모드로 물어봐 주세요!"
)
_GENERATION_MIN_SIMILARITY = 0.30

_TIME_PATTERN = re.compile(r"몇\s*시|지금\s*시간|시간\s*(알려|좀|뭐)")
_DATE_PATTERN = re.compile(r"며칠|몇\s*일이|무슨\s*요일|오늘\s*날짜|날짜\s*(알려|좀|뭐)")
_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# 📝 간단한 HTML 페이지 (브라우저에서 바로 테스트 가능)
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>문서 기반 RAG 챗봇</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #5568d3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .response {
            margin-top: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
            display: none;
        }
        .response.show {
            display: block;
        }
        .answer {
            color: #333;
            line-height: 1.6;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }
        .meta {
            font-size: 13px;
            color: #666;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }
        .error {
            color: #d32f2f;
            padding: 15px;
            background: #ffebee;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        .error.show {
            display: block;
        }
        .loading {
            display: none;
            color: #667eea;
            font-size: 14px;
        }
        .loading.show {
            display: block;
        }
        /* 📝 스피너 글리프: 기호마다 폭이 달라서 고정폭으로 잡아 흔들림 방지 */
        .loading .spin {
            display: inline-block;
            width: 1.2em;
            text-align: center;
        }
        .mode-toggle {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .mode-btn {
            padding: 10px 16px;
            background: #f1f3f8;
            border: 1px solid #dbe0ea;
            border-radius: 999px;
            color: #666;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
        }
        .mode-btn.active {
            color: white;
            background: #667eea;
            border-color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 문서 기반 RAG 챗봇</h1>
        <p class="subtitle">4개 모델 중 하나를 선택해 질문하세요</p>

        <div class="mode-toggle">
            <button class="mode-btn active" data-responder="ai">AI가 답변</button>
            <button class="mode-btn" data-responder="local">로컬 LLM이 답변</button>
            <button class="mode-btn" data-responder="qwen">Qwen이 답변</button>
            <button class="mode-btn" data-responder="agent">Agent가 답변</button>
        </div>

        <div class="input-group">
            <input type="text" id="question" placeholder="질문을 입력하세요... (예: RAG가 뭐야?)" onkeypress="handleKeypress(event)">
            <button onclick="askQuestion()">질문하기</button>
        </div>

        <div class="loading" id="loading"><span class="spin" id="spinGlyph">·</span> 생각 중…</div>
        <div class="error" id="error"></div>
        <div class="response" id="response">
            <div class="answer" id="answer"></div>
            <div class="meta" id="meta"></div>
        </div>
    </div>

    <script>
        let responderMode = 'ai';

        // 📝 생각 중 스피너 (Claude Code 스타일): · ✻ ✽ ✶ ✳ ✢ 를 하나씩 순서대로 반복
        const SPINNER_FRAMES = ['·', '✻', '✽', '✶', '✳', '✢'];
        let spinnerTimer = null;
        let spinnerIndex = 0;

        function showLoading(show) {
            document.getElementById('loading').classList.toggle('show', show);
            if (show && spinnerTimer === null) {
                spinnerTimer = setInterval(() => {
                    spinnerIndex = (spinnerIndex + 1) % SPINNER_FRAMES.length;
                    document.getElementById('spinGlyph').textContent = SPINNER_FRAMES[spinnerIndex];
                }, 120);
            } else if (!show && spinnerTimer !== null) {
                clearInterval(spinnerTimer);
                spinnerTimer = null;
            }
        }

        function setResponderMode(next) {
            const allowed = ['ai', 'local', 'qwen', 'agent'];
            responderMode = allowed.includes(next) ? next : 'ai';
            document.querySelectorAll('.mode-btn').forEach((button) => {
                button.classList.toggle('active', button.dataset.responder === responderMode);
            });
        }

        document.querySelectorAll('.mode-btn').forEach((button) => {
            button.addEventListener('click', () => setResponderMode(button.dataset.responder));
        });

        function handleKeypress(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        }

        async function askQuestion() {
            const question = document.getElementById('question').value.trim();

            if (!question) {
                alert('질문을 입력해주세요');
                return;
            }

            showLoading(true);
            document.getElementById('error').classList.remove('show');
            document.getElementById('response').classList.remove('show');

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, responder: responderMode })
                });

                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}`);
                }

                const data = await res.json();
                showLoading(false);

                if (data.error) {
                    throw new Error(data.error);
                }

                document.getElementById('answer').textContent = data.answer || '(빈 응답)';
                const labelMap = {
                    ai: 'AI',
                    local: '로컬 LLM',
                    qwen: 'Qwen',
                    agent: 'LangGraph Agent'
                };
                document.getElementById('meta').innerHTML = `<strong>응답 모델:</strong> ${labelMap[data.responder] || data.responder}`;
                document.getElementById('response').classList.add('show');
            } catch (err) {
                showLoading(false);
                document.getElementById('error').textContent = '❌ 오류: ' + err.message;
                document.getElementById('error').classList.add('show');
            }
        }

        // 초기 질문 예시
        window.onload = () => {
            document.getElementById('question').value = 'RAG가 뭐야?';
        };
    </script>
</body>
</html>
"""


class AskResponse(BaseModel):
    question: str
    responder: ResponderMode = "ai"
    answer: str | None = None
    error: str | None = None


def _chatbot_call(prompt: str) -> AskResponse:
    try:
        result = _router.generate(
            prompt,
            system=_SYSTEM_PROMPT,
            max_tokens=800,
            temperature=0.5,
        )
        return AskResponse(question=prompt, responder="ai", answer=result.text)
    except Exception as exc:
        return AskResponse(
            question=prompt,
            responder="ai",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


@lru_cache(maxsize=1)
def get_local_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {MODEL_PATH}. 먼저 `python -m chatbot.train`을 실행하세요."
        )
    return load_checkpoint(MODEL_PATH, device="cpu")


def _dynamic_answer(prompt: str) -> str | None:
    now = datetime.now()
    if _TIME_PATTERN.search(prompt):
        return f"지금은 {now.hour}시 {now.minute}분이에요."
    if _DATE_PATTERN.search(prompt):
        weekday = _WEEKDAYS[now.weekday()]
        return f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일이에요."
    return None


def _local_llm_call(prompt: str) -> AskResponse:
    try:
        dynamic = _dynamic_answer(prompt)
        if dynamic is not None:
            return AskResponse(question=prompt, responder="local", answer=dynamic)

        matched, score = best_match(prompt)
        if matched is not None:
            return AskResponse(question=prompt, responder="local", answer=matched)

        if score < _GENERATION_MIN_SIMILARITY:
            return AskResponse(
                question=prompt,
                responder="local",
                answer=_LOCAL_FALLBACK_ANSWER,
            )

        model, tokenizer, config, metadata = get_local_model_bundle()
        if "qa" not in str(metadata.get("corpus", "")):
            return AskResponse(
                question=prompt,
                responder="local",
                answer=_LOCAL_FALLBACK_ANSWER,
            )

        qa_prompt = _QA_PROMPT_FORMAT.format(q=prompt.strip())
        generated = generate_text(
            model,
            tokenizer,
            config,
            qa_prompt,
            max_new_tokens=120,
            temperature=0.5,
            top_k=10,
            device="cpu",
        )
        answer = generated[len(qa_prompt):].strip() if generated.startswith(qa_prompt) else ""
        if len(answer) < 5 or answer in prompt:
            answer = _LOCAL_FALLBACK_ANSWER
        return AskResponse(question=prompt, responder="local", answer=answer)
    except Exception as exc:
        return AskResponse(
            question=prompt,
            responder="local",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


def _qwen_call(prompt: str) -> AskResponse:
    try:
        from .qwen_local import generate_answer

        return AskResponse(question=prompt, responder="qwen", answer=generate_answer(prompt))
    except ImportError:
        return AskResponse(
            question=prompt,
            responder="qwen",
            error="transformers가 설치되지 않았습니다. `uv sync`를 실행하세요.",
        )
    except Exception as exc:
        return AskResponse(
            question=prompt,
            responder="qwen",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


def _agent_call(prompt: str) -> AskResponse:
    try:
        result = _agent.ask(prompt)
        meta = (
            f"\n\n[Agent]\n"
            f"출처: {', '.join(result.sources) if result.sources else '없음'}\n"
            f"재검색: {result.rewrites}회\n"
            f"근거 검증: {'통과' if result.grounded else '미통과'}\n"
            f"실행 경로: {' → '.join(result.trace)}"
        )
        return AskResponse(question=prompt, responder="agent", answer=result.answer + meta)
    except Exception as exc:
        return AskResponse(
            question=prompt,
            responder="agent",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


def _answer_for_mode(prompt: str, responder: ResponderMode) -> AskResponse:
    if responder == "local":
        return _local_llm_call(prompt)
    if responder == "qwen":
        return _qwen_call(prompt)
    if responder == "agent":
        return _agent_call(prompt)
    return _chatbot_call(prompt)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    responder: ResponderMode = "ai"


class ChatResponse(BaseModel):
    """19번 응답 형식"""

    question: str
    answer: str
    sources: list[str]
    confidence: float
    retrieved_chunks: int
    message: str


class AgentResponse(BaseModel):
    """LangGraph Agent 응답: 재검색 횟수, 자기검증 결과, 실행 경로 포함"""

    question: str
    answer: str
    sources: list[str]
    confidence: float
    retrieved_chunks: int
    rewrites: int
    grounded: bool
    trace: list[str]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """웹 UI 홈페이지"""
    return HTML_PAGE


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "docs": "/docs",
        "chat": "POST /chat",
        "agent": "POST /agent",
        "langsmith": langsmith_status(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """질문 → 문서 검색 → 근거 기반 답변 (기본 RAG)"""
    result = await asyncio.to_thread(_chain.ask, request.question)
    return ChatResponse(**result.to_dict())


@app.post("/ask", response_model=AskResponse)
async def ask(request: ChatRequest) -> AskResponse:
    """단일 질문에 대해 4개 응답 모드(ai/local/qwen/agent) 중 하나를 실행한다."""
    result = await asyncio.to_thread(_answer_for_mode, request.question, request.responder)
    return result


@app.post("/agent", response_model=AgentResponse)
async def agent(request: ChatRequest) -> AgentResponse:
    """질문 분석 → 검색 → 문서 평가 → 재검색 → 답변 → 자기검증 (LangGraph Agent)"""
    result = await asyncio.to_thread(_agent.ask, request.question)
    return AgentResponse(**result.to_dict())
