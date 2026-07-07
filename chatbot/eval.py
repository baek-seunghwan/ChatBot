# 📝 eval: 테스트 질문 30개 실행 → 점수 계산 → eval_result.csv 저장
# 📝 실행: uv run python -m chatbot.eval
#
# 자체 점수표 (100점 만점):
#   관련성    30점  질문에 맞게 답했는가
#   정확성    30점  답변 내용이 맞는가
#   문서기반성 20점  검색된 문서에 근거했는가
#   자연스러움 10점  한국어 문장이 자연스러운가
#   완성도    10점  답변이 너무 짧거나 끊기지 않았는가
#   오차율 = 100 - 최종 점수
#
# 추가 지표:
#   검색 성공률@K : 기대 문서가 검색 결과에 포함된 비율
#   환각률        : 문서에 없는 질문에 "확인할 수 없습니다"로 답하지 못한 비율
#   응답속도 P95  : 전체 질문 중 상위 95% 지점의 응답 시간
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from .config import REPO_ROOT
from .providers import LLMRouter
from .rag_chain import NO_ANSWER_TEXT, RagChain

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.jsonl"

# 📝 LLM 심사자(judge)에게 주는 평가 지시문
JUDGE_SYSTEM = (
    "당신은 RAG 챗봇 답변을 채점하는 엄격한 평가자입니다. "
    "반드시 JSON 하나만 출력하세요. 다른 텍스트는 쓰지 마세요."
)

JUDGE_PROMPT = """다음 챗봇 답변을 점수표에 따라 채점하세요.

[질문]
{question}

[정답 기준]
{expected}

[검색된 문서]
{context}

[챗봇 답변]
{answer}

[점수표]
- relevance (관련성, 0~30): 질문에 맞게 답했는가. 30=정확히 맞음, 20=대체로 맞음, 10=조금만 관련, 0=무관
- accuracy (정확성, 0~30): 내용이 맞는가. 30=틀린 내용 없음, 20=대부분 맞음, 10=일부 틀림, 0=핵심이 틀림
- groundedness (문서 기반성, 0~20): 검색된 문서에 근거했는가. 20=문서 근거, 10=일부 추측, 0=문서와 무관
- fluency (자연스러움, 0~10): 한국어가 자연스러운가. 10=자연스러움, 7=조금 어색, 3=많이 어색, 0=문장 깨짐
- completeness (완성도, 0~10): 충분히 설명했는가. 10=충분, 7=짧지만 핵심 있음, 3=너무 부족, 0=답변 실패

참고: 문서에 답이 없는 질문에 "문서에서 확인할 수 없습니다"라고 답한 것은 정답이므로 만점을 준다.

다음 형식의 JSON만 출력하세요:
{{"relevance": 0, "accuracy": 0, "groundedness": 0, "fluency": 0, "completeness": 0}}"""

SCORE_LIMITS = {
    "relevance": 30,
    "accuracy": 30,
    "groundedness": 20,
    "fluency": 10,
    "completeness": 10,
}


def load_questions() -> list[dict]:
    """eval_questions.jsonl에서 질문 목록을 읽는다."""
    questions = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def judge_answer(
    router: LLMRouter, question: str, expected: str, context: str, answer: str
) -> dict[str, int]:
    """LLM 심사자가 답변을 채점해서 5개 항목 점수를 돌려준다."""
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, context=context or "(검색 결과 없음)", answer=answer
    )
    result = router.generate(prompt, system=JUDGE_SYSTEM, max_tokens=200, temperature=0.0)
    # 📝 응답에서 JSON 부분만 추출 (앞뒤에 다른 텍스트가 붙어도 동작하게)
    match = re.search(r"\{[^{}]*\}", result.text)
    if not match:
        raise ValueError(f"심사 결과를 JSON으로 읽지 못했습니다: {result.text[:100]}")
    scores = json.loads(match.group())
    return {
        key: max(0, min(int(scores.get(key, 0)), limit))
        for key, limit in SCORE_LIMITS.items()
    }


def is_refusal(answer: str) -> bool:
    """'문서에서 확인할 수 없습니다' 계열의 답변인지 확인한다."""
    return "확인할 수 없" in answer or "찾지 못했" in answer or "찾을 수 없" in answer


def percentile(values: list[float], p: float) -> float:
    """값 목록에서 p번째 백분위수를 계산한다 (예: p=95 → P95)."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * len(ordered)) - 1))
    return ordered[index]


def run_eval(output_path: Path | None = None) -> Path:
    """전체 질문 세트를 평가하고 eval_result.csv로 저장한다."""
    output_path = output_path or REPO_ROOT / "eval_result.csv"
    questions = load_questions()
    chain = RagChain()
    router = LLMRouter()

    rows = []
    latencies: list[float] = []
    retrieval_hits, retrieval_total = 0, 0
    hallucinations, no_doc_total = 0, 0

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        expected_source = item.get("expected_source")
        expected_answer = item.get("expected_answer", "")
        print(f"[{i}/{len(questions)}] ({item['category']}) {question}")

        # 📝 응답 시간 측정
        started = time.perf_counter()
        result = chain.ask(question)
        latency = time.perf_counter() - started
        latencies.append(latency)

        # 📝 검색 성공률@K: 기대 문서가 검색 결과에 들어 있는가
        if expected_source:
            retrieval_total += 1
            if expected_source in result.sources:
                retrieval_hits += 1

        # 📝 환각률: 문서에 없는 질문인데 지어내서 답했는가
        if expected_source is None:
            no_doc_total += 1
            if not is_refusal(result.answer):
                hallucinations += 1

        context = "\n".join(chunk.text for chunk in result.chunks)
        scores = judge_answer(router, question, expected_answer, context, result.answer)
        total = sum(scores.values())
        print(f"      총점 {total}점 / 응답 {latency:.1f}초")

        rows.append(
            {
                "category": item["category"],
                "question": question,
                "answer": result.answer.replace("\n", " "),
                "sources": ";".join(result.sources),
                "confidence": round(result.confidence, 2),
                "retrieved_chunks": result.retrieved_chunks,
                "latency_sec": round(latency, 2),
                **scores,
                "total": total,
                "error_rate": 100 - total,
            }
        )

    # 📝 CSV 저장 (utf-8-sig: 엑셀에서 한글이 깨지지 않게)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 📝 최종 지표 요약
    average = sum(row["total"] for row in rows) / len(rows)
    print("\n===== 평가 요약 =====")
    print(f"평균 RAG 점수 : {average:.1f}점 (목표 80점 이상)")
    print(f"평균 오차율   : {100 - average:.1f}% (목표 20% 이하)")
    if no_doc_total:
        print(f"환각률        : {hallucinations / no_doc_total * 100:.0f}% (목표 10% 이하)")
    if retrieval_total:
        print(f"검색 성공률@{chain.top_k} : {retrieval_hits / retrieval_total * 100:.0f}% (목표 85% 이상)")
    print(f"응답속도 P95  : {percentile(latencies, 95):.1f}초 (목표 5초 이하)")
    print(f"결과 저장     : {output_path}")
    return output_path


if __name__ == "__main__":
    run_eval()
