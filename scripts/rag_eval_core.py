# 📝 RAG 평가의 공통 로직: 지표 계산 + LLM 심사 + 규칙 기반 폴백 + 진단
#    evaluate_rag.py와 run_rag_experiments.py가 공유한다.
#
# 지표 (전부 0.0 ~ 1.0, MAE/RMSE 같은 수치 예측 지표는 쓰지 않는다):
#   faithfulness      답변이 검색된 문서에 근거하는가
#   answer_relevancy  질문에 직접적으로 답하는가
#   context_precision 검색된 청크 중 기대 문서에서 나온 비율 (규칙 기반)
#   context_recall    기대 문서를 검색 결과가 포함한 비율 (규칙 기반)
#   answer_correctness 기준 정답(ground_truth)과 비교해 맞는가
from __future__ import annotations

import json
import re
from pathlib import Path

METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

# 사람이 보기 좋은 5점 척도 항목
SCALE5_KEYS = ["관련성", "정확성", "문서기반성", "자연스러움", "완성도"]


# ── 데이터셋 ──────────────────────────────
def load_dataset(path: Path) -> list[dict]:
    """eval/dataset.jsonl을 읽는다. (id, question, ground_truth, expected_sources, category, difficulty)"""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise ValueError(f"평가셋이 비어 있습니다: {path}")
    return items


def load_targets(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── 공통 도우미 ──────────────────────────────
def is_refusal(answer: str) -> bool:
    """'문서에서 확인할 수 없습니다' 계열의 답변인지 확인한다."""
    return bool(re.search(r"확인할 수 없|찾지 못했|찾을 수 없", answer))


def content_words(text: str) -> set[str]:
    """비교용 단어 집합 (2글자 이상 한글/영문/숫자 단어만)."""
    return {w for w in re.findall(r"[가-힣A-Za-z0-9_]+", text) if len(w) >= 2}


def coverage(part: str, whole: str) -> float:
    """part의 단어 중 whole에도 나오는 비율 (0~1)."""
    part_words = content_words(part)
    if not part_words:
        return 0.0
    whole_words = content_words(whole)
    return len(part_words & whole_words) / len(part_words)


# ── 규칙 기반 지표 (검색 품질: LLM 없이 항상 계산) ──────────────────────────────
def context_precision(retrieved_sources: list[str], expected: list[str]) -> float:
    """검색된 청크 중 기대 문서에서 나온 청크의 비율.

    문서에 없는 질문(expected가 빈 리스트)은 아무것도 검색되지 않아야 정답이므로,
    검색 결과가 없으면 1.0, 있으면 0.0으로 계산한다.
    """
    if not expected:
        return 1.0 if not retrieved_sources else 0.0
    if not retrieved_sources:
        return 0.0
    hits = sum(1 for s in retrieved_sources if s in expected)
    return hits / len(retrieved_sources)


def context_recall(retrieved_sources: list[str], expected: list[str]) -> float:
    """기대 문서 중 검색 결과에 실제로 포함된 비율."""
    if not expected:
        return 1.0 if not retrieved_sources else 0.0
    found = sum(1 for s in expected if s in retrieved_sources)
    return found / len(expected)


# ── LLM 심사자 (faithfulness / relevancy / correctness + 5점 척도) ──────────────
JUDGE_SYSTEM = (
    "당신은 RAG 챗봇 답변을 채점하는 엄격한 평가자입니다. "
    "반드시 JSON 하나만 출력하세요. 다른 텍스트는 쓰지 마세요."
)

JUDGE_PROMPT = """다음 RAG 챗봇 답변을 채점하세요.

[질문]
{question}

[기준 정답]
{ground_truth}

[검색된 문서]
{contexts}

[챗봇 답변]
{answer}

[채점 항목] (0.0 ~ 1.0 소수)
- faithfulness: 답변의 모든 내용이 검색된 문서에 근거하는가. 문서에 없는 내용을 지어냈으면 크게 감점.
- answer_relevancy: 답변이 질문에 직접적으로 답하는가. 동문서답이면 0에 가깝게.
- answer_correctness: 기준 정답과 비교했을 때 사실적으로 맞는가.

[5점 척도] (1 ~ 5 정수)
- relevance: 관련성 / accuracy: 정확성 / groundedness: 문서 기반성 / fluency: 자연스러움 / completeness: 완성도

참고: 문서에 답이 없는 질문에 "확인할 수 없습니다"라고 답한 것은 모범 답변이므로 모든 항목 만점.

다음 형식의 JSON만 출력:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": 0.0, "relevance": 1, "accuracy": 1, "groundedness": 1, "fluency": 1, "completeness": 1}}"""


def judge_with_llm(router, question: str, ground_truth: str, contexts: str, answer: str) -> dict:
    """LLM 심사자가 채점. 실패하면 예외를 던진다 (호출부에서 규칙 기반으로 폴백)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        contexts=contexts or "(검색 결과 없음)",
        answer=answer,
    )
    result = router.generate(prompt, system=JUDGE_SYSTEM, max_tokens=250, temperature=0.0)
    match = re.search(r"\{[^{}]*\}", result.text)
    if not match:
        raise ValueError(f"심사 JSON 파싱 실패: {result.text[:100]}")
    raw = json.loads(match.group())
    clamp01 = lambda v: max(0.0, min(1.0, float(v)))
    clamp5 = lambda v: max(1, min(5, int(v)))
    return {
        "faithfulness": clamp01(raw.get("faithfulness", 0)),
        "answer_relevancy": clamp01(raw.get("answer_relevancy", 0)),
        "answer_correctness": clamp01(raw.get("answer_correctness", 0)),
        "scale5": {
            "관련성": clamp5(raw.get("relevance", 1)),
            "정확성": clamp5(raw.get("accuracy", 1)),
            "문서기반성": clamp5(raw.get("groundedness", 1)),
            "자연스러움": clamp5(raw.get("fluency", 1)),
            "완성도": clamp5(raw.get("completeness", 1)),
        },
        "judge": "llm",
    }


def judge_with_rules(question: str, ground_truth: str, contexts: str, answer: str) -> dict:
    """LLM 없이 단어 겹침으로 대략 채점하는 폴백. (API 키가 없어도 파이프라인이 돈다)"""
    refusal = is_refusal(answer)
    expected_refusal = is_refusal(ground_truth)

    if expected_refusal:
        # 문서에 없는 질문: 거절했으면 만점, 지어냈으면 0점대
        base = 1.0 if refusal else 0.0
        faithfulness = relevancy = correctness = base
    else:
        # 답변 단어가 문서에 얼마나 근거하는지 / 정답과 얼마나 겹치는지
        faithfulness = 1.0 if refusal else coverage(answer, contexts)
        relevancy = 0.2 if refusal else min(1.0, coverage(question, answer) + 0.4)
        correctness = 0.0 if refusal else coverage(ground_truth, answer)

    to5 = lambda v: max(1, min(5, round(v * 4) + 1))
    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevancy": round(relevancy, 3),
        "answer_correctness": round(correctness, 3),
        "scale5": {
            "관련성": to5(relevancy),
            "정확성": to5(correctness),
            "문서기반성": to5(faithfulness),
            "자연스러움": 3,  # 규칙 기반으로는 판단 불가 → 중간값
            "완성도": to5((relevancy + correctness) / 2),
        },
        "judge": "rules",
    }


# ── 치명적 환각 판정 ──────────────────────────────
def is_critical_hallucination(item: dict, answer: str, faithfulness: float) -> bool:
    """문서에 없는 질문에 지어내서 답했거나, 근거 점수가 매우 낮으면 치명적 환각."""
    if not item["expected_sources"] and not is_refusal(answer):
        return True
    return faithfulness < 0.3 and not is_refusal(answer)


# ── 집계 + pass/fail ──────────────────────────────
def aggregate(results: list[dict]) -> dict:
    """항목별 결과를 평균 지표로 집계한다."""
    n = len(results)
    avg = {key: sum(r["scores"][key] for r in results) / n for key in METRIC_KEYS}
    avg["overall_avg"] = sum(avg[key] for key in METRIC_KEYS) / len(METRIC_KEYS)
    critical = sum(1 for r in results if r["critical_hallucination"])
    avg["critical_hallucination_rate"] = critical / n
    avg["scale5_avg"] = {
        key: round(sum(r["scale5"][key] for r in results) / n, 2) for key in SCALE5_KEYS
    }
    return avg


def check_targets(agg: dict, targets_config: dict) -> dict[str, dict]:
    """지표별 목표 달성 여부를 돌려준다: {metric: {value, target, passed}}"""
    checks = {}
    for metric, target in targets_config["targets"].items():
        value = agg[metric]
        checks[metric] = {"value": round(value, 3), "target": target, "passed": value >= target}
    rate_max = targets_config["critical_hallucination_rate_max"]
    rate = agg["critical_hallucination_rate"]
    checks["critical_hallucination_rate"] = {
        "value": round(rate, 3),
        "target": rate_max,
        "passed": rate <= rate_max,
    }
    return checks


# ── 개선 자동 진단 ──────────────────────────────
DIAGNOSIS_RULES: dict[str, list[str]] = {
    "context_recall": [
        "chunk_size를 조정해 보세요 (현재 문단이 잘려 정보가 흩어질 수 있음): scripts.run_rag_experiments로 400/700/1000 비교",
        "chunk_overlap을 늘려 청크 경계에서 끊기는 문맥을 보완하세요 (50 → 100)",
        "top_k를 늘려 더 많은 청크를 가져오세요 (3 → 5 또는 8)",
        "문서를 수정했다면 `python -m scripts.build_rag_index`로 인덱스를 재생성하세요",
        "기대 문서(expected_sources)에 해당 내용이 실제로 있는지 rag_docs를 확인하세요",
    ],
    "context_precision": [
        "top_k를 줄여 관련 없는 청크 유입을 줄이세요 (8 → 5 → 3)",
        "reranker를 켜세요: RAG_USE_RERANKER=1 (임베딩+단어 겹침으로 재정렬)",
        "min_score(RAG_MIN_RELEVANCE_SCORE)를 올려 낮은 유사도 청크를 걸러내세요",
        "문서별 metadata filter 도입을 검토하세요 (질문 주제로 source 제한)",
    ],
    "faithfulness": [
        "프롬프트 v2를 사용하세요 (문서에 없으면 모른다고 답변 규칙 강화): RAG_PROMPT_VERSION=v2",
        "출처 기반 답변을 강제하세요 (답변 끝에 참고 문서명 표기 규칙)",
        "temperature를 낮추세요 (현재 0.2, 더 낮게도 가능)",
        "min_score를 올려 근거가 약한 청크로 답변을 만들지 않게 하세요",
    ],
    "answer_relevancy": [
        "query rewrite(질문 재작성) 단계 추가를 검토하세요",
        "질문 의도 분류(개념 질문/비교 질문/방법 질문)를 추가해 프롬프트를 다르게 구성하세요",
        "prompt template을 개선하세요: '답변 첫 문장은 질문에 직접 답한다' 규칙 (v2에 포함됨)",
    ],
    "answer_correctness": [
        "검색 품질(context_precision/recall)이 낮으면 그것부터 고치세요 — 잘못된 문서로는 맞는 답이 안 나옵니다",
        "rag_docs 문서 내용 자체가 정확하고 충분한지 확인하세요",
        "답변 프롬프트에 '문서 일부만 관련 있으면 그 부분만 답한다' 규칙을 유지하세요 (v2 포함)",
    ],
    "overall_avg": [
        "개별 지표 중 가장 낮은 것부터 순서대로 개선하세요",
        "`python -m scripts.run_rag_experiments`로 최적 설정 조합을 찾으세요",
    ],
    "critical_hallucination_rate": [
        "min_score(RAG_MIN_RELEVANCE_SCORE)를 올려 근거 없는 질문에서 검색이 되지 않게 하세요",
        "프롬프트 v2의 '근거 부족 시 확인할 수 없습니다' 규칙을 유지하세요",
        "문서에 없는 질문 유형을 평가셋에 더 추가해 회귀를 감시하세요",
    ],
}


def diagnose(checks: dict[str, dict]) -> list[dict]:
    """목표 미달 지표마다 개선 제안 목록을 만든다."""
    report = []
    for metric, check in checks.items():
        if not check["passed"]:
            report.append(
                {
                    "metric": metric,
                    "value": check["value"],
                    "target": check["target"],
                    "suggestions": DIAGNOSIS_RULES.get(metric, ["개선 규칙이 정의되지 않은 지표입니다."]),
                }
            )
    return report
