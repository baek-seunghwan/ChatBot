# 📝 RAG 자동 평가 CLI: 평가셋 질문을 챗봇에 넣고 → 채점 → pass/fail → 개선 진단
#
# 실행:
#   uv run python -m scripts.evaluate_rag                     # 기본 설정으로 평가
#   uv run python -m scripts.evaluate_rag --top-k 5           # 다른 설정으로 재평가
#   uv run python -m scripts.evaluate_rag --prompt-version v1 --reranker
#   uv run python -m scripts.evaluate_rag --no-llm-judge      # API 없이 규칙 기반 채점
#
# 생성 파일:
#   eval/results/latest.json   전체 결과 (설정 + 항목별 + 집계 + 진단)
#   eval/results/latest.csv    항목별 결과 표
#   eval/results/summary.md    사람이 읽는 요약 (pass/fail + 개선 제안)
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from chatbot.config import (
    COLLECTION_NAME,
    MIN_RELEVANCE_SCORE,
    PROMPT_VERSION,
    REPO_ROOT,
    TOP_K,
    USE_RERANKER,
)
from chatbot.providers import LLMRouter
from chatbot.rag_chain import RagChain

from .rag_eval_core import (
    METRIC_KEYS,
    SCALE5_KEYS,
    aggregate,
    check_targets,
    context_precision,
    context_recall,
    diagnose,
    is_critical_hallucination,
    judge_with_llm,
    judge_with_rules,
    load_dataset,
    load_targets,
)

EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"


def evaluate(
    dataset_path: Path,
    targets_path: Path,
    top_k: int = TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
    collection_name: str = COLLECTION_NAME,
    prompt_version: str = PROMPT_VERSION,
    use_reranker: bool = USE_RERANKER,
    use_llm_judge: bool | None = None,
    limit: int | None = None,
    quiet: bool = False,
) -> dict:
    """평가 전체를 실행하고 결과 dict를 돌려준다. (실험 스크립트도 이 함수를 재사용)"""
    items = load_dataset(dataset_path)[: limit or None]
    targets_config = load_targets(targets_path)
    if use_llm_judge is None:
        use_llm_judge = targets_config.get("judge", {}).get("use_llm", True)

    config = {
        "top_k": top_k,
        "min_score": min_score,
        "collection_name": collection_name,
        "prompt_version": prompt_version,
        "use_reranker": use_reranker,
        "judge": "llm" if use_llm_judge else "rules",
        "dataset": str(dataset_path),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    chain = RagChain(
        top_k=top_k,
        min_score=min_score,
        collection_name=collection_name,
        prompt_version=prompt_version,
        use_reranker=use_reranker,
    )
    router = LLMRouter() if use_llm_judge else None
    log = (lambda *a: None) if quiet else print

    results = []
    for i, item in enumerate(items, start=1):
        log(f"[{i}/{len(items)}] ({item['category']}/{item['difficulty']}) {item['question']}")
        started = time.perf_counter()
        answer = chain.ask(item["question"])
        latency = time.perf_counter() - started

        contexts = [c.text for c in answer.chunks]
        contexts_joined = "\n".join(contexts)
        # 📝 검색 지표는 규칙 기반으로 항상 계산 (필터를 통과한 청크의 출처 기준)
        retrieved_sources = [c.source for c in answer.chunks]
        precision = context_precision(retrieved_sources, item["expected_sources"])
        recall = context_recall(retrieved_sources, item["expected_sources"])

        # 📝 생성 지표는 LLM 심사 → 실패 시 규칙 기반 폴백
        judged = None
        if use_llm_judge:
            try:
                judged = judge_with_llm(
                    router, item["question"], item["ground_truth"], contexts_joined, answer.answer
                )
            except Exception as exc:
                log(f"      (LLM 심사 실패 → 규칙 기반 폴백: {type(exc).__name__})")
        if judged is None:
            judged = judge_with_rules(
                item["question"], item["ground_truth"], contexts_joined, answer.answer
            )

        scores = {
            "faithfulness": judged["faithfulness"],
            "answer_relevancy": judged["answer_relevancy"],
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
            "answer_correctness": judged["answer_correctness"],
        }
        critical = is_critical_hallucination(item, answer.answer, judged["faithfulness"])
        item_avg = sum(scores.values()) / len(scores)
        passed = item_avg >= targets_config["targets"]["overall_avg"] and not critical
        log(f"      평균 {item_avg:.2f} / {'PASS' if passed else 'FAIL'}"
            f"{' [치명적 환각]' if critical else ''}")

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "generated_answer": answer.answer,
                "retrieved_contexts": contexts,
                "used_sources": answer.sources,
                "ground_truth": item["ground_truth"],
                "scores": scores,
                "scale5": judged["scale5"],
                "judge": judged["judge"],
                "critical_hallucination": critical,
                "item_avg": round(item_avg, 3),
                "passed": passed,
                "latency_sec": round(latency, 2),
            }
        )

    agg = aggregate(results)
    checks = check_targets(agg, targets_config)
    diagnosis = diagnose(checks)
    all_passed = all(c["passed"] for c in checks.values())

    return {
        "config": config,
        "aggregate": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()},
        "checks": checks,
        "all_passed": all_passed,
        "diagnosis": diagnosis,
        "results": results,
    }


# ── 결과 저장 ──────────────────────────────
def save_outputs(report: dict, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) JSON (전체)
    (results_dir / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) CSV (항목별 표, 엑셀에서 한글 안 깨지게 utf-8-sig)
    with (results_dir / "latest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "id", "category", "difficulty", "question", "generated_answer",
            "used_sources", "ground_truth",
            *METRIC_KEYS, "item_avg", "critical_hallucination", "passed", "latency_sec",
            *SCALE5_KEYS,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report["results"]:
            writer.writerow(
                {
                    "id": r["id"],
                    "category": r["category"],
                    "difficulty": r["difficulty"],
                    "question": r["question"],
                    "generated_answer": r["generated_answer"].replace("\n", " "),
                    "used_sources": ";".join(r["used_sources"]),
                    "ground_truth": r["ground_truth"],
                    **r["scores"],
                    "item_avg": r["item_avg"],
                    "critical_hallucination": r["critical_hallucination"],
                    "passed": r["passed"],
                    "latency_sec": r["latency_sec"],
                    **r["scale5"],
                }
            )

    # 3) summary.md (사람이 읽는 요약)
    lines = [
        "# RAG 평가 요약",
        "",
        f"- 평가 시각: {report['config']['evaluated_at']}",
        f"- 설정: top_k={report['config']['top_k']}, min_score={report['config']['min_score']}, "
        f"prompt={report['config']['prompt_version']}, reranker={report['config']['use_reranker']}, "
        f"judge={report['config']['judge']}",
        f"- 전체 결과: {'✅ 전체 목표 달성' if report['all_passed'] else '❌ 목표 미달 지표 있음'}",
        "",
        "## 지표별 결과",
        "",
        "| 지표 | 결과 | 목표 | 달성 |",
        "|---|---|---|---|",
    ]
    for metric, check in report["checks"].items():
        mark = "✅" if check["passed"] else "❌"
        lines.append(f"| {metric} | {check['value']} | {check['target']} | {mark} |")

    lines += ["", "## 5점 척도 평균", ""]
    for key, value in report["aggregate"]["scale5_avg"].items():
        lines.append(f"- {key}: {value} / 5")

    if report["diagnosis"]:
        lines += ["", "## 개선 진단 (목표 미달 지표)", ""]
        for d in report["diagnosis"]:
            lines.append(f"### {d['metric']}: {d['value']} < 목표 {d['target']}")
            lines.append("")
            for s in d["suggestions"]:
                lines.append(f"- {s}")
            lines.append("")

    failed = [r for r in report["results"] if not r["passed"]]
    if failed:
        lines += ["", "## FAIL 항목", ""]
        for r in failed:
            reason = "치명적 환각" if r["critical_hallucination"] else f"평균 {r['item_avg']}"
            lines.append(f"- {r['id']} ({reason}): {r['question']}")

    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 챗봇 자동 평가")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--targets", type=Path, default=EVAL_DIR / "targets.json")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-score", type=float, default=MIN_RELEVANCE_SCORE)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, choices=["v1", "v2"])
    parser.add_argument("--reranker", action="store_true", default=USE_RERANKER)
    parser.add_argument("--no-llm-judge", action="store_true", help="API 없이 규칙 기반 채점")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개만 평가 (빠른 확인용)")
    args = parser.parse_args()

    report = evaluate(
        dataset_path=args.dataset,
        targets_path=args.targets,
        top_k=args.top_k,
        min_score=args.min_score,
        collection_name=args.collection,
        prompt_version=args.prompt_version,
        use_reranker=args.reranker,
        use_llm_judge=False if args.no_llm_judge else None,
        limit=args.limit,
    )
    save_outputs(report)

    print("\n===== 평가 요약 =====")
    for metric, check in report["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"{metric:32s} {check['value']:>6} / 목표 {check['target']:<5} [{mark}]")
    print(f"\n전체: {'목표 달성 ✅' if report['all_passed'] else '목표 미달 ❌'}")
    if report["diagnosis"]:
        print("\n===== 개선 제안 =====")
        for d in report["diagnosis"]:
            print(f"\n[{d['metric']}] {d['value']} < {d['target']}")
            for s in d["suggestions"][:3]:
                print(f"  - {s}")
    print(f"\n결과 저장: {RESULTS_DIR}/latest.json, latest.csv, summary.md")


if __name__ == "__main__":
    main()
