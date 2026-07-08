# 📝 RAG 설정 조합 실험 CLI: 어떤 설정이 목표 점수에 가장 가까운지 자동으로 찾는다.
#
# 동작 방식 (비용을 아끼는 2단계):
#   1단계 (기본): 검색 지표만 측정 — LLM 호출 없이 context_precision/recall만 계산.
#      chunk_size × overlap × top_k × reranker 전체 그리드를 싸게 훑는다.
#   2단계 (--full): 1단계 상위 N개 설정만 LLM 심사를 포함한 전체 평가를 돌린다.
#      (프롬프트 버전 비교는 생성 품질 문제라 2단계에서만 의미가 있다)
#
# 실행:
#   uv run python -m scripts.run_rag_experiments                        # 1단계만 (무료)
#   uv run python -m scripts.run_rag_experiments --full                 # 상위 3개는 전체 평가
#   uv run python -m scripts.run_rag_experiments --chunk-sizes 300,700 --top-ks 3,5
#
# 생성 파일:
#   eval/results/experiments.csv   설정별 점수 표
#   eval/results/best_config.json  추천 설정 (+ .env에 넣을 값)
from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime
from pathlib import Path

from chatbot.config import MIN_RELEVANCE_SCORE, REPO_ROOT
from chatbot.ingest import build
from chatbot.rag_chain import RagChain

from .evaluate_rag import evaluate
from .rag_eval_core import (
    context_precision,
    context_recall,
    load_dataset,
    load_targets,
)

EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"


def collection_for(chunk_size: int, overlap: int) -> str:
    return f"rag_docs_cs{chunk_size}_ov{overlap}"


def ensure_index(chunk_size: int, overlap: int, built: set[str]) -> str:
    """해당 청크 설정의 인덱스가 없으면 만들고 컬렉션 이름을 돌려준다."""
    name = collection_for(chunk_size, overlap)
    if name not in built:
        print(f"  인덱스 생성: {name}")
        build(chunk_size=chunk_size, chunk_overlap=overlap, collection_name=name, quiet=True)
        built.add(name)
    return name


def retrieval_score(chain: RagChain, items: list[dict]) -> dict:
    """LLM 호출 없이 검색 품질만 측정한다 (1단계용)."""
    precisions, recalls = [], []
    for item in items:
        chunks = [c for c in chain.retrieve(item["question"]) if c.score >= chain.min_score]
        sources = [c.source for c in chunks]
        precisions.append(context_precision(sources, item["expected_sources"]))
        recalls.append(context_recall(sources, item["expected_sources"]))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return {
        "context_precision": round(precision, 3),
        "context_recall": round(recall, 3),
        "retrieval_avg": round((precision + recall) / 2, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 설정 조합 실험")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--targets", type=Path, default=EVAL_DIR / "targets.json")
    parser.add_argument("--chunk-sizes", default="300,400,700,1000")
    parser.add_argument("--overlaps", default="50,100,150")
    parser.add_argument("--top-ks", default="3,5,8")
    parser.add_argument("--rerankers", default="0,1", help="0=끔, 1=켬")
    parser.add_argument("--prompt-versions", default="v2", help="--full 단계에서 비교 (예: v1,v2)")
    parser.add_argument("--full", action="store_true", help="상위 설정은 LLM 포함 전체 평가")
    parser.add_argument("--full-top-n", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    chunk_sizes = [int(v) for v in args.chunk_sizes.split(",")]
    overlaps = [int(v) for v in args.overlaps.split(",")]
    top_ks = [int(v) for v in args.top_ks.split(",")]
    rerankers = [v == "1" for v in args.rerankers.split(",")]
    prompt_versions = args.prompt_versions.split(",")

    items = load_dataset(args.dataset)[: args.limit or None]
    targets = load_targets(args.targets)["targets"]
    built: set[str] = set()

    # ── 1단계: 검색 지표 그리드 (LLM 없이) ──
    grid = [
        (cs, ov, k, rr)
        for cs, ov, k, rr in itertools.product(chunk_sizes, overlaps, top_ks, rerankers)
        if ov < cs  # overlap이 chunk보다 크면 의미 없음
    ]
    print(f"1단계: 검색 실험 {len(grid)}개 조합 (LLM 호출 없음)\n")
    rows = []
    for i, (cs, ov, k, rr) in enumerate(grid, start=1):
        collection = ensure_index(cs, ov, built)
        chain = RagChain(
            top_k=k,
            min_score=MIN_RELEVANCE_SCORE,
            collection_name=collection,
            use_reranker=rr,
        )
        score = retrieval_score(chain, items)
        row = {"chunk_size": cs, "overlap": ov, "top_k": k, "reranker": rr, **score}
        rows.append(row)
        print(
            f"[{i}/{len(grid)}] cs={cs} ov={ov} k={k} rr={int(rr)} "
            f"→ precision={score['context_precision']} recall={score['context_recall']}"
        )

    rows.sort(key=lambda r: r["retrieval_avg"], reverse=True)

    # ── 2단계: 상위 설정 전체 평가 (--full) ──
    full_results = []
    if args.full:
        top_rows = rows[: args.full_top_n]
        print(f"\n2단계: 상위 {len(top_rows)}개 설정 × 프롬프트 {prompt_versions} 전체 평가 (LLM 사용)\n")
        for row in top_rows:
            for pv in prompt_versions:
                print(f"전체 평가: cs={row['chunk_size']} ov={row['overlap']} "
                      f"k={row['top_k']} rr={int(row['reranker'])} prompt={pv}")
                report = evaluate(
                    dataset_path=args.dataset,
                    targets_path=args.targets,
                    top_k=row["top_k"],
                    collection_name=collection_for(row["chunk_size"], row["overlap"]),
                    prompt_version=pv,
                    use_reranker=row["reranker"],
                    limit=args.limit,
                    quiet=True,
                )
                agg = report["aggregate"]
                full_results.append(
                    {
                        **{key: row[key] for key in ("chunk_size", "overlap", "top_k", "reranker")},
                        "prompt_version": pv,
                        "overall_avg": agg["overall_avg"],
                        "faithfulness": agg["faithfulness"],
                        "answer_relevancy": agg["answer_relevancy"],
                        "context_precision": agg["context_precision"],
                        "context_recall": agg["context_recall"],
                        "answer_correctness": agg["answer_correctness"],
                        "critical_hallucination_rate": agg["critical_hallucination_rate"],
                        "all_passed": report["all_passed"],
                    }
                )
                print(f"  → overall_avg={agg['overall_avg']} "
                      f"{'✅ 전체 목표 달성' if report['all_passed'] else ''}")
        full_results.sort(key=lambda r: r["overall_avg"], reverse=True)

    # ── 결과 저장 ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "experiments.csv").open("w", newline="", encoding="utf-8-sig") as f:
        source = full_results or rows
        writer = csv.DictWriter(f, fieldnames=list(source[0].keys()))
        writer.writeheader()
        writer.writerows(source)

    best = (full_results or rows)[0]
    recommendation = {
        "recommended_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "full" if full_results else "retrieval_only",
        "best_config": best,
        "env_settings": {
            "RAG_CHUNK_SIZE": str(best["chunk_size"]),
            "RAG_CHUNK_OVERLAP": str(best["overlap"]),
            "RAG_TOP_K": str(best["top_k"]),
            "RAG_USE_RERANKER": "1" if best["reranker"] else "0",
            "RAG_PROMPT_VERSION": best.get("prompt_version", "v2"),
        },
        "apply_guide": [
            "위 env_settings 값을 .env에 넣으세요.",
            "chunk 설정이 바뀌었으면 `python -m scripts.build_rag_index`로 인덱스를 다시 만드세요.",
            "적용 후 `python -m scripts.evaluate_rag`로 재평가해서 목표 달성 여부를 확인하세요.",
        ],
    }
    (RESULTS_DIR / "best_config.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 추천 설정 =====")
    for key, value in recommendation["env_settings"].items():
        print(f"{key}={value}")
    print(f"\n결과 저장: {RESULTS_DIR}/experiments.csv, best_config.json")
    if not args.full:
        print("(검색 지표 기준 추천입니다. --full을 붙이면 답변 품질까지 비교합니다)")


if __name__ == "__main__":
    main()
