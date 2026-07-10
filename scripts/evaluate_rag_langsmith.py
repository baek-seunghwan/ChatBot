"""현재 RAG 평가셋을 LangSmith dataset/experiment로 실행한다.

실행:
    uv run python -m scripts.evaluate_rag_langsmith

LLM 심사자의 변동과 오채점을 피하기 위해 이 스크립트의 기본 evaluator는
기준 출처/기준 답변을 이용한 결정적(code-based) 지표만 사용한다.
"""
from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path

from langsmith import Client

from chatbot.config import REPO_ROOT
from chatbot.rag_chain import RagChain
from scripts.rag_eval_core import (
    context_precision,
    context_recall,
    coverage,
    is_refusal,
)

DEFAULT_DATASET = REPO_ROOT / "eval" / "dataset.jsonl"
DEFAULT_LANGSMITH_DATASET = "chatbot-rag-eval-v2-20"


def load_items(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def sync_dataset(client: Client, dataset_name: str, items: list[dict]):
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "현재 ChatBot 저장소 eval/dataset.jsonl의 RAG 질문, 기준 답변, "
                "기대 출처 20문항"
            ),
            metadata={"source": "eval/dataset.jsonl", "version": "v2"},
        )

    existing = {
        example.metadata.get("eval_id"): example
        for example in client.list_examples(dataset_id=dataset.id)
        if example.metadata and example.metadata.get("eval_id")
    }
    new_examples = []
    for item in items:
        inputs = {"question": item["question"]}
        outputs = {
            "answer": item["ground_truth"],
            "expected_sources": item["expected_sources"],
        }
        metadata = {
            "eval_id": item["id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "dataset_split": ["test"],
        }
        old = existing.get(item["id"])
        if old is not None:
            client.update_example(
                old.id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                split="test",
                dataset_id=dataset.id,
            )
        else:
            new_examples.append(
                {
                    "id": uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_name}:{item['id']}"),
                    "inputs": inputs,
                    "outputs": outputs,
                    "metadata": metadata,
                    "split": "test",
                }
            )
    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)
    return dataset


def context_precision_eval(outputs: dict, reference_outputs: dict) -> float:
    return context_precision(outputs["sources"], reference_outputs["expected_sources"])


def context_recall_eval(outputs: dict, reference_outputs: dict) -> float:
    return context_recall(outputs["sources"], reference_outputs["expected_sources"])


def abstention_accuracy(outputs: dict, reference_outputs: dict) -> float:
    """답이 있으면 답하고, 없으면 거절했는지 평가한다."""
    should_refuse = not reference_outputs["expected_sources"]
    refused = is_refusal(outputs["answer"])
    return float(should_refuse == refused)


def answer_correctness_rules(outputs: dict, reference_outputs: dict) -> float:
    """거절 오류를 먼저 확정하고, 나머지는 기준 정답 단어 포함률로 계산한다."""
    expected_sources = reference_outputs["expected_sources"]
    refused = is_refusal(outputs["answer"])
    if expected_sources and refused:
        return 0.0
    if not expected_sources:
        return float(refused)
    return round(coverage(reference_outputs["answer"], outputs["answer"]), 3)


def faithfulness_rules(outputs: dict, reference_outputs: dict) -> float:
    """답변 단어가 검색 문맥에 포함되는 정도. 거절은 환각하지 않았으므로 1점."""
    if is_refusal(outputs["answer"]):
        return 1.0
    return round(coverage(outputs["answer"], "\n\n".join(outputs["contexts"])), 3)


def aggregate_rows(rows: list[dict]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        evaluation_results = row["evaluation_results"]
        results = (
            evaluation_results["results"]
            if isinstance(evaluation_results, dict)
            else evaluation_results.results
        )
        for result in results:
            key = result["key"] if isinstance(result, dict) else result.key
            score = result["score"] if isinstance(result, dict) else result.score
            if score is not None:
                scores[key].append(float(score))
    return {
        key: round(sum(values) / len(values), 3)
        for key, values in sorted(scores.items())
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LangSmith RAG 전체 평가")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--langsmith-dataset", default=DEFAULT_LANGSMITH_DATASET)
    parser.add_argument("--experiment-prefix", default="rag-v2-baseline")
    args = parser.parse_args()

    items = load_items(args.dataset)
    client = Client()
    dataset = sync_dataset(client, args.langsmith_dataset, items)
    chain = RagChain()

    def target(inputs: dict) -> dict:
        result = chain.ask(inputs["question"])
        return {
            "answer": result.answer,
            "sources": result.sources,
            "contexts": [chunk.text for chunk in result.chunks],
            "confidence": result.confidence,
            "retrieved_chunks": result.retrieved_chunks,
            "provider": result.provider,
            "model": result.model,
        }

    experiment = client.evaluate(
        target,
        data=dataset.id,
        evaluators=[
            context_precision_eval,
            context_recall_eval,
            abstention_accuracy,
            answer_correctness_rules,
            faithfulness_rules,
        ],
        experiment_prefix=args.experiment_prefix,
        description="현재 RAG 전체 20문항 LangSmith code-based baseline 평가",
        metadata={
            "top_k": chain.top_k,
            "min_score": chain.min_score,
            "prompt_version": chain.prompt_version,
            "reranker": chain.use_reranker,
        },
        max_concurrency=1,
        num_repetitions=1,
    )
    rows = list(experiment)
    print("\nLANGSMITH_EVALUATION_RESULT")
    print(json.dumps({
        "dataset": dataset.name,
        "examples": len(rows),
        "experiment": experiment.experiment_name,
        "experiment_id": str(experiment.experiment_id),
        "url": experiment.url,
        "aggregate": aggregate_rows(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
