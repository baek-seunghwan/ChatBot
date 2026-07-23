from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobility_service.knowledge import default_knowledge_base


DEFAULT_DATASET = PROJECT_ROOT / "eval" / "mobility_knowledge_eval.json"


def evaluate(dataset_path: Path, *, top_k: int = 3) -> dict[str, object]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    knowledge = default_knowledge_base()
    details: list[dict[str, object]] = []
    hits = 0

    for case in cases:
        results = knowledge.search(case["question"], limit=top_k)
        returned = [result.chunk_id for result in results]
        hit = any(
            chunk_id.startswith(case["expectedSource"])
            for chunk_id in returned
        )
        hits += int(hit)
        details.append(
            {
                "question": case["question"],
                "expected": case["expectedSource"],
                "returned": returned,
                "hit": hit,
            }
        )

    total = len(cases)
    return {
        "metric": f"source_hit_at_{top_k}",
        "hits": hits,
        "total": total,
        "score": hits / total if total else 0.0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MOVB 지식 검색 평가")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    result = evaluate(args.dataset, top_k=args.top_k)
    for detail in result["details"]:
        mark = "PASS" if detail["hit"] else "FAIL"
        print(
            f"{mark:4} | {detail['question']} | "
            f"expected={detail['expected']} | returned={detail['returned'][:1]}"
        )
    print(
        f"\n{result['metric']}: {result['hits']}/{result['total']} "
        f"({result['score']:.1%})"
    )
    return 0 if result["score"] >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
