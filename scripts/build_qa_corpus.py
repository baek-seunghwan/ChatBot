# 📝 QA 학습 말뭉치 생성: qa_pairs.jsonl → qa_corpus.txt
#
# 로컬 모델이 "문장 이어쓰기"가 아니라 "질문에 답하기"를 배우도록
# 한 줄에 "질문: ... 답변: ..." 형식으로 학습 데이터를 만든다.
# 학습 시 모델은 "답변:" 뒤에 무엇이 와야 하는지를 배우게 된다.
#
# 실행:
#   uv run python -m scripts.build_qa_corpus
#   uv run python -m scripts.build_qa_corpus --repeat 4   # 데이터를 4배로 복제(작은 데이터 보강)
#
# 그다음 학습:
#   uv run python -m chatbot.train --corpus chatbot/qa_corpus.txt \
#       --epochs 60 --max-steps 0 --block-size 256 --embedding-dim 256 --num-heads 8 --num-layers 6
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chatbot.config import REPO_ROOT

QA_PAIRS_PATH = REPO_ROOT / "chatbot" / "qa_pairs.jsonl"
CORPUS_PATH = REPO_ROOT / "chatbot" / "qa_corpus.txt"

# 📝 추론(app.py)과 반드시 같은 형식이어야 한다!
QA_FORMAT = "질문: {q} 답변: {a}"


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_corpus(pairs: list[dict], repeat: int) -> list[str]:
    """질문 본문 + 변형 질문들을 전부 학습 문장으로 만든다."""
    lines = []
    for pair in pairs:
        questions = [pair["question"], *pair.get("variants", [])]
        for q in questions:
            lines.append(QA_FORMAT.format(q=q.strip(), a=pair["answer"].strip()))
    # 📝 작은 데이터셋 보강: 순서를 섞어 여러 번 반복 (모델이 형식을 확실히 외우게)
    corpus = []
    for _ in range(repeat):
        shuffled = lines[:]
        random.shuffle(shuffled)
        corpus.extend(shuffled)
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="QA 말뭉치 생성")
    parser.add_argument("--pairs", type=Path, default=QA_PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    pairs = load_pairs(args.pairs)
    corpus = build_corpus(pairs, args.repeat)
    args.output.write_text("\n".join(corpus) + "\n", encoding="utf-8")

    total_questions = sum(1 + len(p.get("variants", [])) for p in pairs)
    print(f"QA 쌍 {len(pairs)}개 (변형 포함 질문 {total_questions}개)")
    print(f"학습 문장 {len(corpus)}줄 저장: {args.output}")
    print("\n다음 명령으로 학습하세요:")
    print("uv run python -m chatbot.train --corpus chatbot/qa_corpus.txt \\")
    print("    --epochs 60 --max-steps 0 --block-size 256 --embedding-dim 256 --num-heads 8 --num-layers 6")


if __name__ == "__main__":
    main()
