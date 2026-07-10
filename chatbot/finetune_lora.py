# 📝 finetune_lora: Qwen 모델 LoRA / QLoRA 파인튜닝
#
# LoRA  = 원본 가중치는 얼리고(frozen) 작은 어댑터 행렬만 학습 → 적은 메모리로 파인튜닝
# QLoRA = 원본 가중치를 4bit로 양자화해서 올린 뒤 LoRA 학습 → 더 적은 메모리 (CUDA 전용)
#
# 학습 데이터: chatbot/qa_corpus.txt ("질문: ... 답변: ..." 한 줄 형식)
#
# 실행 (Mac/CPU: LoRA):   uv run python -m chatbot.finetune_lora
# 실행 (Colab/CUDA: QLoRA): python -m chatbot.finetune_lora --qlora
# 어댑터 병합 저장:        uv run python -m chatbot.finetune_lora --merge
#
# 결과물:
#   artifacts/lora_adapter/  ← 어댑터 (수 MB). qwen_local.py가 자동으로 로드
#   artifacts/qwen_merged/   ← --merge 시 병합된 전체 모델 (GGUF 변환용)
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from .config import REPO_ROOT
except ImportError:
    # Support direct execution: python chatbot/finetune_lora.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from chatbot.config import REPO_ROOT

QA_CORPUS = REPO_ROOT / "chatbot" / "qa_corpus.txt"
ADAPTER_DIR = REPO_ROOT / "artifacts" / "lora_adapter"
MERGED_DIR = REPO_ROOT / "artifacts" / "qwen_merged"

SYSTEM_PROMPT = (
    "너는 Leon이 만든 친절한 한국어 챗봇이다. "
    "질문에 한국어로 간결하고 정확하게 답한다."
)


def load_qa_pairs(path: Path = QA_CORPUS) -> list[dict]:
    """"질문: X 답변: Y" 형식의 코퍼스를 (질문, 답변) 목록으로 파싱한다."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"질문:\s*(.+?)\s*답변:\s*(.+)", line.strip())
        if match:
            pairs.append({"question": match.group(1), "answer": match.group(2)})
    if not pairs:
        raise ValueError(f"{path}에서 QA 쌍을 찾지 못했습니다.")
    return pairs


def build_dataset(tokenizer, pairs: list[dict], max_length: int = 512):
    """QA 쌍을 Qwen chat 템플릿 텍스트로 만들고 토크나이즈한다.

    labels에서 프롬프트 부분은 -100으로 마스킹해서
    '답변 부분만' 손실(loss)에 반영되도록 한다.
    """
    from datasets import Dataset

    def to_features(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + example["answer"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        labels = list(full["input_ids"])
        # 📝 프롬프트 토큰은 학습 대상에서 제외 (-100 = loss 무시)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        full["labels"] = labels
        return full

    dataset = Dataset.from_list(pairs)
    return dataset.map(to_features, remove_columns=["question", "answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen LoRA/QLoRA 파인튜닝")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--qlora", action="store_true", help="4bit 양자화 + LoRA (CUDA 필요)")
    parser.add_argument("--merge", action="store_true", help="학습된 어댑터를 원본에 병합해서 저장")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank(r): 어댑터 행렬 크기")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    use_cuda = torch.cuda.is_available()

    # ── --merge: 학습 없이 어댑터 병합만 수행 ──────────────────
    if args.merge:
        from peft import PeftModel

        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(f"먼저 학습을 실행하세요: {ADAPTER_DIR} 없음")
        print(f"[merge] {args.model} + {ADAPTER_DIR} → {MERGED_DIR}")
        try:
            base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
        merged = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).merge_and_unload()
        merged.save_pretrained(str(MERGED_DIR))
        AutoTokenizer.from_pretrained(args.model).save_pretrained(str(MERGED_DIR))
        print("완료. GGUF 변환: bash scripts/convert_gguf.sh")
        return

    # ── 1. 모델 로드 (QLoRA면 4bit 양자화) ─────────────────────
    if args.qlora:
        if not use_cuda:
            raise RuntimeError(
                "QLoRA(4bit)는 bitsandbytes가 CUDA GPU를 요구합니다. "
                "Mac에서는 --qlora 없이 LoRA로 학습하거나, Colab에서 실행하세요."
            )
        from transformers import BitsAndBytesConfig

        print(f"[1/4] 모델 로드 (QLoRA 4bit): {args.model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4: QLoRA 논문의 양자화 방식
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,     # 양자화 상수도 한 번 더 양자화 → 메모리 추가 절약
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto"
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    else:
        print(f"[1/4] 모델 로드 (LoRA): {args.model}")
        device = "cuda" if use_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float16 if device != "cpu" else torch.float32,
            ).to(device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── 2. LoRA 어댑터 부착 ────────────────────────────────────
    print(f"[2/4] LoRA 어댑터 부착 (r={args.rank})")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # 📝 attention의 Q/K/V/O 프로젝션에만 어댑터를 붙인다 (일반적인 선택)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 전체 대비 학습 파라미터 비율 출력

    # ── 3. 데이터셋 준비 ──────────────────────────────────────
    pairs = load_qa_pairs()
    print(f"[3/4] 학습 데이터: {len(pairs)}쌍 ({QA_CORPUS.name})")
    dataset = build_dataset(tokenizer, pairs)

    # ── 4. 학습 ───────────────────────────────────────────────
    print(f"[4/4] 학습 시작 (epochs={args.epochs}, lr={args.lr})")
    training_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "artifacts" / "lora_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=use_cuda and not args.qlora,
        bf16=args.qlora,
        use_cpu=not use_cuda and not torch.backends.mps.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # 📝 배치 안에서 길이가 다른 문장을 패딩으로 맞춰준다 (labels는 -100으로 패딩)
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"완료: 어댑터 저장 → {ADAPTER_DIR}")
    print("적용 확인: QWEN_LORA_ADAPTER=artifacts/lora_adapter uv run python -m chatbot.qwen_local '이름 알려줘'")


if __name__ == "__main__":
    main()
