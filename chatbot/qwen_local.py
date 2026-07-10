# 📝 Qwen 로컬 모드: 진짜 대화가 되는 사전학습 소형 LLM을 내 컴퓨터에서 실행
#
# 직접 학습한 문자 단위 모델(chatbot.pt)은 저장된 QA 범위 안에서만 답할 수 있다.
# 자유로운 대화가 필요하면 이미 수조 토큰으로 사전학습된 Qwen을 쓰는 것이 맞다.
# (모델을 처음 쓸 때 Hugging Face에서 자동 다운로드: 0.5B 기준 약 1GB)
#
# 모델 변경: .env에 QWEN_LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct (더 똑똑, 더 느림)
#
# 양자화 로딩 (CUDA GPU에서만 동작, bitsandbytes 필요):
#   QWEN_QUANT=4bit  → 4비트 양자화 로딩 (메모리 약 1/4)
#   QWEN_QUANT=8bit  → 8비트 양자화 로딩 (메모리 약 1/2)
# LoRA 어댑터 적용 (finetune_lora.py 학습 결과):
#   QWEN_LORA_ADAPTER=artifacts/lora_adapter
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_QWEN_MODEL = os.getenv("QWEN_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN_QUANT = os.getenv("QWEN_QUANT", "").lower()          # "", "4bit", "8bit"
QWEN_LORA_ADAPTER = os.getenv("QWEN_LORA_ADAPTER", "")     # 어댑터 폴더 경로

SYSTEM_PROMPT = (
    "너는 Leon이 만든 친절한 한국어 챗봇이다. "
    "질문에 한국어로 간결하고 정확하게 답한다. "
    "모르는 것은 모른다고 솔직하게 답한다."
)


@lru_cache(maxsize=1)
def _load_model():
    """Qwen 모델과 토크나이저를 최초 1회만 로드한다. (Mac은 MPS 가속 사용)

    QWEN_QUANT가 설정되어 있고 CUDA가 있으면 bitsandbytes 양자화로 로드하고,
    QWEN_LORA_ADAPTER가 설정되어 있으면 학습된 LoRA 어댑터를 얹는다.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_QWEN_MODEL)

    if QWEN_QUANT in ("4bit", "8bit"):
        if device != "cuda":
            print(f"[경고] QWEN_QUANT={QWEN_QUANT}는 CUDA 전용입니다. 일반 로딩으로 대체합니다.")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    DEFAULT_QWEN_MODEL, dtype="auto"
                ).to(device)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    DEFAULT_QWEN_MODEL, torch_dtype="auto"
                ).to(device)
        else:
            from transformers import BitsAndBytesConfig

            if QWEN_QUANT == "4bit":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_QWEN_MODEL, quantization_config=bnb, device_map="auto"
            )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_QWEN_MODEL, dtype="auto"
            ).to(device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_QWEN_MODEL, torch_dtype="auto"
            ).to(device)

    # 📝 파인튜닝한 LoRA 어댑터가 있으면 원본 모델 위에 얹는다.
    if QWEN_LORA_ADAPTER and Path(QWEN_LORA_ADAPTER).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, QWEN_LORA_ADAPTER)
        print(f"[LoRA] 어댑터 적용됨: {QWEN_LORA_ADAPTER}")

    model.eval()
    return tokenizer, model, device


def generate_answer(question: str, max_new_tokens: int = 300) -> str:
    """질문 하나를 받아 Qwen이 생성한 답변을 돌려준다."""
    import torch

    tokenizer, model, device = _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "안녕, 자기소개 해줘"
    print(generate_answer(question))
