#!/usr/bin/env bash
# 📝 GGUF 변환 + 양자화 스크립트
#
# GGUF = llama.cpp가 쓰는 모델 파일 형식. GPU 없이 CPU에서도 빠르게 실행된다.
# 흐름: HF 모델(또는 LoRA 병합 모델) → GGUF(f16) → 양자화(Q4_K_M)
#
# 사전 준비 (LoRA 학습 결과를 변환하려면 먼저 병합):
#   uv run python -m chatbot.finetune_lora --merge
#
# 실행:
#   bash scripts/convert_gguf.sh                     # artifacts/qwen_merged 변환
#   bash scripts/convert_gguf.sh Qwen/Qwen2.5-0.5B-Instruct  # 원본 모델 변환
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${1:-$REPO_ROOT/artifacts/qwen_merged}"
OUT_DIR="$REPO_ROOT/artifacts/gguf"
LLAMA_CPP_DIR="$REPO_ROOT/artifacts/llama.cpp"

mkdir -p "$OUT_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "오류: 변환할 HF 모델 폴더가 없습니다: $MODEL_DIR" >&2
    echo "LoRA 어댑터를 변환하려면 먼저 실행하세요:" >&2
    echo "  uv run python -m chatbot.finetune_lora --merge" >&2
    echo "또는 원본 모델을 바로 변환하려면 예:" >&2
    echo "  bash scripts/convert_gguf.sh Qwen/Qwen2.5-0.5B-Instruct" >&2
    exit 1
fi

# ── 1. llama.cpp 준비 (변환 스크립트 + quantize 바이너리) ─────────
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "[1/4] llama.cpp 클론"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
else
    echo "[1/4] llama.cpp 이미 있음"
fi

echo "[2/4] 변환 의존성 설치"
pip install -q --break-system-packages -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt" 2>/dev/null \
    || pip install -q gguf sentencepiece protobuf

# ── 2. HF → GGUF (f16) ───────────────────────────────────────────
echo "[3/4] GGUF 변환: $MODEL_DIR"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
    --outfile "$OUT_DIR/qwen-f16.gguf" --outtype f16

# ── 3. 양자화 (Q4_K_M: 크기 대비 품질이 좋은 표준 선택) ──────────
echo "[4/4] 양자화 (Q4_K_M)"
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ] || [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-cli" ]; then
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_METAL=ON >/dev/null
    cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize llama-cli -j >/dev/null
fi
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$OUT_DIR/qwen-f16.gguf" "$OUT_DIR/qwen-q4_k_m.gguf" Q4_K_M

echo ""
echo "완료:"
ls -lh "$OUT_DIR"
echo ""
echo "실행 예시 (llama.cpp):"
echo "  $LLAMA_CPP_DIR/build/bin/llama-cli -m $OUT_DIR/qwen-q4_k_m.gguf -p '안녕'"
