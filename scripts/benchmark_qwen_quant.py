"""Benchmark Qwen before/after quantized loading.

This script is for week 9 verification:

    uv run python -m scripts.benchmark_qwen_quant --prompt "안녕"

On CUDA machines it can compare fp/base, 8bit, and 4bit bitsandbytes loading:

    python -m scripts.benchmark_qwen_quant --modes base 8bit 4bit

On Mac/CPU, bitsandbytes quantized modes are reported as skipped because
bitsandbytes 4bit/8bit loading requires CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "안녕, 자기소개 해줘"


def _rss_mb() -> float:
    """Return max RSS in MB.

    macOS reports bytes, Linux reports KB. The threshold keeps both readable.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _worker(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mode = args.mode
    if mode in {"4bit", "8bit"} and not torch.cuda.is_available():
        return {
            "mode": mode,
            "status": "skipped",
            "reason": "bitsandbytes 4bit/8bit loading requires CUDA.",
        }

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if mode == "4bit":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_config, device_map="auto"
        )
    elif mode == "8bit":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_config, device_map="auto"
        )
    else:
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    if args.adapter and Path(args.adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    load_sec = time.perf_counter() - started

    messages = [
        {"role": "system", "content": "한국어로 간결하게 답하세요."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gen_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generate_sec = time.perf_counter() - gen_started
    generated = output[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    cuda_peak_mb = None
    if torch.cuda.is_available():
        cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "mode": mode,
        "status": "ok",
        "device": str(model.device),
        "load_sec": round(load_sec, 2),
        "generate_sec": round(generate_sec, 2),
        "max_rss_mb": round(_rss_mb(), 1),
        "cuda_peak_mb": round(cuda_peak_mb, 1) if cuda_peak_mb is not None else None,
        "answer_preview": answer[:200],
    }


def _run_worker(args: argparse.Namespace, mode: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_qwen_quant",
        "--worker",
        "--mode",
        mode,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.adapter:
        command.extend(["--adapter", args.adapter])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "mode": mode,
            "status": "failed",
            "stderr": completed.stderr[-1000:],
        }
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen quantization benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(REPO_ROOT / "artifacts" / "lora_adapter"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--modes", nargs="+", default=["base", "8bit", "4bit"])
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "week9_quant_benchmark.json"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=["base", "8bit", "4bit"], default="base")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_worker(args), ensure_ascii=False))
        return

    results = [_run_worker(args, mode) for mode in args.modes]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
