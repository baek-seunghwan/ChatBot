from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from chatbot.model import (
    CharacterTokenizer,
    CausalTransformerLM,
    ModelConfig,
    build_next_word_index,
    save_checkpoint,
)


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        if len(token_ids) <= block_size + 1:
            raise ValueError("학습 토큰 수가 block_size보다 커야 합니다.")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.token_ids[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\ufeff", "").strip().split())


def load_corpus_texts(corpus_path: Path, max_lines: int | None) -> list[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"학습 말뭉치 파일이 없습니다: {corpus_path}")

    texts: list[str] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        text = normalize_text(line)
        if not text or text.startswith("#"):
            continue
        texts.append(text)
        if max_lines is not None and len(texts) >= max_lines:
            break
    if not texts:
        raise ValueError(f"학습할 문장이 없습니다: {corpus_path}")
    return texts


def build_training_tokens(texts: list[str], tokenizer: CharacterTokenizer) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        token_ids.append(tokenizer.token_to_id.get("\n", tokenizer.eos_id))
    return token_ids


def split_tokens(token_ids: list[int], validation_ratio: float) -> tuple[list[int], list[int]]:
    split = int(len(token_ids) * (1 - validation_ratio))
    split = max(split, 1)
    return token_ids[:split], token_ids[split:]


def evaluate(
    model: CausalTransformerLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            if loss is not None:
                losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def learning_rate_at_step(
    step: int,
    total_steps: int,
    base_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_learning_rate
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (base_learning_rate - min_learning_rate)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    parser = argparse.ArgumentParser(
        description="일반 한국어 문장 말뭉치로 다음 단어 예측용 문자 단위 Transformer를 학습합니다."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=base_dir / "autocomplete_corpus.txt",
        help="한 줄에 한 문장씩 저장된 자동완성 전용 말뭉치 파일",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "artifacts" / "chatbot.pt",
    )
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="최대 학습 step 수. 0이면 지정한 epochs 전체를 학습합니다.",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=20, help="각 몇 step마다 학습 진행 로그를 출력할지")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    texts = load_corpus_texts(args.corpus, args.max_lines)
    next_word_index = build_next_word_index(texts)
    tokenizer = CharacterTokenizer.build(texts, min_frequency=1)
    token_ids = build_training_tokens(texts, tokenizer)
    train_tokens, validation_tokens = split_tokens(token_ids, validation_ratio=0.08)
    config = ModelConfig(
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_dataset = LanguageModelDataset(train_tokens, config.block_size)
    validation_dataset = LanguageModelDataset(validation_tokens, config.block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = CausalTransformerLM(len(tokenizer), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = min(max_steps, len(train_loader) * args.epochs) if max_steps else len(train_loader) * args.epochs

    print(
        f"device={device}, texts={len(texts):,}, tokens={len(token_ids):,}, "
        f"vocab={len(tokenizer):,}, parameters={sum(p.numel() for p in model.parameters()):,}, "
        f"steps_per_epoch={len(train_loader):,}, total_steps={total_steps:,}, "
        f"max_steps={max_steps or 'full'}"
    )
    global_step = 0
    best_validation_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("loss 계산에 실패했습니다.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            step_number = global_step + 1
            learning_rate = learning_rate_at_step(
                step_number,
                total_steps,
                args.learning_rate,
                args.min_learning_rate,
                args.warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            global_step = step_number

            should_log = global_step == 1 or global_step % args.log_every == 0 or global_step >= (total_steps or 0)
            should_eval = global_step == 1 or global_step % args.eval_every == 0
            if should_eval:
                validation_loss = evaluate(
                    model,
                    validation_loader,
                    device,
                    max_batches=args.eval_batches,
                )
            else:
                validation_loss = None

            if should_log:
                elapsed_minutes = (time.time() - start_time) / 60.0
                progress_percent = (global_step / total_steps * 100.0) if total_steps else 0.0
                log_message = (
                    f"[학습중] epoch={epoch:02d}/{args.epochs} "
                    f"step={global_step:04d}/{total_steps} "
                    f"progress={progress_percent:.1f}% "
                    f"loss={loss.item():.4f} "
                    f"lr={learning_rate:.2e} "
                    f"elapsed={elapsed_minutes:.1f}분"
                )
                if validation_loss is not None:
                    log_message += f" val_loss={validation_loss:.4f}"
                print(log_message)

            if should_eval and validation_loss is not None and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    args.output,
                    model,
                    tokenizer,
                    config,
                    {
                        "architecture": "character_transformer",
                        "data_type": "plain_autocomplete_corpus",
                        "corpus": str(args.corpus),
                        "texts": len(texts),
                        "tokens": len(token_ids),
                        "vocabulary_size": len(tokenizer),
                        "next_word_index": next_word_index,
                        "epochs": args.epochs,
                        "steps": global_step,
                        "best_validation_loss": best_validation_loss,
                    },
                )

            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    if best_validation_loss == float("inf"):
        save_checkpoint(
            args.output,
            model,
            tokenizer,
            config,
            {
                "architecture": "character_transformer",
                "data_type": "plain_autocomplete_corpus",
                "corpus": str(args.corpus),
                "texts": len(texts),
                "tokens": len(token_ids),
                "vocabulary_size": len(tokenizer),
                "next_word_index": next_word_index,
                "epochs": epoch,
                "steps": global_step,
                "best_validation_loss": best_validation_loss,
            },
        )
    print(f"모델 저장: {args.output.resolve()} (validation loss가 가장 좋았던 checkpoint)")


if __name__ == "__main__":
    main()
