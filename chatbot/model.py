from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
WORD_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
DISALLOWED_AUTOCOMPLETE_WORDS = {
    "어",
    "음",
    "뭐",
    "아",
    "그",
    "저",
    "하하",
    "ㅎㅎ",
    "ㅋㅋ",
    "흠",
    "아니",
}
DISALLOWED_AUTOCOMPLETE_PREFIXES = ("뭐", "뭘")


def is_disallowed_autocomplete_word(word: str) -> bool:
    return word in DISALLOWED_AUTOCOMPLETE_WORDS or word.startswith(
        DISALLOWED_AUTOCOMPLETE_PREFIXES
    )


def next_word_prefix(prompt: str) -> str:
    prompt = prompt.rstrip()
    return f"{prompt} " if prompt else ""


def append_word(prompt: str, word: str) -> str:
    prompt = prompt.rstrip()
    return f"{prompt} {word}" if prompt else word


def word_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def has_final_rieul(character: str) -> bool:
    if len(character) != 1 or not "가" <= character <= "힣":
        return False
    return (ord(character) - ord("가")) % 28 == 8


def context_token_variants(tokens: list[str]) -> list[list[str]]:
    variants = [tokens]
    if (
        tokens
        and len(tokens[-1]) > 1
        and tokens[-1].endswith("수")
        and has_final_rieul(tokens[-1][-2])
    ):
        variants.append([*tokens[:-1], tokens[-1][:-1], "수"])
    return variants


def build_next_word_index(
    texts: list[str],
    max_context_words: int = 4,
) -> dict[str, dict[str, int]]:
    index: dict[str, Counter[str]] = {}
    for text in texts:
        tokens = word_tokens(text)
        for target_index, word in enumerate(tokens):
            if is_disallowed_autocomplete_word(word):
                continue
            max_context = min(max_context_words, target_index)
            for context_size in range(max_context + 1):
                start = target_index - context_size
                key = "\t".join(tokens[start:target_index])
                index.setdefault(key, Counter())[word] += 1
    return {key: dict(counter) for key, counter in index.items()}


def predict_next_words_from_index(
    next_word_index: dict[str, dict[str, int]],
    prompt: str,
    top_k: int = 5,
    max_context_words: int = 4,
    allow_empty_context: bool = True,
) -> list[tuple[str, float]]:
    tokens = word_tokens(prompt)
    combined_scores: Counter[str] = Counter()
    total_weight = 0.0
    matched_keys: set[str] = set()

    for candidate_tokens in context_token_variants(tokens):
        for context_size in range(
            min(max_context_words, len(candidate_tokens)),
            0,
            -1,
        ):
            key = "\t".join(candidate_tokens[-context_size:])
            if key in matched_keys:
                continue
            counts = next_word_index.get(key)
            if counts:
                matched_keys.add(key)
                allowed_counts = {
                    word: count
                    for word, count in counts.items()
                    if not is_disallowed_autocomplete_word(word)
                }
                total = sum(allowed_counts.values())
                if not total:
                    continue
                weight = float(2 ** (context_size - 1))
                for word, count in allowed_counts.items():
                    combined_scores[word] += weight * count / total
                total_weight += weight

    if combined_scores and total_weight:
        ranked = sorted(
            combined_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            (word, score / total_weight)
            for word, score in ranked[:top_k]
        ]

    if allow_empty_context:
        counts = next_word_index.get("")
        if counts:
            ranked = sorted(
                (
                    (word, count)
                    for word, count in counts.items()
                    if not is_disallowed_autocomplete_word(word)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            total = sum(count for _, count in ranked)
            if not total:
                return []
            return [(word, count / total) for word, count in ranked[:top_k]]
    return []


def complete_partial_word(
    next_word_index: dict[str, dict[str, int]],
    prompt: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """입력이 미완성 단어로 끝나면(예: "안녕하세") 그 단어의 완성 후보를 반환한다."""
    # 📝 공백으로 끝나면 단어 입력이 끝난 것이므로 완성할 게 없음
    if not prompt or prompt != prompt.rstrip():
        return []
    tokens = word_tokens(prompt)
    if not tokens:
        return []
    prefix = tokens[-1]
    # 📝 문장이 실제로 이 단어 조각으로 끝나는지 확인 (예: "영화를!" 제외)
    if not prompt.endswith(prefix):
        return []
    # 📝 인덱스의 빈 문맥 키("")에는 전체 단어 빈도가 들어 있음
    unigram = next_word_index.get("", {})
    matches = [
        (word, count)
        for word, count in unigram.items()
        if word.startswith(prefix) and word != prefix
    ]
    if not matches:
        return []
    total = sum(count for _, count in matches)
    ranked = sorted(matches, key=lambda item: (-item[1], item[0]))[:top_k]
    return [(word, count / total) for word, count in ranked]


@dataclass
class ModelConfig:
    block_size: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1


class CharacterTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}

    @classmethod
    def build(cls, texts: list[str], min_frequency: int = 1) -> "CharacterTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for character in text:
                counts[character] = counts.get(character, 0) + 1
        characters = sorted(
            character
            for character, count in counts.items()
            if count >= min_frequency and character not in SPECIAL_TOKENS
        )
        tokens = [*SPECIAL_TOKENS, *characters]
        return cls({token: index for index, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = [self.token_to_id.get(character, self.unk_id) for character in text]
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        ignored = set(SPECIAL_TOKENS)
        return "".join(
            token
            for token_id in token_ids
            if (token := self.id_to_token.get(token_id, "")) not in ignored
        )

    def __len__(self) -> int:
        return len(self.token_to_id)


class CausalTransformerLM(nn.Module):
    def __init__(self, vocabulary_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocabulary_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, vocabulary_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=token_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.layer_norm(hidden)
        logits = self.output(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, -1),
                targets.reshape(batch_size * sequence_length),
                ignore_index=0,
            )
        return logits, loss


def crop_context(token_ids: list[int], block_size: int) -> list[int]:
    return token_ids[-block_size:]


def remove_repeated_leading_words(prompt: str, suffix: str) -> str:
    prompt_words = prompt.split()
    suffix_words = suffix.split()
    if not suffix_words:
        return ""
    if prompt_words and suffix_words[0] == prompt_words[-1]:
        suffix_words = suffix_words[1:]

    compact_words: list[str] = []
    for word in suffix_words:
        if not compact_words or compact_words[-1] != word:
            compact_words.append(word)
    return " ".join(compact_words)


def save_checkpoint(
    path: Path,
    model: CausalTransformerLM,
    tokenizer: CharacterTokenizer,
    config: ModelConfig,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "character_transformer",
            "state_dict": model.state_dict(),
            "tokenizer": tokenizer.token_to_id,
            "config": asdict(config),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[CausalTransformerLM, CharacterTokenizer, ModelConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "character_transformer":
        raise ValueError(
            "지원하지 않는 체크포인트입니다. `python -m chatbot.train`으로 다시 학습하세요."
        )
    tokenizer = CharacterTokenizer(checkpoint["tokenizer"])
    config = ModelConfig(**checkpoint["config"])
    model = CausalTransformerLM(len(tokenizer), config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, config, checkpoint.get("metadata", {})


def filter_logits(
    logits: torch.Tensor,
    tokenizer: CharacterTokenizer,
    top_k: int,
    *,
    allow_eos: bool = True,
) -> torch.Tensor:
    logits = logits.clone()
    for token_id in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id):
        logits[token_id] = -torch.inf
    if not allow_eos:
        logits[tokenizer.eos_id] = -torch.inf
    if top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = -torch.inf
    return logits


@torch.inference_mode()
def next_token_distribution(
    model: CausalTransformerLM,
    tokenizer: CharacterTokenizer,
    config: ModelConfig,
    prompt: str,
    *,
    top_k: int = 10,
    allow_eos: bool = True,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    token_ids = tokenizer.encode(prompt, add_bos=True)
    context = torch.tensor(
        [crop_context(token_ids, config.block_size)],
        dtype=torch.long,
        device=device,
    )
    logits, _ = model(context)
    logits = filter_logits(logits[0, -1], tokenizer, top_k, allow_eos=allow_eos)
    probabilities = torch.softmax(logits, dim=0)
    values, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
    return [
        (tokenizer.id_to_token[token_id.item()], probability.item())
        for token_id, probability in zip(indices, values, strict=True)
        if probability.item() > 0
    ]


@torch.inference_mode()
def complete_word(
    model: CausalTransformerLM,
    tokenizer: CharacterTokenizer,
    config: ModelConfig,
    prefix: str,
    *,
    first_character: str,
    max_characters: int = 12,
    device: torch.device | str = "cpu",
) -> str:
    generated = first_character
    stop_characters = set(" \n\t.!?,")
    for _ in range(max_characters - 1):
        prompt = prefix + generated
        candidates = next_token_distribution(
            model,
            tokenizer,
            config,
            prompt,
            top_k=1,
            allow_eos=False,
            device=device,
        )
        if not candidates:
            break
        next_character = candidates[0][0]
        if next_character in SPECIAL_TOKENS or next_character in stop_characters:
            break
        generated += next_character
    return generated.strip()


@torch.inference_mode()
def predict_next_words(
    model: CausalTransformerLM,
    tokenizer: CharacterTokenizer,
    config: ModelConfig,
    prompt: str,
    top_k: int = 5,
    next_word_index: dict[str, dict[str, int]] | None = None,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    if next_word_index:
        indexed_candidates = predict_next_words_from_index(
            next_word_index,
            prompt,
            top_k=top_k,
            allow_empty_context=not bool(word_tokens(prompt)),
        )
        if indexed_candidates:
            return indexed_candidates
        # 📝 문맥 일치가 없으면 미완성 단어 완성을 먼저 시도 (예: "안녕하세" → "안녕하세요")
        completions = complete_partial_word(next_word_index, prompt, top_k=top_k)
        if completions:
            return completions

    prefix = next_word_prefix(prompt)
    prompt_words = prompt.strip().split()
    last_prompt_word = prompt_words[-1] if prompt_words else ""
    candidates = next_token_distribution(
        model,
        tokenizer,
        config,
        prefix,
        top_k=max(top_k * 4, 8),
        allow_eos=False,
        device=device,
    )
    next_words: list[tuple[str, float]] = []
    seen: set[str] = set()
    for character, probability in candidates:
        if character.isspace() or character in ".!?,":
            continue
        word = complete_word(
            model,
            tokenizer,
            config,
            prefix,
            first_character=character,
            device=device,
        )
        if (
            word
            and not is_disallowed_autocomplete_word(word)
            and word != last_prompt_word
            and word not in seen
        ):
            seen.add(word)
            next_words.append((word, probability))
        if len(next_words) >= top_k:
            break
    return next_words


def generate_text_from_index(
    next_word_index: dict[str, dict[str, int]],
    prompt: str,
    max_new_words: int = 8,
    top_k: int = 5,
) -> str:
    result = prompt.strip()
    for _ in range(max_new_words):
        candidates = predict_next_words_from_index(
            next_word_index,
            result,
            top_k=top_k,
            allow_empty_context=False,
        )
        if not candidates:
            break

        current_words = word_tokens(result)
        last_word = current_words[-1] if current_words else ""
        next_word = next(
            (word for word, _ in candidates if word != last_word),
            "",
        )
        if not next_word:
            break
        result = append_word(result, next_word)
    return result or prompt


def generate_text_hybrid(
    model: CausalTransformerLM,
    tokenizer: CharacterTokenizer,
    config: ModelConfig,
    prompt: str,
    max_new_words: int = 8,
    top_k: int = 5,
    next_word_index: dict[str, dict[str, int]] | None = None,
    device: torch.device | str = "cpu",
) -> str:
    result = prompt.strip()
    if next_word_index:
        prompt_word_count = len(word_tokens(result))
        indexed_max_words = min(max_new_words, 2) if prompt_word_count <= 1 else max_new_words
        indexed_result = generate_text_from_index(
            next_word_index,
            result,
            max_new_words=indexed_max_words,
            top_k=top_k,
        )
        if indexed_result != result:
            new_word_count = len(word_tokens(indexed_result)) - len(word_tokens(result))
            if new_word_count >= min(2, max_new_words) or prompt_word_count <= 1:
                return indexed_result
            result = indexed_result

    generated_words: list[str] = []
    # 📝 미완성 단어로 끝나면(문맥 일치도 없으면) 완성형으로 교체 후 시작 (예: "안녕하세" → "안녕하세요")
    if next_word_index and word_tokens(result):
        has_context_match = bool(
            predict_next_words_from_index(
                next_word_index,
                result,
                top_k=1,
                allow_empty_context=False,
            )
        )
        if not has_context_match:
            completions = complete_partial_word(next_word_index, result, top_k=1)
            if completions:
                last = word_tokens(result)[-1]
                result = result[: len(result) - len(last)] + completions[0][0]
    for _ in range(max_new_words):
        candidates = predict_next_words(
            model,
            tokenizer,
            config,
            result,
            top_k=top_k,
            next_word_index=next_word_index,
            device=device,
        )
        current_words = word_tokens(result)
        last_word = current_words[-1] if current_words else ""
        next_word = next(
            (
                word
                for word, _ in candidates
                if word != last_word and word not in generated_words[-2:]
            ),
            "",
        )
        if not next_word:
            break
        result = append_word(result, next_word)
        generated_words.append(next_word)
    return result or prompt


@torch.inference_mode()
def generate_text(
    model: CausalTransformerLM,
    tokenizer: CharacterTokenizer,
    config: ModelConfig,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 20,
    min_new_tokens: int = 8,
    device: torch.device | str = "cpu",
) -> str:
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature는 0.1 이상 2.0 이하여야 합니다.")
    prompt = prompt.rstrip()
    token_ids = tokenizer.encode(prompt, add_bos=True)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        context = torch.tensor(
            [crop_context([*token_ids, *generated], config.block_size)],
            dtype=torch.long,
            device=device,
        )
        logits, _ = model(context)
        next_logits = filter_logits(
            logits[0, -1] / temperature,
            tokenizer,
            top_k,
            allow_eos=len(generated) >= min_new_tokens,
        )
        probabilities = torch.softmax(next_logits, dim=0)
        next_id = torch.multinomial(probabilities, num_samples=1).item()
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        next_character = tokenizer.id_to_token.get(next_id, "")
        if len(generated) >= min_new_tokens and next_character in ".!?\n":
            break

    suffix = tokenizer.decode(generated).strip()
    suffix = remove_repeated_leading_words(prompt, suffix)
    if not suffix:
        return prompt
    if suffix[0] in ".!?,":
        return f"{prompt}{suffix}"
    return f"{prompt} {suffix}" if prompt else suffix
