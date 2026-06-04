from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .instruct import (
    BEGIN_ANSWER,
    BEGIN_ASSISTANT,
    BEGIN_CONTENT,
    BEGIN_REFLECT,
    BEGIN_ROLE,
    BEGIN_SYSTEM,
    BEGIN_THINK,
    BEGIN_USER,
    END,
    END_ANSWER,
    END_REFLECT,
    END_THINK,
)

TokenMode = Literal["byte", "char", "bpe"]


@dataclass(frozen=True)
class TextCodec:
    mode: TokenMode
    vocab: tuple[str, ...] | None = None
    tokenizer_json: str | None = None

    @property
    def vocab_size(self) -> int:
        if self.mode == "bpe":
            return self.tokenizer.get_vocab_size()
        return 256 if self.mode == "byte" else len(self.vocab or ())

    @property
    def tokenizer(self) -> Tokenizer:
        if self.mode != "bpe" or self.tokenizer_json is None:
            raise ValueError("codec is not backed by a tokenizer")
        return Tokenizer.from_str(self.tokenizer_json)

    def encode(self, text: str) -> torch.Tensor:
        if self.mode == "byte":
            return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if self.mode == "bpe":
            return torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)
        assert self.vocab is not None
        stoi = {char: index for index, char in enumerate(self.vocab)}
        missing = sorted(set(text) - set(stoi))
        if missing:
            raise ValueError(f"text contains characters outside checkpoint vocab: {missing!r}")
        return torch.tensor([stoi[char] for char in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        if self.mode == "byte":
            return bytes(max(0, min(255, int(t))) for t in tokens).decode(
                "utf-8",
                errors="replace",
            )
        if self.mode == "bpe":
            return self.tokenizer.decode([int(token) for token in tokens])
        assert self.vocab is not None
        return "".join(self.vocab[int(token)] for token in tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "vocab": list(self.vocab) if self.vocab else None,
            "tokenizer_json": self.tokenizer_json,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "TextCodec":
        if not data:
            return cls(mode="byte")
        mode = data.get("mode", "byte")
        vocab = data.get("vocab")
        tokenizer_json = data.get("tokenizer_json")
        if mode == "bpe":
            if not isinstance(tokenizer_json, str):
                raise ValueError("bpe codec checkpoint is missing tokenizer_json")
            return cls(mode="bpe", tokenizer_json=tokenizer_json)
        if mode == "char":
            if not isinstance(vocab, list):
                raise ValueError("char codec checkpoint is missing vocab")
            return cls(mode="char", vocab=tuple(str(char) for char in vocab))
        return cls(mode="byte")


def build_codec(text: str, mode: TokenMode, vocab_size: int) -> TextCodec:
    if mode == "byte":
        return TextCodec(mode="byte")
    if mode == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=[
                "<unk>",
                BEGIN_SYSTEM,
                BEGIN_USER,
                BEGIN_ASSISTANT,
                BEGIN_ROLE,
                BEGIN_CONTENT,
                END,
                BEGIN_THINK,
                END_THINK,
                BEGIN_REFLECT,
                END_REFLECT,
                BEGIN_ANSWER,
                END_ANSWER,
            ],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator([text], trainer=trainer)
        return TextCodec(mode="bpe", tokenizer_json=tokenizer.to_str())
    vocab = tuple(sorted(set(text)))
    if len(vocab) < 2:
        raise ValueError("character-level training needs at least two unique characters")
    return TextCodec(mode="char", vocab=vocab)


def load_text_dataset(
    path: str | Path,
    mode: TokenMode,
    vocab_size: int = 4096,
) -> tuple[torch.Tensor, TextCodec]:
    text = Path(path).read_text(encoding="utf-8")
    return encode_dataset_text(text, mode, vocab_size)


def encode_dataset_text(
    text: str,
    mode: TokenMode,
    vocab_size: int = 4096,
) -> tuple[torch.Tensor, TextCodec]:
    if len(text) < 2:
        raise ValueError("training text must contain at least two characters")
    codec = build_codec(text, mode, vocab_size)
    return codec.encode(text), codec


def encode_text_with_codec(text: str, codec: TextCodec) -> torch.Tensor:
    if len(text) < 2:
        raise ValueError("training text must contain at least two characters")
    return codec.encode(text)


def split_tokens(tokens: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be greater than 0 and less than 0.5")
    split = max(1, int(len(tokens) * (1.0 - val_fraction)))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    if len(train_tokens) < 2 or len(val_tokens) < 2:
        raise ValueError("dataset is too small for the requested validation split")
    return train_tokens.contiguous(), val_tokens.contiguous()


def sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a flat tensor")
    if len(tokens) <= seq_len + 1:
        raise ValueError("text split is too short for the requested sequence length")

    starts = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    xs = torch.stack([tokens[i : i + seq_len] for i in starts]).to(device)
    ys = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in starts]).to(device)
    return xs, ys
