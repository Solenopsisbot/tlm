from __future__ import annotations

import argparse

import torch

from .data import TextCodec
from .instruct import END
from .model import TinyLanguageModel, TinyLanguageModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a tiny language model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--no-stop", action="store_true")
    parser.add_argument("--context", type=int, default=0, help="Context cap for non-cached models.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def config_from_checkpoint(data: dict[str, object]) -> TinyLanguageModelConfig:
    config = dict(data)
    if "architecture" not in config:
        config["architecture"] = "rnn"
    return TinyLanguageModelConfig(**config)


def choose_device(name: str, config: TinyLanguageModelConfig) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if config.architecture in {"stream", "swift", "fastconv"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = config_from_checkpoint(checkpoint["config"])
    device = choose_device(args.device, config)
    model = TinyLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    codec = TextCodec.from_dict(checkpoint.get("codec"))

    prompt = codec.encode(args.prompt)
    generated = model.generate(
        prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        stop_sequences=None if args.no_stop else [codec.encode(END)],
        context=args.context or checkpoint.get("train_args", {}).get("seq_len"),
    )
    print(codec.decode(generated))


if __name__ == "__main__":
    main()
