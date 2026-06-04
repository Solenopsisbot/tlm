from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import (
    TextCodec,
    encode_dataset_text,
    encode_text_with_codec,
    load_text_dataset,
    sample_batch,
    split_tokens,
)
from .instruct import END, load_instruction_jsonl
from .model import TinyLanguageModel, TinyLanguageModelConfig, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny non-transformer LM.")
    parser.add_argument("--text", help="Path to a UTF-8/plain text file.")
    parser.add_argument("--instruct-jsonl", help="Path to instruction JSONL data.")
    parser.add_argument("--out", default="checkpoints/tlm.pt", help="Checkpoint path.")
    parser.add_argument("--resume", action="store_true", help="Resume from --out if it exists.")
    parser.add_argument("--init-checkpoint", help="Initialize model and tokenizer from a checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--mode", choices=["char", "byte", "bpe"], default="bpe")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--conv-kernel", type=int, default=5)
    parser.add_argument(
        "--architecture",
        choices=["stream", "swift", "fastconv", "conv", "rnn"],
        default="stream",
    )
    parser.add_argument("--recurrent", choices=["gated", "gru"], default="gru")
    parser.add_argument("--state-kernel", type=int, default=3)
    parser.add_argument("--max-dilation", type=int, default=8192)
    parser.add_argument("--mlp-expansion", type=int, default=2)
    parser.add_argument("--state-heads", type=int, default=4)
    parser.add_argument("--state-rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--sample-prompt", default="ROMEO:\n")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def config_from_checkpoint(data: dict[str, object]) -> TinyLanguageModelConfig:
    config = dict(data)
    if "architecture" not in config:
        config["architecture"] = "rnn"
    return TinyLanguageModelConfig(**config)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    decay_steps = max(1, args.steps - args.warmup_steps)
    progress = min(1.0, max(0.0, (step - args.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + cosine * (args.lr - args.min_lr)


def parameter_groups(
    model: TinyLanguageModel,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("embed.weight"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def estimate_loss(
    model: TinyLanguageModel,
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(tokens, batch_size, seq_len, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def sample_text(
    model: TinyLanguageModel,
    codec: TextCodec,
    prompt: str,
    tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    seq_len: int,
) -> str:
    prompt_tokens = codec.encode(prompt)
    generated = model.generate(
        prompt_tokens,
        max_new_tokens=tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        stop_sequences=[codec.encode(END)],
        context=seq_len,
    )
    return codec.decode(generated)


def save_checkpoint(
    path: str | Path,
    model: TinyLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
    val_loss: float | None,
    codec: TextCodec,
    args: argparse.Namespace,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.__dict__,
            "codec": codec.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: TinyLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, float | None]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        int(checkpoint.get("step", 0)),
        float(checkpoint.get("train_loss", checkpoint.get("loss", float("nan")))),
        checkpoint.get("val_loss"),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    init_checkpoint = None
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        config = config_from_checkpoint(init_checkpoint["config"])
        codec = TextCodec.from_dict(init_checkpoint.get("codec"))
    else:
        codec = None
        config = None

    if args.instruct_jsonl:
        dataset_text = load_instruction_jsonl(args.instruct_jsonl)
    elif args.text:
        dataset_text = Path(args.text).read_text(encoding="utf-8")
    else:
        raise ValueError("provide --text or --instruct-jsonl")

    if codec is None:
        tokens, codec = encode_dataset_text(dataset_text, args.mode, args.tokenizer_vocab_size)
    else:
        tokens = encode_text_with_codec(dataset_text, codec)

    train_tokens, val_tokens = split_tokens(tokens, args.val_fraction)
    tokens_per_step = args.batch_size * args.seq_len
    planned_tokens = args.steps * tokens_per_step
    planned_epochs = planned_tokens / len(train_tokens)
    if config is None:
        config = TinyLanguageModelConfig(
            vocab_size=codec.vocab_size,
            dim=args.dim,
            layers=args.layers,
            conv_kernel=args.conv_kernel,
            dropout=args.dropout,
            architecture=args.architecture,
            recurrent=args.recurrent,
            state_kernel=args.state_kernel,
            max_dilation=args.max_dilation,
            mlp_expansion=args.mlp_expansion,
            state_heads=args.state_heads,
            state_rank=args.state_rank,
        )
    model = TinyLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    if init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model"])
        print(f"initialized={args.init_checkpoint} step={init_checkpoint.get('step', 0)}")

    start_step = 0
    loss_value = float("nan")
    val_loss = None
    if args.resume and Path(args.out).exists():
        checkpoint = torch.load(args.out, map_location=device)
        checkpoint_config = config_from_checkpoint(checkpoint["config"])
        checkpoint_codec = TextCodec.from_dict(checkpoint.get("codec"))
        if checkpoint_config != config:
            raise ValueError("checkpoint config does not match requested model settings")
        if checkpoint_codec != codec:
            raise ValueError("checkpoint codec does not match requested dataset/mode")
        start_step, loss_value, val_loss = load_checkpoint(args.out, model, optimizer, device)
        print(f"resumed={args.out} step={start_step}")

    print(
        " ".join(
            [
                f"device={device}",
                f"mode={codec.mode}",
                f"vocab={codec.vocab_size}",
                f"parameters={count_parameters(model):,}",
                f"train_tokens={len(train_tokens):,}",
                f"val_tokens={len(val_tokens):,}",
                f"tokens_per_step={tokens_per_step:,}",
                f"planned_train_tokens={planned_tokens:,}",
                f"planned_epochs={planned_epochs:.2f}",
            ]
        )
    )

    start = time.time()
    last = start

    for step in range(start_step + 1, args.steps + 1):
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = sample_batch(train_tokens, args.batch_size, args.seq_len, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = float(loss.item())
        if step % args.log_every == 0 or step == 1:
            now = time.time()
            reported_steps = args.log_every if step != 1 else 1
            steps_per_sec = reported_steps / max(now - last, 1e-9)
            tokens_per_sec = steps_per_sec * tokens_per_step
            epochs = (step * tokens_per_step) / len(train_tokens)
            last = now
            print(
                f"step={step} train_loss={loss_value:.4f} lr={lr:.2e} "
                f"steps_per_sec={steps_per_sec:.2f} tokens_per_sec={tokens_per_sec:,.0f} "
                f"epochs={epochs:.2f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = estimate_loss(
                model,
                val_tokens,
                args.batch_size,
                args.seq_len,
                device,
                args.eval_batches,
            )
            print(f"step={step} val_loss={val_loss:.4f}")

        if step % args.sample_every == 0 or step == args.steps:
            print("\n--- sample ---")
            print(
                sample_text(
                    model,
                    codec,
                    args.sample_prompt,
                    args.sample_tokens,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    args.repetition_penalty,
                    args.repetition_window,
                    args.seq_len,
                )
            )
            print("--- end sample ---\n")

        if step % args.save_every == 0:
            save_checkpoint(args.out, model, optimizer, step, loss_value, val_loss, codec, args)

    save_checkpoint(args.out, model, optimizer, args.steps, loss_value, val_loss, codec, args)
    elapsed = time.time() - start
    trained_tokens = max(0, args.steps - start_step) * tokens_per_step
    total_seen_tokens = args.steps * tokens_per_step
    total_epochs = total_seen_tokens / len(train_tokens)
    print(
        f"saved={args.out} elapsed_sec={elapsed:.1f} "
        f"trained_tokens_this_run={trained_tokens:,} total_seen_tokens={total_seen_tokens:,} "
        f"total_epochs={total_epochs:.2f}"
    )


if __name__ == "__main__":
    main()
