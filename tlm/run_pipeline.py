from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_or_print(cmd: list[str], run: bool) -> None:
    text = command(cmd)
    print(f"\n{text}")
    if run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print the staged TLM training pipeline.")
    parser.add_argument("--base-text", default="data/tinyshakespeare.txt")
    parser.add_argument("--workdir", default="runs/reasoning")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true", help="Actually execute commands.")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=8192)
    parser.add_argument("--base-steps", type=int, default=10_000)
    parser.add_argument("--sft-steps", type=int, default=2_000)
    parser.add_argument("--examples-per-stage", type=int, default=50_000)
    parser.add_argument("--rl-problems", type=int, default=2000)
    parser.add_argument("--rl-candidates", type=int, default=8)
    return parser.parse_args()


def train_cmd(
    out: str,
    device: str,
    steps: int,
    dim: int,
    layers: int,
    seq_len: int,
    batch_size: int,
    tokenizer_vocab_size: int,
    text: str | None = None,
    instruct_jsonl: str | None = None,
    init_checkpoint: str | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "tlm.train",
        "--out",
        out,
        "--device",
        device,
        "--steps",
        str(steps),
        "--mode",
        "bpe",
        "--tokenizer-vocab-size",
        str(tokenizer_vocab_size),
        "--architecture",
        "stream",
        "--dim",
        str(dim),
        "--layers",
        str(layers),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--eval-every",
        "500",
        "--sample-every",
        "1000",
    ]
    if text:
        cmd += ["--text", text]
    if instruct_jsonl:
        cmd += ["--instruct-jsonl", instruct_jsonl]
    if init_checkpoint:
        cmd += ["--init-checkpoint", init_checkpoint]
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(args.workdir)
    data_dir = root / "data"
    ckpt_dir = root / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = str(ckpt_dir / "base.pt")
    run_or_print(
        train_cmd(
            out=base_ckpt,
            device=args.device,
            steps=args.base_steps,
            dim=args.dim,
            layers=args.layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
            text=args.base_text,
        ),
        args.run,
    )

    previous = base_ckpt
    stages = ["add_1digit", "addsub_2digit", "mul_1digit", "mixed_2digit", "mixed_3digit"]
    for index, stage in enumerate(stages, start=1):
        data_path = str(data_dir / f"{index}_{stage}.jsonl")
        ckpt_path = str(ckpt_dir / f"sft_{index}_{stage}.pt")
        run_or_print(
            [
                "uv",
                "run",
                "python",
                "-m",
                "tlm.curriculum",
                "--out",
                data_path,
                "--stage",
                stage,
                "--count",
                str(args.examples_per_stage),
                "--seed",
                str(index),
                "--chain-of-thought",
            ],
            args.run,
        )
        run_or_print(
            train_cmd(
                out=ckpt_path,
                device=args.device,
                steps=args.sft_steps,
                dim=args.dim,
                layers=args.layers,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                tokenizer_vocab_size=args.tokenizer_vocab_size,
                instruct_jsonl=data_path,
                init_checkpoint=previous,
            ),
            args.run,
        )
        run_or_print(
            [
                "uv",
                "run",
                "python",
                "-m",
                "tlm.eval_arithmetic",
                "--checkpoint",
                ckpt_path,
                "--count",
                "200",
                "--max-value",
                "99",
                "--device",
                "cpu",
            ],
            args.run,
        )
        previous = ckpt_path

    rl_data = str(data_dir / "rl_winners.jsonl")
    rl_ckpt = str(ckpt_dir / "rl_rft.pt")
    run_or_print(
        [
            "uv",
            "run",
            "python",
            "-m",
            "tlm.rl_arithmetic",
            "--checkpoint",
            previous,
            "--out",
            rl_data,
            "--scored-out",
            str(data_dir / "rl_scored.jsonl"),
            "--problems",
            str(args.rl_problems),
            "--candidates",
            str(args.rl_candidates),
            "--max-value",
            "99",
            "--device",
            "cpu",
        ],
        args.run,
    )
    run_or_print(
        train_cmd(
            out=rl_ckpt,
            device=args.device,
            steps=args.sft_steps,
            dim=args.dim,
            layers=args.layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
            instruct_jsonl=rl_data,
            init_checkpoint=previous,
        ),
        args.run,
    )

    print("\nPipeline prepared.")


if __name__ == "__main__":
    main()
