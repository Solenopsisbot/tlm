from __future__ import annotations

import argparse
import re
import random

import torch

from .curriculum import STAGES, solve
from .data import TextCodec
from .instruct import render_prompt
from .model import TinyLanguageModel, TinyLanguageModelConfig
from .sample import config_from_checkpoint


ANSWER_RE = re.compile(r"-?\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate arithmetic exact-match accuracy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--ops", nargs="+", choices=["+", "-", "*"])
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def make_problem(
    rng: random.Random,
    max_value: int,
    ops: list[str] | None = None,
) -> tuple[str, int]:
    op = rng.choice(ops or ["+", "-", "*"])
    a = rng.randint(0, max_value)
    b = rng.randint(0, max_value)
    if op == "-" and b > a:
        a, b = b, a
    answer = solve(a, op, b)
    prompt = f"Solve this arithmetic problem. Give the final answer.\n\n{a} {op} {b}"
    return prompt, answer


def extract_answer(text: str) -> int | None:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


@torch.no_grad()
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = config_from_checkpoint(checkpoint["config"])
    codec = TextCodec.from_dict(checkpoint.get("codec"))
    model = TinyLanguageModel(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    correct = 0
    if args.stage:
        stage = STAGES[args.stage]
        ops = list(stage["ops"])
        max_value = int(stage["max"])
    else:
        ops = args.ops
        max_value = args.max_value
    for index in range(args.count):
        problem, expected = make_problem(rng, max_value, ops)
        prompt = render_prompt(problem)
        prompt_tokens = codec.encode(prompt)
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=args.tokens,
            temperature=max(args.temperature, 1e-6),
            top_k=args.top_k,
        )
        response = codec.decode(generated[len(prompt_tokens) :])
        actual = extract_answer(response)
        is_correct = actual == expected
        correct += int(is_correct)
        if index < 5:
            print(f"problem={problem.splitlines()[-1]!r} expected={expected} actual={actual} correct={is_correct}")

    accuracy = correct / max(args.count, 1)
    print(f"accuracy={accuracy:.4f} correct={correct} total={args.count}")


if __name__ == "__main__":
    main()
