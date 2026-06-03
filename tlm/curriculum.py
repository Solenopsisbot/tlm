from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


STAGES = {
    "add_1digit": {"ops": ["+"], "max": 9},
    "addsub_2digit": {"ops": ["+", "-"], "max": 99},
    "mul_1digit": {"ops": ["*"], "max": 9},
    "mixed_2digit": {"ops": ["+", "-", "*"], "max": 99},
    "mixed_3digit": {"ops": ["+", "-", "*"], "max": 999},
}


def solve(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unknown operator: {op}")


def reasoning(a: int, op: str, b: int, answer: int) -> str:
    if op == "+":
        return f"Add the two numbers: {a} + {b} = {answer}."
    if op == "-":
        return f"Subtract the second number from the first: {a} - {b} = {answer}."
    return f"Multiply the two numbers: {a} * {b} = {answer}."


def make_example(rng: random.Random, stage: str, chain_of_thought: bool) -> dict[str, str]:
    spec = STAGES[stage]
    op = rng.choice(spec["ops"])
    a = rng.randint(0, spec["max"])
    b = rng.randint(0, spec["max"])
    if op == "-" and b > a:
        a, b = b, a
    answer = solve(a, op, b)
    if chain_of_thought:
        output = f"{reasoning(a, op, b, answer)}\nAnswer: {answer}"
    else:
        output = f"Answer: {answer}"
    return {
        "instruction": f"Solve this arithmetic problem. Give the final answer.\n\n{a} {op} {b}",
        "output": output,
        "stage": stage,
    }


def write_stage(
    path: str | Path,
    stage: str,
    count: int,
    seed: int,
    chain_of_thought: bool,
) -> None:
    rng = random.Random(seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        for _ in range(count):
            file.write(json.dumps(make_example(rng, stage, chain_of_thought)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate staged arithmetic curriculum JSONL.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="add_1digit")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chain-of-thought", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_stage(args.out, args.stage, args.count, args.seed, args.chain_of_thought)
    print(f"saved={args.out} stage={args.stage} examples={args.count}")


if __name__ == "__main__":
    main()
