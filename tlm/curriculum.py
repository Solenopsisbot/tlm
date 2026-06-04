from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Literal

from .instruct import render_reasoned_response

ReasoningStyle = Literal["direct", "column", "structured"]


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


def direct_reasoning(a: int, op: str, b: int, answer: int) -> str:
    if op == "+":
        return f"Add the two numbers: {a} + {b} = {answer}."
    if op == "-":
        return f"Subtract the second number from the first: {a} - {b} = {answer}."
    return f"Multiply the two numbers: {a} * {b} = {answer}."


def digits2(value: int) -> tuple[int, int]:
    return divmod(value, 10)


def column_add_reasoning(a: int, b: int, answer: int) -> str:
    a_tens, a_ones = digits2(a)
    b_tens, b_ones = digits2(b)
    ones_total = a_ones + b_ones
    ones_digit = ones_total % 10
    carry = ones_total // 10
    tens_total = a_tens + b_tens + carry
    lines = [
        f"Add the ones digits: {a_ones} + {b_ones} = {ones_total}.",
    ]
    if carry:
        lines.append(f"Write down {ones_digit} in the ones place and carry {carry}.")
    else:
        lines.append(f"Write down {ones_digit} in the ones place and carry 0.")
    lines.append(f"Add the tens digits and the carry: {a_tens} + {b_tens} + {carry} = {tens_total}.")
    lines.append(f"So {a} + {b} = {answer}.")
    return "\n".join(lines)


def column_sub_reasoning(a: int, b: int, answer: int) -> str:
    a_tens, a_ones = digits2(a)
    b_tens, b_ones = digits2(b)
    if a_ones >= b_ones:
        ones_result = a_ones - b_ones
        tens_result = a_tens - b_tens
        lines = [
            f"Subtract the ones digits: {a_ones} - {b_ones} = {ones_result}.",
            "No borrowing is needed.",
            f"Subtract the tens digits: {a_tens} - {b_tens} = {tens_result}.",
        ]
    else:
        borrowed_ones = a_ones + 10
        ones_result = borrowed_ones - b_ones
        reduced_tens = a_tens - 1
        tens_result = reduced_tens - b_tens
        lines = [
            f"The ones digit {a_ones} is smaller than {b_ones}, so borrow 1 ten.",
            f"Now the ones subtraction is {borrowed_ones} - {b_ones} = {ones_result}.",
            f"The tens digit becomes {reduced_tens}.",
            f"Subtract the tens digits: {reduced_tens} - {b_tens} = {tens_result}.",
        ]
    lines.append(f"So {a} - {b} = {answer}.")
    return "\n".join(lines)


def structured_add_reasoning(a: int, b: int, answer: int) -> str:
    a_tens, a_ones = digits2(a)
    b_tens, b_ones = digits2(b)
    ones_sum = a_ones + b_ones
    ones_digit = ones_sum % 10
    carry = ones_sum // 10
    tens_sum = a_tens + b_tens + carry
    return "\n".join(
        [
            f"problem = {a} + {b}",
            f"a = {a}",
            f"b = {b}",
            f"a_tens = {a_tens}",
            f"a_ones = {a_ones}",
            f"b_tens = {b_tens}",
            f"b_ones = {b_ones}",
            f"ones_sum = a_ones + b_ones = {a_ones} + {b_ones} = {ones_sum}",
            f"ones_digit = {ones_digit}",
            f"carry = {carry}",
            f"tens_sum = a_tens + b_tens + carry = {a_tens} + {b_tens} + {carry} = {tens_sum}",
            f"result = {answer}",
        ]
    )


def structured_sub_reasoning(a: int, b: int, answer: int) -> str:
    a_tens, a_ones = digits2(a)
    b_tens, b_ones = digits2(b)
    borrow = int(a_ones < b_ones)
    adjusted_ones = a_ones + 10 * borrow
    ones_digit = adjusted_ones - b_ones
    adjusted_tens = a_tens - borrow
    tens_digit = adjusted_tens - b_tens
    return "\n".join(
        [
            f"problem = {a} - {b}",
            f"a = {a}",
            f"b = {b}",
            f"a_tens = {a_tens}",
            f"a_ones = {a_ones}",
            f"b_tens = {b_tens}",
            f"b_ones = {b_ones}",
            f"borrow = {borrow}",
            f"adjusted_ones = a_ones + 10 * borrow = {a_ones} + 10 * {borrow} = {adjusted_ones}",
            f"ones_digit = adjusted_ones - b_ones = {adjusted_ones} - {b_ones} = {ones_digit}",
            f"adjusted_tens = a_tens - borrow = {a_tens} - {borrow} = {adjusted_tens}",
            f"tens_digit = adjusted_tens - b_tens = {adjusted_tens} - {b_tens} = {tens_digit}",
            f"result = {answer}",
        ]
    )


def reasoning(a: int, op: str, b: int, answer: int, style: ReasoningStyle) -> str:
    if style == "direct":
        return direct_reasoning(a, op, b, answer)
    if style == "structured":
        if op == "+" and a < 100 and b < 100:
            return structured_add_reasoning(a, b, answer)
        if op == "-" and a < 100 and b < 100:
            return structured_sub_reasoning(a, b, answer)
    if op == "+" and a < 100 and b < 100:
        return column_add_reasoning(a, b, answer)
    if op == "-" and a < 100 and b < 100:
        return column_sub_reasoning(a, b, answer)
    return direct_reasoning(a, op, b, answer)


def make_example(
    rng: random.Random,
    stage: str,
    chain_of_thought: bool,
    reasoning_style: ReasoningStyle = "direct",
) -> dict[str, str]:
    spec = STAGES[stage]
    op = rng.choice(spec["ops"])
    a = rng.randint(0, spec["max"])
    b = rng.randint(0, spec["max"])
    if op == "-" and b > a:
        a, b = b, a
    answer = solve(a, op, b)
    if chain_of_thought:
        output = render_reasoned_response(reasoning(a, op, b, answer, reasoning_style), answer)
    else:
        output = f"<answer>\n{answer}\n</answer>"
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
    reasoning_style: ReasoningStyle = "direct",
) -> None:
    rng = random.Random(seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        for _ in range(count):
            file.write(json.dumps(make_example(rng, stage, chain_of_thought, reasoning_style)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate staged arithmetic curriculum JSONL.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="add_1digit")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chain-of-thought", action="store_true")
    parser.add_argument("--reasoning-style", choices=["direct", "column", "structured"], default="direct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_stage(args.out, args.stage, args.count, args.seed, args.chain_of_thought, args.reasoning_style)
    print(f"saved={args.out} stage={args.stage} examples={args.count}")


if __name__ == "__main__":
    main()
