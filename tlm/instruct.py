from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

BEGIN_USER = "<|user|>"
BEGIN_ASSISTANT = "<|assistant|>"
BEGIN_SYSTEM = "<|system|>"
BEGIN_ROLE = "<|role|>"
BEGIN_CONTENT = "<|content|>"
END = "<|end|>"
BEGIN_THINK = "<think>"
END_THINK = "</think>"
BEGIN_REFLECT = "<reflect>"
END_REFLECT = "</reflect>"
BEGIN_ANSWER = "<answer>"
END_ANSWER = "</answer>"


def render_example(example: dict[str, Any]) -> str:
    system = str(example.get("system", "")).strip()
    instruction = str(
        example.get("instruction")
        or example.get("prompt")
        or example.get("question")
        or example.get("input")
        or ""
    ).strip()
    extra_input = str(example.get("input", "")).strip()
    response = str(
        example.get("output")
        or example.get("response")
        or example.get("answer")
        or ""
    ).strip()

    if "instruction" in example and extra_input and extra_input != instruction:
        instruction = f"{instruction}\n\n{extra_input}"
    if not instruction:
        raise ValueError("instruction example is missing an instruction/prompt/question")
    if not response:
        raise ValueError("instruction example is missing an output/response/answer")

    parts = []
    if system:
        parts.append(render_message("system", system))
    parts.append(render_message("user", instruction))
    parts.append(render_message("assistant", response))
    return "\n".join(parts) + "\n"


def render_message(role: str, content: str, close: bool = True) -> str:
    text = f"{BEGIN_ROLE}\n{role.strip()}\n{BEGIN_CONTENT}\n{content.strip()}"
    if close:
        text += f"\n{END}"
    return text


def render_reasoned_response(
    reasoning: str,
    answer: str | int,
    answer_style: str = "plain",
) -> str:
    think = f"{BEGIN_THINK}\n{str(reasoning).strip()}\n{END_THINK}"
    if answer_style == "tag":
        return f"{think}\n{BEGIN_ANSWER}\n{answer}\n{END_ANSWER}"
    if answer_style != "plain":
        raise ValueError(f"unknown answer style: {answer_style}")
    return f"{think}\n{answer}"


def render_prompt(instruction: str, system: str = "") -> str:
    parts = []
    if system.strip():
        parts.append(render_message("system", system))
    parts.append(render_message("user", instruction))
    parts.append(render_message("assistant", "", close=False))
    return "\n".join(parts)


def load_instruction_jsonl(path: str | Path) -> str:
    rendered = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            rendered.append(render_example(example))
    if not rendered:
        raise ValueError("instruction file did not contain any examples")
    return "\n".join(rendered)


def arithmetic_example(rng: random.Random, max_value: int, chain_of_thought: bool) -> dict[str, str]:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(0, max_value)
    b = rng.randint(0, max_value)
    if op == "+":
        answer = a + b
        reasoning = f"{a} + {b} = {answer}."
    elif op == "-":
        if b > a:
            a, b = b, a
        answer = a - b
        reasoning = f"{a} - {b} = {answer}."
    else:
        answer = a * b
        reasoning = f"{a} * {b} = {answer}."

    output = render_reasoned_response(reasoning, answer) if chain_of_thought else f"{BEGIN_ANSWER}\n{answer}\n{END_ANSWER}"
    return {
        "instruction": f"Solve this arithmetic problem. Give the final answer.\n\n{a} {op} {b}",
        "output": output,
    }


def make_arithmetic_jsonl(
    path: str | Path,
    count: int,
    max_value: int,
    seed: int,
    chain_of_thought: bool,
) -> None:
    rng = random.Random(seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        for _ in range(count):
            file.write(json.dumps(arithmetic_example(rng, max_value, chain_of_thought)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic instruction data.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--max-value", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chain-of-thought", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_arithmetic_jsonl(
        args.out,
        count=args.count,
        max_value=args.max_value,
        seed=args.seed,
        chain_of_thought=args.chain_of_thought,
    )
    print(f"saved={args.out} examples={args.count}")


if __name__ == "__main__":
    main()
