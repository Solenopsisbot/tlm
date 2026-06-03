from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

BASE_SOURCES = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "split": "train",
        "field": "text",
    },
    "cosmopedia": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "cosmopedia-v2",
        "split": "train",
        "field": "text",
    },
    "fineweb_edu_dedup": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "fineweb-edu-dedup",
        "split": "train",
        "field": "text",
    },
    "openwebmath": {
        "path": "open-web-math/open-web-math",
        "split": "train",
        "field": "text",
    },
}

SFT_SOURCES = {
    "openhermes": {
        "path": "teknium/OpenHermes-2.5",
        "split": "train",
    },
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
    },
    "no_robots": {
        "path": "HuggingFaceH4/no_robots",
        "split": "train",
    },
}


def stream_dataset(spec: dict[str, str]) -> Iterable[dict[str, Any]]:
    kwargs = {"split": spec["split"], "streaming": True}
    if "name" in spec:
        return load_dataset(spec["path"], spec["name"], **kwargs)
    return load_dataset(spec["path"], **kwargs)


def row_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field, "")
    if value is None:
        return ""
    return str(value).strip()


def normalize_role(role: str) -> str:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def messages_to_instruction(row: dict[str, Any]) -> dict[str, str] | None:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list):
        return None

    system = ""
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(str(message.get("role") or message.get("from") or ""))
        content = str(message.get("content") or message.get("value") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content
        else:
            turns.append((role, content))

    for index in range(len(turns) - 1, 0, -1):
        if turns[index][0] == "assistant" and turns[index - 1][0] == "user":
            result = {
                "instruction": turns[index - 1][1],
                "output": turns[index][1],
            }
            if system:
                result["system"] = system
            return result
    return None


def row_instruction(row: dict[str, Any]) -> dict[str, str] | None:
    direct = messages_to_instruction(row)
    if direct:
        return direct

    instruction = (
        row.get("instruction")
        or row.get("prompt")
        or row.get("question")
        or row.get("input")
    )
    output = row.get("output") or row.get("response") or row.get("answer")
    if instruction and output:
        return {"instruction": str(instruction), "output": str(output)}
    return None


def write_base(args: argparse.Namespace) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sources = args.sources or ["tinystories", "cosmopedia", "fineweb_edu_dedup", "openwebmath"]
    written = 0
    with Path(args.out).open("w", encoding="utf-8") as file:
        for source in sources:
            spec = BASE_SOURCES[source]
            count = 0
            for row in stream_dataset(spec):
                text = row_text(row, spec["field"])
                if len(text) >= args.min_chars:
                    file.write(text)
                    file.write("\n\n")
                    count += 1
                    written += 1
                if count >= args.docs_per_source:
                    break
            print(f"source={source} docs={count}")
    print(f"saved={args.out} docs={written}")


def write_sft(args: argparse.Namespace) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sources = args.sources or ["openhermes", "ultrachat", "no_robots"]
    written = 0
    with Path(args.out).open("w", encoding="utf-8") as file:
        for source in sources:
            spec = SFT_SOURCES[source]
            count = 0
            for row in stream_dataset(spec):
                example = row_instruction(row)
                if example:
                    file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                    written += 1
                if count >= args.examples_per_source:
                    break
            print(f"source={source} examples={count}")
    print(f"saved={args.out} examples={written}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local training files from HF datasets.")
    subparsers = parser.add_subparsers(dest="kind", required=True)

    base = subparsers.add_parser("base", help="Write plain text base-pretraining data.")
    base.add_argument("--out", required=True)
    base.add_argument("--sources", nargs="+", choices=sorted(BASE_SOURCES))
    base.add_argument("--docs-per-source", type=int, default=25_000)
    base.add_argument("--min-chars", type=int, default=200)
    base.set_defaults(func=write_base)

    sft = subparsers.add_parser("sft", help="Write instruction JSONL data.")
    sft.add_argument("--out", required=True)
    sft.add_argument("--sources", nargs="+", choices=sorted(SFT_SOURCES))
    sft.add_argument("--examples-per-source", type=int, default=25_000)
    sft.set_defaults(func=write_sft)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
