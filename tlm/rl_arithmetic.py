from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .data import TextCodec
from .eval_arithmetic import extract_answer, make_problem
from .instruct import render_prompt
from .model import TinyLanguageModel
from .sample import config_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reward-filtered arithmetic SFT data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scored-out")
    parser.add_argument("--problems", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored_file = None
    if args.scored_out:
        Path(args.scored_out).parent.mkdir(parents=True, exist_ok=True)
        scored_file = Path(args.scored_out).open("w", encoding="utf-8")

    kept = 0
    with Path(args.out).open("w", encoding="utf-8") as out_file:
        for _ in range(args.problems):
            problem, expected = make_problem(rng, args.max_value)
            prompt = render_prompt(problem)
            prompt_tokens = codec.encode(prompt)
            best_response = None
            best_reward = 0
            for _ in range(args.candidates):
                generated = model.generate(
                    prompt_tokens,
                    max_new_tokens=args.tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                response = codec.decode(generated[len(prompt_tokens) :])
                actual = extract_answer(response)
                reward = int(actual == expected)
                if scored_file:
                    scored_file.write(
                        json.dumps(
                            {
                                "instruction": problem,
                                "response": response,
                                "expected": expected,
                                "actual": actual,
                                "reward": reward,
                            }
                        )
                        + "\n"
                    )
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
            if best_reward and best_response:
                out_file.write(
                    json.dumps(
                        {
                            "instruction": problem,
                            "output": best_response.strip(),
                            "reward": best_reward,
                        }
                    )
                    + "\n"
                )
                kept += 1

    if scored_file:
        scored_file.close()
    print(f"saved={args.out} kept={kept} problems={args.problems} keep_rate={kept / max(args.problems, 1):.4f}")


if __name__ == "__main__":
    main()
