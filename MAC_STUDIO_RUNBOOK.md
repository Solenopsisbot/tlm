# Mac Studio Runbook

These commands assume the repo is on the M3 Ultra Mac Studio and you are in the
repo root.

## 1. Install

```bash
uv sync --extra dev
```

Optional Hugging Face cache location:

```bash
export HF_HOME="$PWD/.hf-cache"
export HF_DATASETS_CACHE="$PWD/.hf-cache/datasets"
```

## 2. Prepare Local Data

Small sanity data:

```bash
uv run python -m tlm.prepare_hf_data base \
  --out data/base_small.txt \
  --docs-per-source 1000 \
  --sources tinystories cosmopedia fineweb_edu_dedup openwebmath
```

First serious base mix:

```bash
uv run python -m tlm.prepare_hf_data base \
  --out data/base_mix.txt \
  --docs-per-source 25000 \
  --sources tinystories cosmopedia fineweb_edu_dedup openwebmath
```

Instruction mix:

```bash
uv run python -m tlm.prepare_hf_data sft \
  --out data/sft_mix.jsonl \
  --examples-per-source 25000 \
  --sources openhermes ultrachat no_robots
```

Reasoning curriculum:

```bash
uv run python -m tlm.curriculum \
  --out data/reason_add_1digit.jsonl \
  --stage add_1digit \
  --count 50000 \
  --chain-of-thought

uv run python -m tlm.curriculum \
  --out data/reason_addsub_2digit.jsonl \
  --stage addsub_2digit \
  --count 50000 \
  --chain-of-thought

uv run python -m tlm.curriculum \
  --out data/reason_mul_1digit.jsonl \
  --stage mul_1digit \
  --count 50000 \
  --chain-of-thought
```

## 3. Smoke Train

Run this first to make sure the machine/environment is happy:

```bash
uv run python -m tlm.train \
  --text data/base_small.txt \
  --out checkpoints/base_smoke.pt \
  --device mps \
  --steps 1000 \
  --mode bpe \
  --tokenizer-vocab-size 4096 \
  --architecture stream \
  --dim 256 \
  --layers 6 \
  --seq-len 1024 \
  --batch-size 16
```

Sample:

```bash
uv run python -m tlm.sample \
  --checkpoint checkpoints/base_smoke.pt \
  --prompt "Explain why the sky is blue." \
  --tokens 200
```

## 4. First Real Base Model

Recommended first conversational target:

```bash
uv run python -m tlm.train \
  --text data/base_mix.txt \
  --out checkpoints/base_384x8.pt \
  --device mps \
  --steps 50000 \
  --mode bpe \
  --tokenizer-vocab-size 8192 \
  --architecture stream \
  --dim 384 \
  --layers 8 \
  --seq-len 2048 \
  --batch-size 16 \
  --eval-every 1000 \
  --sample-every 2000
```

If memory allows, increase `--batch-size` to `24` or `32`.

## 5. General Instruction Tune

```bash
uv run python -m tlm.train \
  --instruct-jsonl data/sft_mix.jsonl \
  --init-checkpoint checkpoints/base_384x8.pt \
  --out checkpoints/sft_general_384x8.pt \
  --device mps \
  --steps 10000 \
  --seq-len 2048 \
  --batch-size 16 \
  --eval-every 1000 \
  --sample-every 2000
```

## 6. Reasoning Curriculum SFT

```bash
uv run python -m tlm.train \
  --instruct-jsonl data/reason_add_1digit.jsonl \
  --init-checkpoint checkpoints/sft_general_384x8.pt \
  --out checkpoints/reason_add_1digit.pt \
  --device mps \
  --steps 3000 \
  --seq-len 1024 \
  --batch-size 32

uv run python -m tlm.train \
  --instruct-jsonl data/reason_addsub_2digit.jsonl \
  --init-checkpoint checkpoints/reason_add_1digit.pt \
  --out checkpoints/reason_addsub_2digit.pt \
  --device mps \
  --steps 3000 \
  --seq-len 1024 \
  --batch-size 32

uv run python -m tlm.train \
  --instruct-jsonl data/reason_mul_1digit.jsonl \
  --init-checkpoint checkpoints/reason_addsub_2digit.pt \
  --out checkpoints/reason_mul_1digit.pt \
  --device mps \
  --steps 3000 \
  --seq-len 1024 \
  --batch-size 32
```

Evaluate:

```bash
uv run python -m tlm.eval_arithmetic \
  --checkpoint checkpoints/reason_mul_1digit.pt \
  --count 200 \
  --max-value 99 \
  --device cpu
```

## 7. Reward-Filtered RL-Style Loop

Generate candidates, keep exact-match winners:

```bash
uv run python -m tlm.rl_arithmetic \
  --checkpoint checkpoints/reason_mul_1digit.pt \
  --out data/rl_winners.jsonl \
  --scored-out data/rl_scored.jsonl \
  --problems 5000 \
  --candidates 16 \
  --max-value 99 \
  --device cpu
```

Fine-tune on winners:

```bash
uv run python -m tlm.train \
  --instruct-jsonl data/rl_winners.jsonl \
  --init-checkpoint checkpoints/reason_mul_1digit.pt \
  --out checkpoints/reason_rl_rft.pt \
  --device mps \
  --steps 3000 \
  --seq-len 1024 \
  --batch-size 32
```

Re-evaluate:

```bash
uv run python -m tlm.eval_arithmetic \
  --checkpoint checkpoints/reason_rl_rft.pt \
  --count 500 \
  --max-value 99 \
  --device cpu
```

## 8. One-Command Pipeline

To print the full staged pipeline:

```bash
uv run python -m tlm.run_pipeline \
  --base-text data/base_mix.txt \
  --device mps \
  --dim 384 \
  --layers 8 \
  --seq-len 2048 \
  --batch-size 16
```

To run it:

```bash
uv run python -m tlm.run_pipeline \
  --base-text data/base_mix.txt \
  --device mps \
  --dim 384 \
  --layers 8 \
  --seq-len 2048 \
  --batch-size 16 \
  --run
```
