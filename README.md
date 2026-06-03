# Tiny Language Model

`tlm` is a deliberately small non-transformer language model. It is designed to
train quickly on ordinary text without a tokenizer, attention cache, or large
framework around it.

The default training path is token-level byte-pair encoding with a compact
stream-memory model:

- **BPE vocabulary:** dataset-trained byte-level tokenizer, robust decoding.
- **Stream memory:** fixed-size weighted running memory for unbounded generation.
- **Parallel training:** sequence memory is computed with cumulative sums, not
  attention matrices.
- **Constant generation cost:** generation state is fixed-size and does not grow
  with output length.
- **Fallbacks:** optimized GRU and a custom gated stack are still available with
  `--architecture rnn`.
- **Tied embeddings:** the input embedding matrix is reused as the output head.
- **Training harness:** validation loss, resume, cosine learning-rate decay, and
  periodic samples.

This is not trying to be a miniature GPT. The bias is toward something easy to
understand, cheap to train, compact on disk, and useful as a playground for
custom small-model ideas.

## Install

```bash
uv sync --extra dev
```

## Train

```bash
uv run python -m tlm.train --text data.txt --steps 1000 --device cpu
```

Useful small-model knobs:

```bash
uv run python -m tlm.train \
  --text data.txt \
  --mode bpe \
  --tokenizer-vocab-size 4096 \
  --architecture stream \
  --dim 256 \
  --layers 8 \
  --seq-len 16384 \
  --batch-size 2 \
  --eval-every 250 \
  --sample-every 500
```

Resume a run:

```bash
uv run python -m tlm.train --text data.txt --out checkpoints/tlm.pt --resume
```

Tiny Shakespeare example:

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/tinyshakespeare.txt

uv run python -m tlm.train \
  --text data/tinyshakespeare.txt \
  --out checkpoints/tinyshakespeare-char.pt \
  --device mps \
  --steps 5000 \
  --mode bpe \
  --tokenizer-vocab-size 4096 \
  --architecture stream \
  --dim 256 \
  --layers 8 \
  --seq-len 1024
```

16k context experiment:

```bash
uv run python -m tlm.train \
  --text data/tinyshakespeare.txt \
  --out checkpoints/tinyshakespeare-stream-16k.pt \
  --device mps \
  --steps 1000 \
  --mode bpe \
  --tokenizer-vocab-size 4096 \
  --architecture stream \
  --dim 256 \
  --layers 8 \
  --seq-len 16384 \
  --batch-size 2 \
  --eval-batches 2
```

## Sample

```bash
uv run python -m tlm.sample --checkpoint checkpoints/tlm.pt --prompt "Once upon a"
```

For `stream`, `swift`, and `fastconv` checkpoints, `--device auto` uses CPU for
generation. On Apple Silicon, cached one-token generation is much faster on CPU
than MPS because MPS pays heavy overhead for many tiny per-token kernels.

## Instruction Tuning

Instruction data is JSONL with fields such as `instruction`/`output`,
`prompt`/`response`, or `question`/`answer`.

Generate a synthetic arithmetic instruction set:

```bash
uv run python -m tlm.instruct \
  --out data/arithmetic_train.jsonl \
  --count 5000 \
  --max-value 99 \
  --chain-of-thought
```

Train on instruction JSONL:

```bash
uv run python -m tlm.train \
  --instruct-jsonl data/arithmetic_train.jsonl \
  --out checkpoints/arithmetic-stream-bpe.pt \
  --device mps \
  --steps 1000 \
  --mode bpe \
  --architecture stream
```

Evaluate arithmetic exact-match accuracy:

```bash
uv run python -m tlm.eval_arithmetic \
  --checkpoint checkpoints/arithmetic-stream-bpe.pt \
  --count 100 \
  --max-value 99
```

Generate reward-filtered samples for a simple RL-style improvement loop:

```bash
uv run python -m tlm.rl_arithmetic \
  --checkpoint checkpoints/arithmetic-stream-bpe.pt \
  --out data/rl_winners.jsonl \
  --scored-out data/rl_scored.jsonl \
  --problems 2000 \
  --candidates 8
```

Print the full staged pipeline:

```bash
uv run python -m tlm.run_pipeline
```

Run it:

```bash
uv run python -m tlm.run_pipeline --run
```

Suggested model sizes:

- **Smoke:** `dim=128`, `layers=4`, about 1M params.
- **First chat-capable target:** `dim=384`, `layers=8`, about 15M-25M params.
- **Serious small model:** `dim=512`, `layers=12`, about 50M-80M params.
- **Ambitious local model:** `dim=768`, `layers=16`, likely 150M+ params.

For a model you can actually talk to, start with `dim=384`, `layers=8`,
`seq_len=2048`, `tokenizer_vocab_size=8192`, then scale only after evals improve.

## Why This Shape?

Transformers are powerful, but attention is expensive and architecturally heavy
for very small experiments. This model keeps the things tiny LMs need most:
fast stream memory, long-range parallel mixing, and cached generation. The
result is a small, trainable baseline that can later absorb more custom ideas:
tokenizers, sparse experts, learned forgetting, neural cache, or fused recurrent
generation kernels.
