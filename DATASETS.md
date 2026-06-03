# Dataset Plan

This project should train in stages. Do not start with huge instruction data
before the base model can model ordinary text.

## Stage 1: Base Pretraining

Recommended starter mix:

- `roneneldan/TinyStories`
  - Best first sanity corpus for small models.
  - Teaches coherent short English with simple vocabulary.
- `HuggingFaceTB/smollm-corpus`
  - Use `cosmopedia-v2` for synthetic educational text.
  - Use `fineweb-edu-dedup` for high-quality filtered web text.
- `HuggingFaceFW/fineweb-edu`
  - Huge educational web corpus. Stream subsets; do not download all at once.
- `open-web-math/open-web-math`
  - Math-heavy pretraining corpus, useful before reasoning SFT/RL.
- `HuggingFaceTB/finemath`
  - Math pretraining corpus. Use after the pipeline works on smaller mixes.

Suggested base mix for the first real conversational model:

```text
40% TinyStories / simple prose
35% Cosmopedia / educational synthetic text
15% FineWeb-Edu subset
10% OpenWebMath / FineMath subset
```

For a reasoning-heavy model:

```text
25% TinyStories / simple prose
30% Cosmopedia
20% FineWeb-Edu subset
25% OpenWebMath / FineMath subset
```

## Stage 2: Instruction Tuning

Good public SFT sources:

- `teknium/OpenHermes-2.5`
  - Broad instruction/chat mixture, about 1M samples.
- `HuggingFaceH4/ultrachat_200k`
  - Multi-turn conversational instruction data.
- `allenai/tulu-v2-sft-mixture`
  - Curated open SFT mixture.
- `Open-Orca/OpenOrca`
  - Large FLAN/Orca-style instruction data.
- `HuggingFaceH4/no_robots`
  - Small, human-written instruction set; useful as high-quality polishing data.

Suggested SFT mix:

```text
40% OpenHermes-2.5
25% UltraChat 200k
20% Tulu v2 SFT mixture
10% project synthetic arithmetic/curriculum
5% no_robots / high-quality small data
```

## Stage 3: Reasoning SFT

Use staged curriculum first, then public math reasoning data:

- project-generated arithmetic curriculum via `tlm.curriculum`
- `gsm8k`
- `meta-math/MetaMathQA`
- `nvidia/OpenMathInstruct-1`
- `open-r1/OpenR1-Math-220k`
- `PRM800K`

Suggested progression:

```text
1. add_1digit
2. addsub_2digit
3. mul_1digit
4. mixed_2digit
5. GSM8K-style word problems
6. MetaMathQA / OpenMathInstruct
7. PRM800K-style process data
```

## Stage 4: RL / Reward Filtering

Start with verifiable rewards:

- exact arithmetic answer
- unit tests for generated code
- JSON/schema validity
- string transformation exact match
- puzzle/game state validity

Use `tlm.rl_arithmetic` for the first rejection-sampling loop:

```bash
uv run python -m tlm.rl_arithmetic \
  --checkpoint runs/reasoning/checkpoints/sft_5_mixed_3digit.pt \
  --out runs/reasoning/data/rl_winners.jsonl \
  --scored-out runs/reasoning/data/rl_scored.jsonl \
  --problems 2000 \
  --candidates 8
```

Then fine-tune on winners with `--init-checkpoint`.

## Hardware Assignment

### M5 MacBook Pro, 24 GB RAM

Use for:

- smoke tests
- tokenizer experiments
- small SFT runs
- eval and generation

Good target:

```text
dim=128-256
layers=4-8
seq_len=512-2048
```

### Ryzen 7 3700X + GTX 970 + 32 GB DDR4

Use mostly as a CPU data machine. The GTX 970 is too old/small for serious
modern training.

Use for:

- dataset download/stream/filter
- JSONL generation
- CPU generation benchmarks
- eval jobs

### i9-12900K + 128 GB DDR5

Use for:

- dataset preprocessing
- CPU training experiments
- long generation/eval jobs
- tokenizer training on larger corpora

This is probably your best always-on orchestration/data box.

### Maxed M3 Ultra Mac Studio

Use for:

- main training
- 15M-100M parameter experiments
- long-context stream training
- SFT stages

Recommended first serious model:

```text
architecture=stream
mode=bpe
tokenizer_vocab_size=8192
dim=384
layers=8
seq_len=2048
batch_size=8-32, depending on memory
```

Recommended larger target:

```text
dim=512
layers=12
seq_len=4096
tokenizer_vocab_size=16384
```

Do not jump to the larger target until the 384x8 model improves on evals.
