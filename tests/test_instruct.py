import json

from tlm.instruct import (
    BEGIN_ASSISTANT,
    BEGIN_USER,
    END,
    load_instruction_jsonl,
    make_arithmetic_jsonl,
    render_example,
    render_prompt,
)


def test_render_example() -> None:
    text = render_example({"instruction": "Say hi.", "output": "Hi."})

    assert BEGIN_USER in text
    assert BEGIN_ASSISTANT in text
    assert text.count(END) == 2
    assert "Say hi." in text
    assert "Hi." in text


def test_render_prompt_leaves_assistant_open() -> None:
    text = render_prompt("Solve 2 + 2.")

    assert text.endswith(f"{BEGIN_ASSISTANT}\n")


def test_make_and_load_arithmetic_jsonl(tmp_path) -> None:
    path = tmp_path / "math.jsonl"
    make_arithmetic_jsonl(path, count=3, max_value=10, seed=1, chain_of_thought=True)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rendered = load_instruction_jsonl(path)

    assert len(rows) == 3
    assert "Answer:" in rendered
