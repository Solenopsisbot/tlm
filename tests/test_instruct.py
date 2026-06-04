import json

from tlm.instruct import (
    END,
    BEGIN_ANSWER,
    BEGIN_CONTENT,
    BEGIN_REFLECT,
    BEGIN_ROLE,
    BEGIN_THINK,
    load_instruction_jsonl,
    make_arithmetic_jsonl,
    render_reasoned_response,
    render_example,
    render_prompt,
)
from tlm.eval_arithmetic import extract_answer


def test_render_example() -> None:
    text = render_example({"instruction": "Say hi.", "output": "Hi."})

    assert text.count(BEGIN_ROLE) == 2
    assert text.count(BEGIN_CONTENT) == 2
    assert "user" in text
    assert "assistant" in text
    assert text.count(END) == 2
    assert "Say hi." in text
    assert "Hi." in text


def test_render_prompt_leaves_assistant_open() -> None:
    text = render_prompt("Solve 2 + 2.")

    assert text.endswith(f"{BEGIN_ROLE}\nassistant\n{BEGIN_CONTENT}\n")


def test_make_and_load_arithmetic_jsonl(tmp_path) -> None:
    path = tmp_path / "math.jsonl"
    make_arithmetic_jsonl(path, count=3, max_value=10, seed=1, chain_of_thought=True)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rendered = load_instruction_jsonl(path)

    assert len(rows) == 3
    assert "</think>" in rendered


def test_extract_answer_prefers_first_answer_before_end() -> None:
    response = "Add the two numbers: 9 + 1 = 10.\nAnswer: 10\n<|end|>\nAnswer: 6"

    assert extract_answer(response) == 10


def test_render_reasoned_response_uses_tags() -> None:
    text = render_reasoned_response("2 + 2 = 4.", 4)

    assert BEGIN_THINK in text
    assert BEGIN_ANSWER not in text
    assert text.endswith("</think>\n4")


def test_render_reasoned_response_can_use_answer_tag() -> None:
    text = render_reasoned_response("2 + 2 = 4.", 4, answer_style="tag")

    assert "<answer>\n4\n</answer>" in text


def test_extract_answer_prefers_answer_tag() -> None:
    response = "<think>\nwrong number 99\n</think>\n<answer>\n4\n</answer>"

    assert extract_answer(response) == 4


def test_extract_answer_after_think_for_plain_answer() -> None:
    response = "<think>\nwrong number 99\n</think>\n4"

    assert extract_answer(response) == 4
